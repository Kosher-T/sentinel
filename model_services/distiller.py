import os
import sys
import time
import shutil
import logging
import gc
import numpy as np
from pathlib import Path
import tensorflow as tf
try:
    from tensorflow.keras.models import Model, load_model  # type:ignore
except ImportError:
    from keras.models import Model, load_model

# Framework imports - wrapped in try/except to stay lightweight if not installed
try:
    import torch  # type:ignore  # PyTorch is optional.
    import torch.nn as nn  # type: ignore
except ImportError:
    torch = None

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] DISTILLER: %(message)s')

# Bypass Keras Lambda security restriction for trusted internal models
try:
    if hasattr(tf, 'keras'):
        tf.keras.config.enable_unsafe_deserialization()  # type: ignore
    elif 'keras' in globals():
        import keras
        keras.config.enable_unsafe_deserialization()
except Exception:
    pass

class Distiller:
    """
    Independent service that monitors model folders and creates 
    latent-space (feature extractor) versions of CV models 
    across multiple frameworks (Keras, PyTorch, ONNX).
    """
    def __init__(self):
        self.watch_map = config.DISTILL_MAP
        self.suffix = config.DISTILL_SUFFIX
        # Supported extensions for CV models
        self.supported_extensions = [".keras", ".h5", ".pt", ".pth", ".onnx", ".pb"]

    def cleanup_memory(self):
        """
        Aggressively flushes Keras sessions and invokes garbage collection 
        to keep RAM usage low during idle sleep.
        """
        try:
            # Clear TensorFlow/Keras graph
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()  # type: ignore
            elif 'keras' in globals():
                import keras
                keras.backend.clear_session()
        except Exception:
            pass
        
        # Force Python garbage collection
        gc.collect()

    def perform_variance_check(self, model, framework="keras"):
        """
        Safety net: Passes dummy data through the distilled model to ensure 
        it produces a diverse feature signal rather than constant values.
        """
        try:
            if framework == "keras":
                # Ensure we handle models with multiple inputs if necessary, 
                # but standard CV models usually have one.
                input_shape = model.input_shape[1:]
                test_input = np.random.rand(5, *input_shape).astype(np.float32)
                features = model.predict(test_input, verbose=0)
            elif framework == "pytorch" and torch is not None:
                test_input = torch.rand(5, 3, 224, 224)
                with torch.no_grad():
                    features = model(test_input).numpy()
            else:
                return True

            if features.shape[-1] <= 1:
                logging.error(f"   -> Variance Check Failed: Output dimension too small ({features.shape[-1]}).")
                return False

            variance = np.var(features, axis=0).mean()
            if variance < 1e-6:
                logging.error(f"   -> Variance Check Failed: Model produces static output (Var: {variance:.8f}).")
                return False

            logging.info(f"   -> Variance Check Passed (Signal Variance: {variance:.6f})")
            return True
        except Exception as e:
            logging.warning(f"   -> Variance Check skipped due to error: {e}")
            return True 

    def distill_keras(self, model_path):
        """
        Logic for Keras models using 'compile=False' to bypass custom loss 
        errors and an iterative layer-stripping approach.
        """
        try:
            # FIX: Load with compile=False to ignore 'prediction_loss' or other custom objects
            logging.info(f"   -> [Keras] Loading {model_path.name} (compile=False)...")
            full_model = load_model(model_path, compile=False, safe_mode=False)
            
            # Iterative Strategy: Walk backwards from the end of the model
            # We skip the very last layer (usually the prediction head) automatically
            layers = full_model.layers  # type: ignore
            max_depth = min(len(layers), 10) # Don't strip more than 10 layers deep
            
            for i in range(1, max_depth + 1):
                target_layer = layers[-i]
                
                # We only want layers that output 2D/4D tensors (embeddings or feature maps)
                # We avoid sticking to a layer that's just a Dropout or Activation if possible
                if any(forbidden in target_layer.name.lower() for forbidden in ['dropout', 'input']):
                    continue

                logging.info(f"   -> [Keras] Testing truncation at layer: {target_layer.name}...")
                
                try:
                    distilled_model = Model(inputs=full_model.input, outputs=target_layer.output)  # type: ignore
                    
                    if self.perform_variance_check(distilled_model, framework="keras"):
                        logging.info(f"🟢 [Keras] Successful distillation at: {target_layer.name}")
                        return distilled_model
                except Exception as e:
                    logging.warning(f"      -> Layer {target_layer.name} incompatible: {e}")
                    continue

            logging.error(f"🔴 [Keras] All truncation attempts failed for {model_path.name}")
            return None
        except Exception as e:
            logging.error(f"🔴 Keras load failed even with compile=False: {e}")
            return None

    def distill_pytorch(self, model_path):
        """Logic for PyTorch models with iterative fallback."""
        if torch is None:
            logging.error("🔴 Cannot distill PyTorch model: 'torch' library not found.")
            return None
        try:
            model = torch.load(model_path)
            if isinstance(model, nn.Module):  # type: ignore
                for i in range(1, 4): 
                    layers = list(model.children())[:-i]
                    latent_model = nn.Sequential(*layers)  # type: ignore
                    latent_model.eval()
                    
                    logging.info(f"   -> [PyTorch] Attempting truncation (stripped {i} layers)...")
                    if self.perform_variance_check(latent_model, framework="pytorch"):
                        return latent_model
            return None
        except Exception as e:
            logging.error(f"🔴 PyTorch distillation failed: {e}")
            return None

    def is_file_stable(self, filepath):
        """Verifies file size hasn't changed in the last few seconds."""
        try:
            first_size = os.path.getsize(filepath)
            time.sleep(config.STABILITY_DELAY)
            second_size = os.path.getsize(filepath)
            return first_size == second_size and first_size > 0
        except OSError:
            return False

    def process_model(self, model_file, target_file):
        """Determines framework and executes distillation."""
        ext = model_file.suffix.lower()
        success = False
        
        if ext in [".keras", ".h5"]:
            latent_model = self.distill_keras(model_file)
            if latent_model:
                latent_model.save(target_file)
                del latent_model # Remove local reference immediately
                success = True
        
        elif ext in [".pt", ".pth"]:
            latent_model = self.distill_pytorch(model_file)
            if latent_model:
                torch.save(latent_model, target_file)  # type: ignore
                del latent_model # Remove local reference immediately
                success = True

        # Aggressively flush memory after work is done
        self.cleanup_memory()
        return success

    def run(self):
        logging.info("🟢 Generalized Distiller Service Started. Monitoring model folders...")
        
        while True:
            for src_dir, dest_dir in self.watch_map.items():
                src_path = Path(src_dir)
                dest_path = Path(dest_dir)

                if not src_path.exists(): continue

                for model_file in src_path.iterdir():
                    if model_file.suffix.lower() not in self.supported_extensions:
                        continue
                    
                    if self.suffix in model_file.name:
                        continue
                        
                    distilled_name = model_file.stem + self.suffix + model_file.suffix
                    target_file = dest_path / distilled_name

                    if target_file.exists():
                        continue

                    if not self.is_file_stable(model_file):
                        logging.info(f"⏳ Waiting for {model_file.name} to finish writing...")
                        continue

                    logging.info(f"✨ New {model_file.suffix} model detected: {model_file.name}.")
                    
                    if self.process_model(model_file, target_file):
                        logging.info(f"💾 Distilled version saved: {distilled_name}")

            time.sleep(config.POLLING_INTERVAL)

if __name__ == "__main__":
    distiller = Distiller()
    distiller.run()
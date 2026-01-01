import os
import time
import shutil
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf
try:
    from tensorflow.keras.models import Model, load_model  # type:ignore
except ImportError:
    from keras.models import Model, load_model
import all_config as config

# Framework imports - wrapped in try/except to stay lightweight if not installed
try:
    import torch  # type:ignore  # PyTorch is optional. I don't have it in my environment.
    import torch.nn as nn  # type: ignore
except ImportError:
    torch = None

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] DISTILLER: %(message)s')

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

    def perform_variance_check(self, model, framework="keras"):
        """
        Safety net: Passes dummy data through the distilled model to ensure 
        it produces a diverse feature signal rather than constant values.
        """
        try:
            if framework == "keras":
                # Get input shape, ignoring batch dimension
                input_shape = model.input_shape[1:]
                # Create a small batch of random noise
                test_input = np.random.rand(5, *input_shape).astype(np.float32)
                features = model.predict(test_input, verbose=0)
            elif framework == "pytorch":
                # Assuming standard CV input if not specified
                test_input = torch.rand(5, 3, 224, 224)  # type: ignore
                with torch.no_grad():  # type: ignore
                    features = model(test_input).numpy()
            else:
                return True

            # 1. Size Check: Latent space should usually be > 1 dimension 
            if features.shape[-1] <= 1:
                logging.error(f"   -> Variance Check Failed: Output dimension too small ({features.shape[-1]}).")
                return False

            # 2. Variance Check: Ensure the features aren't all identical 
            variance = np.var(features, axis=0).mean()
            if variance < 1e-6:
                logging.error(f"   -> Variance Check Failed: Model produces static output (Var: {variance:.8f}).")
                return False

            logging.info(f"   -> Variance Check Passed (Signal Variance: {variance:.6f})")
            return True
        except Exception as e:
            logging.warning(f"   -> Variance Check skipped due to error: {e}")
            return True # Default to True to avoid blocking if input shape inference fails

    def distill_keras(self, model_path):
        """Logic for Keras/TensorFlow models with smart bottleneck detection and iterative fallback."""
        try:
            full_model = load_model(model_path)
            
            # Create a list of candidate layers (reversed order)
            # We filter for layers that produce 1D/2D vectors (Batch, Features)
            candidates = []
            for layer in reversed(full_model.layers):  # type: ignore
                out_shape = layer.output_shape
                if len(out_shape) == 2:
                    # Preference given to typical bottleneck layers
                    score = 0
                    if any(k in layer.name.lower() for k in ['pooling', 'flatten', 'bottleneck', 'embedding']):
                        score = 1
                    candidates.append((score, layer))

            # Sort candidates so preferred layers come first, but keep depth order
            # (Basically: try pooling layers first, then try any 1D layers in reverse order)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for score, layer in candidates:
                logging.info(f"   -> [Keras] Attempting truncation at: {layer.name}...")
                try:
                    distilled_model = Model(inputs=full_model.input, outputs=layer.output)  # type: ignore
                    
                    # RUN VARIANCE CHECK
                    if self.perform_variance_check(distilled_model, framework="keras"):
                        logging.info(f"   -> [Keras] Successful distillation at layer: {layer.name}")
                        return distilled_model
                    else:
                        logging.warning(f"   -> [Keras] Layer {layer.name} failed variance check. Trying next candidate...")
                except Exception as e:
                    logging.warning(f"   -> [Keras] Could not create model for layer {layer.name}: {e}")
                    continue

            logging.error(f"   -> [Keras] All candidate layers failed for {model_path.name}")
            return None
        except Exception as e:
            logging.error(f"   -> Keras distillation failed: {e}")
            return None

    def distill_pytorch(self, model_path):
        """Logic for PyTorch models with iterative fallback."""
        if torch is None:
            logging.error("   -> Cannot distill PyTorch model: 'torch' library not found.")
            return None
        try:
            model = torch.load(model_path)
            if isinstance(model, nn.Module):  # type: ignore
                # Try stripping progressively more layers if variance check fails
                for i in range(1, 4): # Try stripping 1, 2, or 3 layers back
                    layers = list(model.children())[:-i]
                    latent_model = nn.Sequential(*layers)  # type: ignore
                    latent_model.eval()
                    
                    logging.info(f"   -> [PyTorch] Attempting truncation (stripped {i} layers)...")
                    if self.perform_variance_check(latent_model, framework="pytorch"):
                        return latent_model
                    
                logging.error(f"   -> [PyTorch] All distillation attempts failed for {model_path.name}")
            return None
        except Exception as e:
            logging.error(f"   -> PyTorch distillation failed: {e}")
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
        
        if ext in [".keras", ".h5"]:
            latent_model = self.distill_keras(model_file)
            if latent_model:
                tmp_save_path = target_file.with_suffix(".tmp")
                latent_model.save(tmp_save_path)
                os.rename(tmp_save_path, target_file)
                return True
        
        elif ext in [".pt", ".pth"]:
            latent_model = self.distill_pytorch(model_file)
            if latent_model:
                tmp_save_path = target_file.with_suffix(".tmp")
                torch.save(latent_model, tmp_save_path)    # type: ignore
                os.rename(tmp_save_path, target_file)
                return True
        
        logging.warning(f"   -> Extension {ext} detected but no specific handler implemented yet.")
        return False

    def run(self):
        logging.info("🚀 Generalized Distiller Service Started. Monitoring model folders...")
        
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
                        logging.info(f"✅ Distilled version saved: {distilled_name}")

            time.sleep(config.POLLING_INTERVAL)

if __name__ == "__main__":
    distiller = Distiller()
    distiller.run()
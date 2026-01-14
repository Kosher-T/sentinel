import os
import sys
import time
import json
import hashlib
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

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:
    torch = None

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] DISTILLER: %(message)s')

# Bypass Security
try:
    if hasattr(tf, 'keras'):
        tf.keras.config.enable_unsafe_deserialization()  # type: ignore
    elif 'keras' in globals():
        import keras
        keras.config.enable_unsafe_deserialization()
except Exception:
    pass

class Distiller:
    def __init__(self):
        self.watch_map = config.DISTILL_MAP
        self.suffix = config.DISTILL_SUFFIX
        self.supported_extensions = [".keras", ".h5", ".pt", ".pth", ".onnx", ".pb"]
        self.memory_file = file_path.parent / "distiller_memory.json"
        self.memory = self.load_memory()

    def load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    mem = json.load(f)
                logging.info(f"🧠 Loaded knowledge for {len(mem)} architectures.")
                return mem
            except Exception:
                return {}
        return {}

    def get_fingerprint(self, model):
        """Generates architectural hash."""
        try:
            sig = []
            for l in model.layers:
                # Use class name and output shape to identify architecture
                sig.append(f"{l.__class__.__name__}:{str(l.output_shape)}")
            full_sig = "|".join(sig)
            return hashlib.md5(full_sig.encode()).hexdigest()
        except Exception:
            return None

    def cleanup_memory(self):
        try:
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()  # type: ignore
            elif 'keras' in globals():
                import keras
                keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    def perform_variance_check(self, model, framework="keras"):
        try:
            if framework == "keras":
                input_shape = model.input_shape[1:]
                # Handle potential multi-channel inputs (like 18 channels)
                if input_shape[-1] is None: c = 3 # fallback
                else: c = input_shape[-1]
                
                # Construct safe shape
                safe_shape = [5]
                for dim in input_shape:
                    safe_shape.append(dim if dim is not None else 224)
                
                test_input = np.random.rand(*safe_shape).astype(np.float32)
                features = model.predict(test_input, verbose=0)
            elif framework == "pytorch" and torch is not None:
                test_input = torch.rand(5, 3, 224, 224)
                with torch.no_grad():
                    features = model(test_input).numpy()
            else:
                return True

            if len(features.shape) > 1 and features.shape[-1] <= 1:
                return False

            # Flatten for variance check if 4D
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)

            variance = np.var(features, axis=0).mean()
            if variance < 1e-6:
                return False

            return True
        except Exception:
            return True 

    def distill_keras(self, model_path):
        try:
            logging.info(f"   -> [Keras] Loading {model_path.name} (compile=False)...")
            full_model = load_model(model_path, compile=False, safe_mode=False)
            
            # 1. SMART CHECK: Do we know this model?
            fp = self.get_fingerprint(full_model)
            if fp and fp in self.memory:
                target_layer_name = self.memory[fp]
                logging.info(f"🧠 Recognized architecture. Cutting at learned layer: '{target_layer_name}'")
                try:
                    return Model(inputs=full_model.input, outputs=full_model.get_layer(target_layer_name).output)  # type: ignore
                except Exception as e:
                    logging.warning(f"   -> Learned layer not found (model changed?): {e}. Fallback to auto.")
            
            # 2. AUTO CHECK: Iterative Fallback
            layers = full_model.layers  # type: ignore
            max_depth = min(len(layers), 15)
            
            for i in range(1, max_depth + 1):
                target_layer = layers[-i]
                if any(forbidden in target_layer.name.lower() for forbidden in ['dropout', 'input']):
                    continue

                logging.info(f"   -> [Auto] Testing cut at: {target_layer.name}...")
                try:
                    distilled_model = Model(inputs=full_model.input, outputs=target_layer.output)  # type: ignore
                    if self.perform_variance_check(distilled_model, framework="keras"):
                        logging.info(f"🟢 [Auto] Successful distillation at: {target_layer.name}")
                        return distilled_model
                except Exception:
                    continue

            logging.error(f"🔴 [Keras] All truncation attempts failed for {model_path.name}")
            return None
        except Exception as e:
            logging.error(f"🔴 Keras load failed: {e}")
            return None

    def distill_pytorch(self, model_path):
        # PyTorch logic remains the same (Manual stripping)
        if torch is None: return None
        try:
            model = torch.load(model_path)
            if isinstance(model, nn.Module):  # type: ignore
                for i in range(1, 4): 
                    layers = list(model.children())[:-i]
                    latent_model = nn.Sequential(*layers)  # type: ignore
                    latent_model.eval()
                    if self.perform_variance_check(latent_model, framework="pytorch"):
                        return latent_model
            return None
        except Exception:
            return None

    def is_file_stable(self, filepath):
        try:
            first_size = os.path.getsize(filepath)
            time.sleep(config.STABILITY_DELAY)
            second_size = os.path.getsize(filepath)
            return first_size == second_size and first_size > 0
        except OSError:
            return False

    def process_model(self, model_file, target_file):
        # Refresh memory occasionally in case CLI updated it
        self.memory = self.load_memory()
        
        ext = model_file.suffix.lower()
        success = False
        
        if ext in [".keras", ".h5"]:
            latent_model = self.distill_keras(model_file)
            if latent_model:
                latent_model.save(target_file)
                del latent_model 
                success = True
        elif ext in [".pt", ".pth"]:
            latent_model = self.distill_pytorch(model_file)
            if latent_model:
                torch.save(latent_model, target_file)  # type: ignore
                del latent_model
                success = True

        self.cleanup_memory()
        return success

    def run(self):
        logging.info("🟢 Smart Distiller Service Started. Monitoring model folders...")
        while True:
            for src_dir, dest_dir in self.watch_map.items():
                src_path = Path(src_dir)
                dest_path = Path(dest_dir)
                if not src_path.exists(): continue
                for model_file in src_path.iterdir():
                    if model_file.suffix.lower() not in self.supported_extensions: continue
                    if self.suffix in model_file.name: continue
                    
                    distilled_name = model_file.stem + self.suffix + model_file.suffix
                    target_file = dest_path / distilled_name

                    if target_file.exists(): continue
                    if not self.is_file_stable(model_file): continue

                    logging.info(f"✨ New {model_file.suffix} model detected: {model_file.name}.")
                    if self.process_model(model_file, target_file):
                        logging.info(f"💾 Distilled version saved: {distilled_name}")
            time.sleep(config.POLLING_INTERVAL)

if __name__ == "__main__":
    distiller = Distiller()
    distiller.run()
# Logs environment details (OS, CPU, Backend) for hardware parity debugging.
# Loads VFI models (old and fresh) to generate interpolated frames on the golden set.
# Extracts and saves feature embeddings for both models to be used for decay analysis.
# Performs Wasserstein Distance analysis to quantify model decay.

import os
import sys
import numpy as np
import cv2
try:
    import keras
except ImportError:
    keras = None
import platform
import psutil
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

# 1. Setup paths and handle project root for imports IMMEDIATELY
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 2. Internal Module Imports
try:
    import detector_model_decay.extractor as extractor
    import detector_model_decay.analyzer as analyzer
except ImportError:
    from . import extractor as extractor
    from . import analyzer as analyzer

import all_config as config

# --- OPTIMIZED CPU CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
if keras:
    try:
        keras.mixed_precision.set_global_policy('float32')
    except Exception:
        pass

# --- PERSISTENT MODEL TRACKING ---
HISTORY_FILE = config.MODEL_DECAY_ROOT / "model_run_history.json"

class ModelMetadataManager:
    def __init__(self, history_path):
        self.history_path = history_path
        self.history = self._load_history()

    def _load_history(self):
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_history(self):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def get_file_metadata(self, file_path):
        if not file_path or not file_path.exists():
            return None
        stats = file_path.stat()
        return {
            "filename": file_path.name,
            "size_bytes": stats.st_size,
            "modified_time": stats.st_mtime
        }

    def has_model_changed(self, model_role, model_path):
        current_meta = self.get_file_metadata(model_path)
        if not current_meta:
            return True
            
        last_meta = self.history.get(model_role)
        
        if last_meta != current_meta:
            print(f"🔄 Model Change Detected for {model_role}!")
            return True
        return False

    def update_entry(self, model_role, model_path):
        self.history[model_role] = self.get_file_metadata(model_path)
        self._save_history()

# --- UTILITIES ---

def log_environment():
    print("\n" + "=" * 45)
    print("🖥️  ENVIRONMENT DIAGNOSTICS")
    print("-" * 45)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    try:
        if keras:
            print(f"Keras Backend: {keras.backend.backend()}")
        else:
             print("Keras Backend: Not Available")
    except AttributeError:
        pass # Keras 3+ might not have backend() immediately available or different API
    print(f"OpenCV Version: {cv2.__version__}")
    print("=" * 45 + "\n")

def load_vfi_model(model_path):
    print(f"🧠 Loading Challenger Model: {model_path.name}...")
    if keras is None:
        print("❌ Keras is not installed. Cannot load model.")
        return None
        
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def scan_golden_set():
    """
    Scans the Golden Set directory for 'sample_XXXX' folders.
    Returns a list of valid sample directories.
    """
    if not config.GOLDEN_SET_DIR.exists():
        print(f"❌ Golden Set directory not found at: {config.GOLDEN_SET_DIR}")
        return []

    samples = sorted([
        d for d in config.GOLDEN_SET_DIR.iterdir() 
        if d.is_dir() and d.name.startswith("sample_")
    ])
    
    if not samples:
        print("⚠️ No samples found in Golden Set directory.")
    else:
        print(f"✅ Found {len(samples)} Golden Set samples.")
        
    return samples

def load_sample_data(sample_dir):
    """
    Loads input data and baseline prediction from a sample directory.
    
    Returns:
        input_data (numpy array), baseline_output (numpy array), input_path (Path)
    """
    input_dir = sample_dir / "input"
    output_dir = sample_dir / "output"
    
    if not input_dir.exists() or not output_dir.exists():
        return None, None, None

    # Load baseline prediction
    baseline_path = output_dir / "prediction.npy"
    if not baseline_path.exists():
        return None, None, None
    
    try:
        baseline_output = np.load(baseline_path)
    except Exception:
        return None, None, None

    # Load input using logic similar to Curator
    # We need to reconstruct the batch from the input directory
    # For now, we reuse the Extractor or implement simple loading logic here
    # Since Extractor is focused on Embeddings, let's implement a simple loader here that mirrors the Curator's
    
    try:
        input_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        ])
        
        if not input_files:
            # Maybe it's a single file?
            if input_dir.is_file(): # Unlikely specific structure but good safety
                pass
            return None, None, None

        # Load images
        images = []
        for img_file in input_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        if not images:
            return None, None, None
            
        batch = np.stack(images, axis=0)
        
        # Normalize to [0, 1]
        if batch.max() > 1.0:
            batch = batch.astype(np.float32) / 255.0
            
        return batch, baseline_output, input_dir

    except Exception as e:
        print(f"Error loading sample {sample_dir.name}: {e}")
        return None, None, None


def run_decay_analysis(challenger_model_path):
    """
    Main Logic:
    1. Iterates over Golden Set samples.
    2. Runs Challenger Model on Inputs.
    3. Compares Challenger Output vs Baseline Output (stored in Golden Set).
    4. Computes Drift Score.
    """
    log_environment()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Load Challenger Model
    challenger_model = load_vfi_model(challenger_model_path)
    if challenger_model is None:
        print("❌ Could not load Challenger Model. Aborting.")
        return False

    # 2. Get Golden Set Samples
    samples = scan_golden_set()
    if not samples:
        return False # No data to test against

    # 3. Initialize Feature Extractor (Backbone)
    print("🚀 Initializing Feature Extractor (VGG16)...")
    feature_model = extractor.create_embedding_model() # Assumes this exists in extractor.py via VGG16 default code or we need to check

    # Storage
    baseline_embeddings = []
    challenger_embeddings = []
    
    processed_count = 0
    
    print("\n🔍 Starting Inference & Comparison...")
    
    for sample in samples:
        input_batch, baseline_output, _ = load_sample_data(sample)
        
        if input_batch is None or baseline_output is None:
            continue
            
        # A. Run Challenger Inference
        try:
            # Predict
            challenger_output = challenger_model.predict(input_batch, verbose=0)
        except Exception as e:
            print(f"⚠️ Inference failed on {sample.name}: {e}")
            continue
            
        # B. Check Shapes
        # Standardize shapes if needed (e.g. if single output vs list)
        if isinstance(challenger_output, list):
            challenger_output = challenger_output[0]
        if isinstance(baseline_output, list): # Should be numpy array if loaded from npy, but maybe saved as list
            pass
            
        # Ensure compatible shapes for feature extraction
        # Current extractor expects batch of images or generic array
        # We need to extract features from the OUTPUTS (predicted frames)
        
        # C. Extract Features from Outputs
        # We use the NEW extract_features_from_array function
        try:
            b_emb = extractor.extract_features_from_array(feature_model, baseline_output)
            c_emb = extractor.extract_features_from_array(feature_model, challenger_output)
            
            if len(b_emb) > 0 and len(c_emb) > 0:
                baseline_embeddings.append(b_emb)
                challenger_embeddings.append(c_emb)
                processed_count += 1
                
        except Exception as e:
            print(f"Extract failed on {sample.name}: {e}")

        if processed_count % 10 == 0:
            print(f"   > Processed {processed_count} samples...")

    if processed_count == 0:
        print("❌ No valid samples processed.")
        return False

    # 4. Aggregate & Analyze
    print(f"\n📊 Aggregating Results ({processed_count} samples)...")
    
    # Stack all embeddings
    baseline_full = np.vstack(baseline_embeddings)
    challenger_full = np.vstack(challenger_embeddings)
    
    # Use the analyzer module
    # Need to match input format of analyze_drift/calculate_decay_score
    # Previously analyzer.calculate_decay_score(f_emb, o_emb...)
    # We'll re-calculate metrics manually here to be safe or use analyzer primitives
    
    # PCA
    n_comp = min(16, baseline_full.shape[0], baseline_full.shape[1])
    pca = analyzer.PCA(n_components=n_comp, svd_solver='full', random_state=42)
    b_pca = pca.fit_transform(baseline_full)
    c_pca = pca.transform(challenger_full)
    
    # Distance Metrics
    avg_wd = np.mean([analyzer.wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])])
    
    # Final Score Calculation
    # Simple Decay Score based on WD for now, reusing config threshold
    decay_score = avg_wd * 100 # Arbitrary scaling to make it %-like if needed, or just use raw WD
    
    # Normalized Probability (0-1)
    drift_prob = 1 - np.exp(-0.1 * avg_wd)
    
    print("\n==================================================")
    print("           DECAY CHECK REPORT")
    print("==================================================")
    print(f"Timestamp:      {timestamp}")
    print(f"Samples Tested: {processed_count}")
    print(f"Avg Wasserstein:{avg_wd:.4f}")
    print(f"Drift Prob:     {drift_prob:.4f}")
    print("--------------------------------------------------")
    
    # Threshold Check
    # We use drift_prob as the main metric now generally, or the raw score?
    # Let's align with Config DECAY_THRESHOLD which is 5.0 (percent drop?)
    # If using WD, we need to calibrate.
    # For this implementation, let's Fail if Drift Prob > 0.3 (significant deviation)
    
    PASSED = drift_prob < 0.3
    STATUS = "PASS" if PASSED else "FAIL"
    
    print(f"Status:         {STATUS}")
    print("==================================================")
    
    return PASSED


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(config.FRESH_MODEL_PATH), help="Path to Challenger Model")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if model_path.is_dir():
         # Logic to pick file inside
         # Reuse utilities logic if needed, or expect direct file
         # For now, simplistic check
         potential = list(model_path.glob("*.keras")) + list(model_path.glob("*.h5"))
         if potential:
             model_path = potential[0]
             
    if not model_path.exists() or model_path.is_dir():
        # Fallback if specific file not found
        print(f"❌ Invalid model path: {model_path}")
        sys.exit(1)
        
    success = run_decay_analysis(model_path)
    sys.exit(0 if success else 1)
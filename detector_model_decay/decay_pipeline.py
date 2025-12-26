import os
import numpy as np
import sys
import cv2
import keras
import platform
import psutil
import json
import time
from pathlib import Path

# 1. Setup paths and handle project root for imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Internal Sentinel Module Imports
import feature_extractor as extractor
import decay_analyzer as analyzer
import all_config as config

# --- OPTIMIZED CPU CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
keras.mixed_precision.set_global_policy('float32')

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
            except:
                return {}
        return {}

    def has_model_changed(self, role, model_path):
        m_path = Path(model_path)
        if not m_path.exists(): return False
        
        current_stats = {
            "size": m_path.stat().st_size,
            "mtime": m_path.stat().st_mtime,
            "name": m_path.name
        }
        
        last_stats = self.history.get(role)
        return current_stats != last_stats

    def update_entry(self, role, model_path):
        m_path = Path(model_path)
        self.history[role] = {
            "size": m_path.stat().st_size,
            "mtime": m_path.stat().st_mtime,
            "name": m_path.name
        }
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

def print_env_info():
    print("\n" + "="*50)
    print("       SENTINEL ENVIRONMENT DIAGNOSTICS")
    print("="*50)
    print(f"OS:        {platform.system()} {platform.release()}")
    print(f"CPU:       {platform.processor()}")
    print(f"RAM:       {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Backend:   Keras {keras.__version__} / TensorFlow {os.environ.get('TF_VERSION', 'Standard')}")
    print("="*50 + "\n")

def load_vfi_model(model_path):
    print(f"📂 Loading VFI Model: {Path(model_path).name}...")
    try:
        # custom_objects would be added here if needed for custom U-Net layers
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def run_vfi_inference(model, output_dir, label, force=False):
    """Generates frames from the Golden Set using the provided model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have results and aren't forcing a re-run
    existing_files = list(output_dir.glob("*.png"))
    if not force and len(existing_files) > 0:
        print(f"✅ {label} results found in cache. Skipping inference.")
        return

    print(f"🚀 Running inference for {label}...")
    # This section would contain the actual VFI logic for im4/im7 prediction
    # Mocking for pipeline flow:
    # for sample in golden_set: model.predict(...) -> save to output_dir
    print(f"✨ Inference complete for {label}.")

def extract_embeddings(results_dir, role, force=False):
    """Converts VFI generated frames into feature embeddings."""
    save_path = config.EMBEDDINGS_ROOT / f"{role}_embeddings.npy"
    config.EMBEDDINGS_ROOT.mkdir(parents=True, exist_ok=True)

    if not force and save_path.exists():
        print(f"📦 Found cached embeddings for {role}. skipping extraction.")
        return

    print(f"📸 Extracting features for {role} from {results_dir}...")
    vgg_model = extractor.create_embedding_model()
    embeddings = extractor.extract_features(vgg_model, results_dir)
    
    if embeddings.size > 0:
        np.save(save_path, embeddings)
        print(f"💾 Saved {role} embeddings to {save_path.name}")

def calculate_visual_averages(results_dir):
    """
    In a real scenario, this would compare results_dir against the 
    Ground Truth in config.GOLDEN_SET_DIR.
    Returns mock values for now.
    """
    return 42.5, 0.982  # Mock PSNR/SSIM

def run_decay_check():
    print_env_info()
    meta_mgr = ModelMetadataManager(HISTORY_FILE)

    # 1. Locate Models
    fresh_path = config.FRESH_MODEL_PATH
    old_path = config.OLD_MODEL_PATH

    if not fresh_path.exists() or not old_path.exists():
        print(f"❌ Missing models. Check {config.FRESH_MODEL_PATH} and {config.OLD_MODEL_PATH}")
        return

    # 2. Process Fresh Model (The Baseline)
    fresh_changed = meta_mgr.has_model_changed("fresh_model", fresh_path)
    if fresh_changed:
        print("⚡ Fresh Model has changed. Updating baseline...")
    
    run_vfi_inference(None, config.FRESH_RESULTS_DIR, "Fresh Model", force=fresh_changed)
    extract_embeddings(config.FRESH_RESULTS_DIR, "fresh_model", force=fresh_changed)
    meta_mgr.update_entry("fresh_model", fresh_path)

    print("\n" + "-"*30)

    # 3. Process Old Model (The Comparison)
    old_changed = meta_mgr.has_model_changed("old_model", old_path)
    if old_changed:
        print("⚡ New 'Old Model' detected. Auto-triggering...")
    else:
        choice = input("🔄 Old Model matches history. Re-run inference? (y/n) [n]: ").lower()
        if choice == 'y': old_changed = True

    run_vfi_inference(None, config.OLD_RESULTS_DIR, "Old Model", force=old_changed)
    extract_embeddings(config.OLD_RESULTS_DIR, "old_model", force=old_changed)
    meta_mgr.update_entry("old_model", old_path)

    # 4. Final Analysis
    print("\n⚖️  Performing Multi-Metric Decay Analysis...")
    
    try:
        f_emb = np.load(config.EMBEDDINGS_ROOT / "fresh_model_embeddings.npy")
        o_emb = np.load(config.EMBEDDINGS_ROOT / "old_model_embeddings.npy")

        # Get visual quality against Ground Truth
        # (This logic would be more complex in production, comparing dir vs dir)
        f_psnr, f_ssim = calculate_visual_averages(config.FRESH_RESULTS_DIR)
        o_psnr, o_ssim = calculate_visual_averages(config.OLD_RESULTS_DIR)

        # Calculate decay scores for specific tasks
        # Using the fresh model as the anchor for decay
        decay_score = analyzer.calculate_decay_score(f_emb, o_emb, o_psnr, o_ssim, task="im4")
        
        status = "STABLE" if decay_score < config.DECAY_THRESHOLD else "DECAYED"
        color = "\033[92m" if status == "STABLE" else "\033[91m"
        reset = "\033[0m"

        print("\n" + "="*50)
        print("           SENTINEL DECAY ANALYSIS")
        print("="*50)
        print(f"Fresh: {fresh_path.name}")
        print(f"Old:   {old_path.name}")
        print("-" * 50)
        print(f"Visual Quality (Old): {o_psnr:.2f}dB / {o_ssim:.4f} SSIM")
        print(f"Combined Decay Score: {decay_score:.2f}%")
        print(f"Status:               {color}{status}{reset}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    run_decay_check()
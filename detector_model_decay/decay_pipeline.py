# Logs environment details (OS, CPU, Backend) for hardware parity debugging.
# Loads VFI models (old and fresh) to generate interpolated frames on the golden set.
# Extracts and saves feature embeddings for both models to be used for decay analysis.
# Performs Wasserstein Distance analysis to quantify model decay.


import os
import sys
import numpy as np
import cv2
import keras
import platform
import psutil
import json
import time
from datetime import datetime
from pathlib import Path

# 1. Setup paths and handle project root for imports IMMEDIATELY
# This must happen before any internal module or config imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 2. Internal Module Imports
# Import feature_extractor from the same directory as this script
try:
    import feature_extractor as extractor
except ImportError:
    from . import feature_extractor as extractor

# Updated import to include statistical analysis for the detailed report
# Using the absolute path logic to ensure accuracy inside Docker
from detector_model_decay.decay_analyzer import (
    calculate_decay_score, 
    calculate_visual_metrics, 
    analyze_statistical_drift,
    kl_divergence,
    mmd_linear,
    wasserstein_distance,
    cosine_similarity,
    PCA
)

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
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_history(self):
        # Ensure directory exists before saving
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def get_file_metadata(self, file_path):
        """Returns a unique signature for the file: name + size + mtime."""
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
            print(f"   Old: {last_meta.get('filename') if last_meta else 'None'}")
            print(f"   New: {current_meta['filename'] if current_meta else 'None'}")
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
    print(f"Keras Backend: {keras.backend.backend()}")
    print(f"OpenCV Version: {cv2.__version__}")
    print("=" * 45 + "\n")

def select_model_file(directory, label="Model"):
    """
    Scans a directory for model files and prompts user selection if multiple exist.
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return None

    valid_exts = {'.keras', '.h5', '.hdf5'}
    # Find files with valid extensions or directories that look like SavedModels
    model_files = sorted([f for f in directory.iterdir() if f.suffix in valid_exts or (f.is_dir() and (f / "saved_model.pb").exists())])

    if not model_files:
        print(f"⚠️ No model files found in {directory}")
        return None

    if len(model_files) == 1:
        return model_files[0]

    print(f"\n📂 Multiple files found for {label}:")
    for idx, f in enumerate(model_files):
        print(f"  [{idx}] {f.name}")
    
    while True:
        try:
            choice = int(input(f"👉 Select {label} index: "))
            if 0 <= choice < len(model_files):
                return model_files[choice]
        except ValueError:
            pass
        print("❌ Invalid selection. Try again.")

def load_vfi_model(model_path):
    print(f"🧠 Loading: {model_path.name}...")
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def prepare_input_sequence(seq_path):
    frames = []
    original_dims = None
    for i in range(1, 7):
        img_path = seq_path / f"im{i}.webp"
        if not img_path.exists():
            img_path = seq_path / f"im{i}.png"
        img = cv2.imread(str(img_path))
        if img is None: return None, None
        if original_dims is None:
            h, w = img.shape[:2]
            original_dims = (w, h)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        frames.append(img_resized)
    input_data = np.concatenate(frames, axis=-1).astype('float32') / 255.0
    return np.expand_dims(input_data, axis=0), original_dims

def run_vfi_inference(vfi_model, output_base_dir, model_name="model", force=False):
    print(f"\n🎬 Inference: {model_name}")
    print("-" * 30)
    raw_data_path = config.GOLDEN_SET_DIR / "sequences" 
    if not raw_data_path.exists():
        print(f"❌ Missing sequences at {raw_data_path}")
        return False

    sequence_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir()])
    processed = 0
    
    for seq_dir in sequence_dirs:
        out_path = output_base_dir / seq_dir.name
        if not force and (out_path / "im7_pred.webp").exists():
            continue

        out_path.mkdir(parents=True, exist_ok=True)
        input_tensor, original_dims = prepare_input_sequence(seq_dir)
        if input_tensor is None: continue

        preds = vfi_model.predict(input_tensor, verbose=0)
        im7_f = cv2.resize((preds[0][0] * 255).astype(np.uint8), original_dims, interpolation=cv2.INTER_CUBIC)
        im4_f = cv2.resize((preds[1][0] * 255).astype(np.uint8), original_dims, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(str(out_path / "im4_pred.webp"), cv2.cvtColor(im4_f, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_path / "im7_pred.webp"), cv2.cvtColor(im7_f, cv2.COLOR_RGB2BGR))
        
        processed += 1
        if processed % 10 == 0:
            print(f"   > Processed {processed} sequences...")

    print(f"✅ {model_name} processing complete.")
    return True

def extract_embeddings(results_dir, model_id, force=False):
    print(f"\n🚀 Feature Extraction: {model_id}")
    target_dir = config.EMBEDDINGS_ROOT / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    feature_model = extractor.create_embedding_model()
    
    for tag in ["im4", "im7"]:
        save_file = target_dir / f"{tag}_embeddings.npy"
        if not force and save_file.exists():
            continue
            
        paths = [str(p / f"{tag}_pred.webp") for p in sorted(results_dir.iterdir()) if p.is_dir()]
        paths = [p for p in paths if os.path.exists(p)]
        
        if paths:
            print(f"   > Extracting {tag}...")
            emb = extractor.extract_features(feature_model, paths)
            np.save(save_file, emb)

def run_analysis():
    threshold = getattr(config, 'DECAY_THRESHOLD', 7.0)
    backbone = getattr(config, 'EMBEDDING_MODEL_TYPE', "VGG16")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Storage for combined results
    results = {}

    for task in ["im4", "im7"]:
        f_path = config.EMBEDDINGS_ROOT / "fresh_model" / f"{task}_embeddings.npy"
        o_path = config.EMBEDDINGS_ROOT / "old_model" / f"{task}_embeddings.npy"
        
        if f_path.exists() and o_path.exists():
            # 1. Load Data
            f_emb = np.load(f_path)
            o_emb = np.load(o_path)
            
            # 2. Visual metrics
            psnrs, ssims = [], []
            seqs = sorted([d for d in config.FRESH_RESULTS_DIR.iterdir() if d.is_dir()])
            for s in seqs:
                img_f_path = s / f"{task}_pred.webp"
                img_o_path = config.OLD_RESULTS_DIR / s.name / f"{task}_pred.webp"
                if img_f_path.exists() and img_o_path.exists():
                    im1 = cv2.imread(str(img_f_path))
                    im2 = cv2.imread(str(img_o_path))
                    if im1 is not None and im2 is not None:
                        p, sm = calculate_visual_metrics(im1, im2)
                        psnrs.append(p); ssims.append(sm)
            
            avg_p = np.mean(psnrs) if psnrs else 0
            avg_s = np.mean(ssims) if ssims else 0

            # 3. Calculate detailed stats for breakdown
            n_comp = min(16, f_emb.shape[0], f_emb.shape[1])
            pca = PCA(n_components=n_comp, svd_solver='full', random_state=42)
            b_pca = pca.fit_transform(f_emb)
            c_pca = pca.transform(o_emb)

            avg_wd = np.mean([wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])])
            avg_kl = np.mean([kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])])  # type:ignore
            
            b_centroid = np.mean(f_emb, axis=0).reshape(1, -1)
            c_centroid = np.mean(o_emb, axis=0).reshape(1, -1)
            cos_dist = 1 - cosine_similarity(b_centroid, c_centroid)[0][0]
            
            mmd_val = mmd_linear(f_emb, o_emb)

            # 4. Aggregated Score
            score = calculate_decay_score(f_emb, o_emb, avg_p, avg_s, task=task)
            
            # Store in results dict
            task_key = "Interpolation" if task == "im4" else "Prediction"
            results[task_key] = {
                "score": score,
                "samples": len(f_emb),
                "wasserstein": 1 - np.exp(-0.1 * avg_wd),
                "mmd": 1 - np.exp(-0.05 * mmd_val),
                "cosine": 1 - np.exp(-5.0 * cos_dist),
                "kl": 1 - np.exp(-0.5 * avg_kl)
            }

    if not results:
        print("❌ No data available for report generation.")
        return

    # Determine overall status (FAIL if either fails)
    overall_status = "PASS"
    for r in results.values():
        if r["score"] > threshold:
            overall_status = "FAIL"
            break

    # 5. Combined Report Formatting
    print("\n==================================================")
    print("           SENTINEL COMBINED DRIFT REPORT")
    print("==================================================")
    print(f"Timestamp:         {timestamp}")
    print(f"Backbone:          {backbone}")
    print(f"Overall Status:    {overall_status}")
    print("--------------------------------------------------")
    print(f"Baseline Samples:  {results.get('Interpolation', results.get('Prediction'))['samples']}")  # type: ignore
    print(f"Incoming Samples:  {results.get('Interpolation', results.get('Prediction'))['samples']}")  # type: ignore
    print("--------------------------------------------------")
    print("FINAL DRIFT SCORE")
    for task_name, r in results.items():
        print(f"{task_name}: {r['score']:.2f}%")
    print("--------------------------------------------------")
    print("THRESHOLD")
    for task_name in results.keys():
        print(f"{task_name}: {threshold:.2f}%")
    print("--------------------------------------------------")
    print("METRIC BREAKDOWN (Normalized 0-1 probability):")
    
    for metric_label, key in [("Wasserstein", "wasserstein"), ("Mmd", "mmd"), ("Cosine", "cosine"), ("Kl", "kl")]:
        print(f" -> {metric_label}")
        for task_name, r in results.items():
            print(f"{task_name}: {r[key]:.4f}")
        print() # Add small gap between metrics

    print("==================================================")

if __name__ == "__main__":
    log_environment()
    
    meta_mgr = ModelMetadataManager(HISTORY_FILE)
    
    # Use select_model_file on the directory paths defined in config
    fresh_path = select_model_file(config.FRESH_MODEL_PATH, "Fresh Model")
    old_path = select_model_file(config.OLD_MODEL_PATH, "Old Model")

    if not fresh_path or not old_path:
        print("❌ Model selection failed. Exiting.")
        sys.exit(1)

    # 1. Fresh Model Logic
    fresh_changed = meta_mgr.has_model_changed("fresh_model", fresh_path)
    # Check if embeddings already exist for fresh model
    fresh_emb_im4 = config.EMBEDDINGS_ROOT / "fresh_model" / "im4_embeddings.npy"
    fresh_emb_im7 = config.EMBEDDINGS_ROOT / "fresh_model" / "im7_embeddings.npy"
    fresh_missing = not (fresh_emb_im4.exists() and fresh_emb_im7.exists())

    if fresh_changed or fresh_missing:
        fresh_m = load_vfi_model(fresh_path)
        if fresh_m:
            run_vfi_inference(fresh_m, config.FRESH_RESULTS_DIR, "Fresh Model", force=fresh_changed)
            meta_mgr.update_entry("fresh_model", fresh_path)
            extract_embeddings(config.FRESH_RESULTS_DIR, "fresh_model", force=fresh_changed)
            del fresh_m
            keras.backend.clear_session()
    else:
        print("✅ Fresh Model embeddings found. Skipping inference and extraction.")

    # 2. Old Model Logic
    print("\n" + "-"*30)
    old_changed = meta_mgr.has_model_changed("old_model", old_path)
    
    # Check if embeddings already exist for old model
    old_emb_im4 = config.EMBEDDINGS_ROOT / "old_model" / "im4_embeddings.npy"
    old_emb_im7 = config.EMBEDDINGS_ROOT / "old_model" / "im7_embeddings.npy"
    old_missing = not (old_emb_im4.exists() and old_emb_im7.exists())
    
    should_run_old = False
    if old_changed or old_missing:
        if old_changed:
            print("⚡ New model detected for 'Old Model' role. Auto-triggering inference.")
        else:
            print("⚠️ Old Model embeddings missing. Triggering inference.")
        should_run_old = True
    else:
        user_choice = input("🔄 Old Model matches history. Re-run anyway? (y/n) [n]: ").lower()
        if user_choice == 'y':
            should_run_old = True

    if should_run_old:
        old_m = load_vfi_model(old_path)
        if old_m:
            run_vfi_inference(old_m, config.OLD_RESULTS_DIR, "Old Model", force=True)
            meta_mgr.update_entry("old_model", old_path)
            extract_embeddings(config.OLD_RESULTS_DIR, "old_model", force=True)
            del old_m
            keras.backend.clear_session()
    else:
        print("✅ Old Model embeddings found. Skipping inference and extraction.")

    run_analysis()
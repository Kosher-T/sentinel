import os
import sys
import numpy as np
import keras
from datetime import datetime
from pathlib import Path

# 1. Setup paths and handle project root for imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Internal Sentinel Module Imports
import feature_extractor as detector
import data_analyzer as analyzer
import all_config as config

# Define persistence path
EMBEDDINGS_DIR = config.BASE_DATA_DIR / "data_drift" / "embeddings"
BASELINE_CACHE = EMBEDDINGS_DIR / "baseline_embeddings.npy"
INCOMING_CACHE = EMBEDDINGS_DIR / "incoming_embeddings.npy"

def ensure_dirs():
    """Ensures the embedding directory exists."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def print_drift_report(score, metrics_breakdown, num_baseline, num_incoming):
    """
    Prints a clean, formatted report to the console.
    """
    status = "FAIL" if score > config.DRIFT_THRESHOLD else "PASS"
    color_code = "\033[91m" if status == "FAIL" else "\033[92m"
    reset = "\033[0m"

    print("\n" + "="*50)
    print("         SENTINEL DATA DRIFT REPORT")
    print("="*50)
    print(f"Timestamp:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backbone:          {config.EMBEDDING_MODEL_TYPE}")
    print(f"Overall Status:    {color_code}{status}{reset}")
    print("-" * 50)
    print(f"Baseline Samples:  {num_baseline}")
    print(f"Incoming Samples:  {num_incoming}")
    print("-" * 50)
    print(f"FINAL DRIFT SCORE: {score:.2f}%")
    print(f"THRESHOLD:         {config.DRIFT_THRESHOLD:.2f}%")
    print("-" * 50)
    print("METRIC BREAKDOWN (Normalized 0-1 probability):")
    for metric, val in metrics_breakdown.items():
        print(f" -> {metric.capitalize():<12}: {val:.4f}")
    
    if status == "FAIL":
        print("\n[!] WARNING: Significant distribution shift detected.")
        print("    Consider retraining or investigating data source integrity.")
    print("="*50 + "\n")

def run_drift_check(incoming_path=None):
    """
    Orchestrates discovery, caching, extraction, and analysis.
    """
    ensure_dirs()
    incoming_path = incoming_path or config.INCOMING_DATA_PATH
    
    print(f"\n🔍 Sentinel starting drift check...")
    print(f"📁 Baseline: {config.ORIGINAL_DATA_PATH}")
    print(f"📁 Incoming: {incoming_path}\n")

    # 1. Discovery
    baseline_files = detector.get_recursive_image_paths(config.ORIGINAL_DATA_PATH)
    incoming_files = detector.get_recursive_image_paths(incoming_path)

    if not baseline_files or not incoming_files:
        print(f"\n❌ Error: Missing images. Baseline: {len(baseline_files)}, Incoming: {len(incoming_files)}")
        return

    baseline_emb = None
    incoming_emb = None
    model = None

    try:
        # --- BASELINE HANDLING ---
        if BASELINE_CACHE.exists():
            print(f"\n📦 Found cached baseline embeddings at {BASELINE_CACHE.name}.")
            baseline_emb = np.load(BASELINE_CACHE)
            # Basic sanity check: does the embedding count match the image count?
            if len(baseline_emb) != len(baseline_files):
                print("⚠️  Cache mismatch (count). Re-extracting baseline...")
                baseline_emb = None
        
        if baseline_emb is None:
            if model is None: model = detector.create_embedding_model()
            print(f"\n📸 Extracting {len(baseline_files)} baseline images...")
            baseline_emb = detector.extract_features_from_list(model, baseline_files)
            np.save(BASELINE_CACHE, baseline_emb)
            print(f"💾 Baseline embeddings saved to disk.")

        # --- INCOMING HANDLING ---
        run_incoming = True
        if INCOMING_CACHE.exists():
            print(f"\n📦 Found cached incoming embeddings at {INCOMING_CACHE.name}.")
            choice = input("❓ Incoming cache exists. Re-run feature extraction? (y/N): ").lower()
            if choice != 'y':
                incoming_emb = np.load(INCOMING_CACHE)
                if len(incoming_emb) != len(incoming_files):
                    print("⚠️  Cache mismatch (count). Forcing re-extraction...")
                    run_incoming = True
                else:
                    run_incoming = False
            
        if run_incoming:
            if model is None: model = detector.create_embedding_model()
            print(f"\n📸 Extracting {len(incoming_files)} incoming images...")
            incoming_emb = detector.extract_features_from_list(model, incoming_files)
            np.save(INCOMING_CACHE, incoming_emb)
            print(f"💾 Incoming embeddings saved to disk.")

        # --- ANALYSIS ---
        print("\n⚖️  Calculating multi-metric distribution divergence...")
        drift_prob, metrics_breakdown = analyzer.analyze_drift(baseline_emb, incoming_emb)
        
        final_percentage = drift_prob * 100
        print_drift_report(final_percentage, metrics_breakdown, len(baseline_files), len(incoming_files))
        
        keras.backend.clear_session()
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_drift_check(path_arg)
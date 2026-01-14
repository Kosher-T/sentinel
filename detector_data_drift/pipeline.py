import os
import sys
import numpy as np
import keras
from datetime import datetime
from pathlib import Path

# Setup paths and handle project root for imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Internal Sentinel Module Imports
try:
    import detector_data_drift.extractor as detector
    import detector_data_drift.analyzer as analyzer
except ImportError:
    from . import extractor as detector
    from . import analyzer as analyzer

import all_config as config

# Define persistence path
EMBEDDINGS_DIR = config.BASE_DATA_DIR / "data_drift" / "embeddings"
BASELINE_CACHE = EMBEDDINGS_DIR / "baseline_embeddings.npy"
INCOMING_CACHE = EMBEDDINGS_DIR / "incoming_embeddings.npy"

def ensure_dirs():
    """Ensures the embedding directory exists."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def print_drift_report(score, metrics_breakdown, num_baseline, num_incoming):
    """Prints a clean, formatted report to the console."""
    threshold = getattr(config, 'DRIFT_THRESHOLD', 25.0)
    status = "FAIL" if score > threshold else "PASS"
    
    color_icon = "🔴" if status == "FAIL" else "🟢"
    
    print("-" * 30)
    print(f"📡 DATA DRIFT REPORT")
    print("-" * 30)
    print(f"Status:      {color_icon} {status}")
    print(f"Drift Score: {score:.2f}%")
    print(f"Baseline:    {num_baseline} samples")
    print(f"Incoming:    {num_incoming} samples")
    print("-" * 30)
    print("Metrics Breakdown:")
    for m, v in metrics_breakdown.items():
        print(f" -> {m}: {v:.4f}")
    print("-" * 30)
    return status

def run_drift_analysis(baseline_path, incoming_path, force_recalc=False, latent_model_path=None):
    """
    Executes the drift detection suite. 
    Can use a specific distilled latent model if provided.
    """
    ensure_dirs()
    
    try:
        # Check image counts for cache validation
        baseline_files = detector.get_recursive_image_paths(baseline_path)
        incoming_files = detector.get_recursive_image_paths(incoming_path)
        
        if not baseline_files or not incoming_files:
            print("🔴 Error: One of the data paths is empty or contains no supported files.")
            return None, "ERROR"

        # --- BASELINE EMBEDDINGS ---
        model = None
        run_baseline = force_recalc or not BASELINE_CACHE.exists()
        
        if not run_baseline:
            baseline_emb = np.load(BASELINE_CACHE)
            if len(baseline_emb) != len(baseline_files):
                print("⚠️  Baseline cache count mismatch. Re-extracting...")
                run_baseline = True
        
        if run_baseline:
            if latent_model_path:
                print(f"🟢 Loading Latent Distiller Model: {Path(latent_model_path).name}")
                model = keras.models.load_model(latent_model_path, compile=False, safe_mode=False) # type: ignore
            else:
                # If no latent model path is provided, extractor must handle default model creation
                # Note: If extract_features requires a model, it must be passed here.
                # We assume extract_features handles the logic.
                model = None 
            
            print(f"📸 Extracting features for {len(baseline_files)} baseline samples...")
            # Using the universal entry point 'extract_features'
            baseline_emb = detector.extract_features(model, baseline_path)
            np.save(BASELINE_CACHE, baseline_emb)
            print(f"💾 Baseline embeddings saved.")

        # --- INCOMING EMBEDDINGS ---
        run_incoming = True
        if not force_recalc and INCOMING_CACHE.exists():
            incoming_emb = np.load(INCOMING_CACHE)
            if len(incoming_emb) == len(incoming_files):
                run_incoming = False
            
        if run_incoming:
            if model is None and latent_model_path:
                print(f"🟢 Loading Latent Distiller Model: {Path(latent_model_path).name}")
                model = keras.models.load_model(latent_model_path, compile=False, safe_mode=False) # type: ignore
                    
            print(f"📸 Extracting features for {len(incoming_files)} incoming samples...")
            # Using the universal entry point 'extract_features'
            incoming_emb = detector.extract_features(model, incoming_path)
            np.save(INCOMING_CACHE, incoming_emb)
            print(f"💾 Incoming embeddings saved.")

        # --- ANALYSIS ---
        print("\n⚖️  Calculating multi-metric distribution divergence...")
        drift_prob, metrics_breakdown = analyzer.analyze_drift(baseline_emb, incoming_emb)
        
        final_percentage = drift_prob * 100
        status = print_drift_report(final_percentage, metrics_breakdown, len(baseline_files), len(incoming_files))
        
        keras.backend.clear_session()
        return final_percentage, status
        
    except Exception as e:
        print(f"\n🔴 Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, "ERROR"

if __name__ == "__main__":
    # Test run
    baseline = config.ORIGINAL_DATA_PATH
    incoming = config.INCOMING_DATA_PATH
    
    # Check for distilled production model in the config-defined directory
    distilled_prod = list(config.PRODUCTION_DISTILLED_DIR.glob(f"*{config.DISTILL_SUFFIX}.keras"))
    prod_path = str(distilled_prod[0]) if distilled_prod else None
    
    run_drift_analysis(baseline, incoming, force_recalc=True, latent_model_path=prod_path)
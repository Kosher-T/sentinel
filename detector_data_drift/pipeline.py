import os
import sys
import numpy as np
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

def detect_domain(directory):
    """
    Scans the directory to determine the data domain (IMAGE, TEXT, etc).
    """
    path = Path(directory)
    for f in path.rglob('*'):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                return "IMAGE"
            if ext in ['.txt', '.json', '.csv', '.md']:
                return "TEXT"
    return "UNKNOWN"

def prepare_data_groups(directory, stack_size, domain):
    """
    Retrieves file paths and chunks them into groups based on specs provided by extractor.
    """
    if domain == "IMAGE":
        raw_paths = detector.get_recursive_image_paths(directory)
    else:
        raw_paths = [str(p) for p in Path(directory).rglob('*') if p.is_file()]
    
    raw_paths = sorted(raw_paths)
    
    if stack_size <= 1:
        return raw_paths
        
    grouped_data = []
    for i in range(0, len(raw_paths), stack_size):
        chunk = raw_paths[i:i + stack_size]
        if len(chunk) == stack_size:
            grouped_data.append(chunk)
            
    return grouped_data

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
    print(f"Baseline:    {num_baseline} samples (stacks/units)")
    print(f"Incoming:    {num_incoming} samples (stacks/units)")
    print("-" * 30)
    print("Metrics Breakdown:")
    for m, v in metrics_breakdown.items():
        print(f" -> {m}: {v:.4f}")
    print("-" * 30)
    return status

def run_drift_analysis(baseline_path, incoming_path, force_recalc=False, latent_model_path=None):
    """
    Executes the drift detection suite. 
    Pipeline manages data prep; Extractor manages the model and feature generation.
    """
    ensure_dirs()
    
    # 🔴 To avoid memory fragmentation, we use the extractor's model instance directly
    model_instance = None
    
    try:
        # --- PHASE 1: SURVEY & SPECIFICATION ---
        domain = detect_domain(baseline_path)
        print(f"🔍 Detected Data Domain: {domain}")

        # Request specs and model instance from Extractor
        print(f"🛰️  Requesting model specifications from Extractor...")
        model_instance, specs = detector.get_model_specs(latent_model_path)
        
        if specs is None:
            print("🔴 Error: Could not retrieve model specifications. Aborting.")
            return None, "ERROR"

        stack_size = specs.get("stack_size", 1)
        if stack_size > 1:
            print(f"🧠 Smart Stacking Activated: Grouping {stack_size} units per input.")

        # --- PHASE 2: DATA PREPARATION ---
        baseline_groups = prepare_data_groups(baseline_path, stack_size, domain)
        incoming_groups = prepare_data_groups(incoming_path, stack_size, domain)
        
        if not baseline_groups or not incoming_groups:
            print("🔴 Error: One of the data paths is empty or yielded no valid groups.")
            return None, "ERROR"

        # --- PHASE 3: EXECUTION (EXTRACTOR) ---
        run_baseline = force_recalc or not BASELINE_CACHE.exists()
        
        if not run_baseline:
            baseline_emb = np.load(BASELINE_CACHE)
            if len(baseline_emb) != len(baseline_groups):
                print("⚠️  Baseline cache count mismatch. Re-extracting...")
                run_baseline = True
        
        if run_baseline:
            print(f"📸 Extracting features for {len(baseline_groups)} baseline groups...")
            # Use the already loaded model_instance and specs
            baseline_emb = detector.extract_features(model_instance, baseline_groups, specs)
            np.save(BASELINE_CACHE, baseline_emb)
            print(f"💾 Baseline embeddings saved.")

        run_incoming = True
        if not force_recalc and INCOMING_CACHE.exists():
            incoming_emb = np.load(INCOMING_CACHE)
            if len(incoming_emb) == len(incoming_groups):
                run_incoming = False
            
        if run_incoming:
            print(f"📸 Extracting features for {len(incoming_groups)} incoming groups...")
            incoming_emb = detector.extract_features(model_instance, incoming_groups, specs)
            np.save(INCOMING_CACHE, incoming_emb)
            print(f"💾 Incoming embeddings saved.")

        # --- PHASE 4: ANALYSIS ---
        print("\n⚖️  Calculating multi-metric distribution divergence...")
        drift_prob, metrics_breakdown = analyzer.analyze_drift(baseline_emb, incoming_emb) # type: ignore
        
        final_percentage = drift_prob * 100
        status = print_drift_report(final_percentage, metrics_breakdown, len(baseline_groups), len(incoming_groups))
        
        return final_percentage, status
        
    except Exception as e:
        print(f"\n🔴 Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, "ERROR"
    
    finally:
        # Cleanup model instance and clear Keras session at the very end
        if model_instance is not None:
            import keras
            del model_instance
            keras.backend.clear_session()

def run_drift_check():
    """
    Sentinel Watch Wrapper: Automatically locates the latest distilled 
    production model and runs analysis using configured paths.
    """
    baseline = config.ORIGINAL_DATA_PATH
    incoming = config.INCOMING_DATA_PATH
    
    # 🔍 Locate the latest distilled production model
    distilled_models = list(config.PRODUCTION_DISTILLED_DIR.glob(f"*{config.DISTILL_SUFFIX}.keras"))
    
    if distilled_models:
        # Sort by modification time to get the newest model
        distilled_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        prod_path = str(distilled_models[0])
        print(f"📡 Sentinel: Using latest distilled model: {distilled_models[0].name}")
    else:
        prod_path = None
        print("⚠️ Sentinel: No distilled production model found. Using default extractor.")

    # Execute the curated analysis pipeline
    # force_recalc=True ensures we always look at the fresh 24-hour window
    return run_drift_analysis(baseline, incoming, force_recalc=True, latent_model_path=prod_path)

if __name__ == "__main__":
    # Your manual testing block remains the same
    # But now it could also just call run_drift_check()
    run_drift_check()
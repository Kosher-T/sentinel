import os
import sys
import numpy as np
import keras
from datetime import datetime
from pathlib import Path

# Sentinel Module Imports
import feature_extractor as detector
import data_analyzer as analyzer

# Load global configuration
try:
    import all_config as config
except ImportError:
    # Minimal fallback for standalone testing
    class InternalConfig:
        ORIGINAL_DATA_PATH = Path("data/data_drift/original_dataset/")
        INCOMING_DATA_PATH = Path("data/data_drift/incoming_data/")
        DRIFT_THRESHOLD = 30.0
        EMBEDDING_MODEL_TYPE = "MobileNetV2"
        EMBEDDING_INPUT_SHAPE = (224, 224, 3)
    config = InternalConfig()

def print_drift_report(score, num_baseline, num_incoming):
    """Prints a clean, formatted report to the console."""
    status = "FAIL" if score > config.DRIFT_THRESHOLD else "PASS"
    color_code = "\033[91m" if status == "FAIL" else "\033[92m"
    reset = "\033[0m"

    print("\n" + "="*50)
    print("         SENTINEL DATA DRIFT REPORT")
    print("="*50)
    print(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backbone:         {config.EMBEDDING_MODEL_TYPE}")
    print("-" * 50)
    print(f"Baseline Samples: {num_baseline}")
    print(f"Incoming Samples: {num_incoming}")
    print("-" * 50)
    print(f"Drift Score:      {score:.2f}%")
    print(f"Threshold:        {config.DRIFT_THRESHOLD}%")
    print(f"Status:           {color_code}{status}{reset}")
    print("="*50)
    
    if status == "FAIL":
        print("🚨 WARNING: High divergence detected. The incoming data")
        print("distribution differs significantly from training data.")
    else:
        print("✅ SUCCESS: Data distributions are within safety margins.")
    print("="*50 + "\n")

def run_drift_pipeline(incoming_subdir=None):
    """
    Core Drift Pipeline:
    Extracts features from the original dataset and new data,
    then compares them using the analyzer.
    """
    # 1. Setup paths
    incoming_path = config.INCOMING_DATA_PATH
    if incoming_subdir:
        incoming_path = incoming_path / incoming_subdir

    print(f"🚀 Initializing Drift Analysis...")
    print(f"📂 Baseline: {config.ORIGINAL_DATA_PATH}")
    print(f"📂 Incoming: {incoming_path}")

    # 2. Get Image Paths
    baseline_files = detector.get_image_paths(config.ORIGINAL_DATA_PATH)
    incoming_files = detector.get_image_paths(incoming_path)

    if not incoming_files:
        print(f"❌ Error: No images found in {incoming_path}.")
        return

    # 3. Load Model & Extract
    # Note: Using create_embedding_model(pooling='avg') logic from extractor
    model = detector.create_embedding_model()
    
    print(f"📸 Extracting features from {len(baseline_files)} baseline images...")
    baseline_emb = detector.generate_embeddings_from_directory(model, config.ORIGINAL_DATA_PATH)
    
    print(f"📸 Extracting features from {len(incoming_files)} incoming images...")
    incoming_emb = detector.generate_embeddings_from_directory(model, incoming_path)

    # 4. Analyze & Report
    print("⚖️  Calculating distribution divergence (Wasserstein Distance)...")
    try:
        drift_score = analyzer.analyze_drift(baseline_emb, incoming_emb)
        print_drift_report(drift_score, len(baseline_files), len(incoming_files))
        
        # Cleanup Keras session to free memory
        keras.backend.clear_session()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    # Optional subdirectory argument
    target_sub = sys.argv[1] if len(sys.argv) > 1 else None
    run_drift_pipeline(target_sub)
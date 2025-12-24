# This module defines the parameters for the VFI model being monitored.
# The idea is that if you switch to a different model (e.g., VGG, Custom CNN),
# you only change this file and the embedding generation code.

from pathlib import Path
import os

EMBEDDING_MODEL_TYPE = "VGG16"
EMBEDDING_INPUT_SHAPE = (224, 224, 3) # Required input shape for the model
EMBEDDING_FEATURE_COUNT = 512 # Output feature vector length (MobileNetV2's final dense layer size)

# --- PROJECT ROOT ---
# This allows the config to work regardless of which subfolder a script is run from
PROJECT_ROOT = Path(__file__).resolve().parent

# --- DATA SOURCE PATHS (THE GOLDEN SET) ---
GOLDEN_SET_DIR = PROJECT_ROOT / "data" / "golden_set_septuplets"
BASE_DATA_DIR = PROJECT_ROOT / "data"

# --- MONITORING ROOTS ---
MODEL_DECAY_ROOT = PROJECT_ROOT / "data" / "monitoring" / "decay"
DRIFT_MONITOR_ROOT = PROJECT_ROOT / "data" / "monitoring" / "drift"

# --- MODEL PATHS ---
# Directories where the latest 'Fresh' and 'Old' models are stored
OLD_MODEL_DIR = PROJECT_ROOT / "models" / "old_model"
FRESH_MODEL_DIR = PROJECT_ROOT / "models" / "fresh_model"

# Specific fallback paths for legacy scripts
OLD_MODEL_PATH = OLD_MODEL_DIR / "vfi_model_old.keras"
FRESH_MODEL_PATH = FRESH_MODEL_DIR / "vfi_model_fresh.keras"

# --- RESULTS & EMBEDDINGS ---
FRESH_RESULTS_DIR = MODEL_DECAY_ROOT / "results" / "fresh"
OLD_RESULTS_DIR = MODEL_DECAY_ROOT / "results" / "old"
EMBEDDINGS_ROOT = MODEL_DECAY_ROOT / "embeddings"

# --- DRIFT & DECAY THRESHOLDS ---
DECAY_THRESHOLD = 15.0  # % drop in Golden Set performance
DRIFT_THRESHOLD = 30.0  # Wasserstein distance limit for incoming data

# --- CONTINUOUS MONITORING SETTINGS ---
MONITOR_SCHEDULE = "0 */6 * * *" # Every 6 hours
RETRAIN_TRIGGER_COUNT = 5       # Consecutive fails to trigger retrain
DRIFT_FAILURE_RATIO = 0.6       # 60% failure in recent window triggers alert

# --- DATABASE & LOGGING (For Dashboard) ---
DRIFT_HISTORY_DB = DRIFT_MONITOR_ROOT / "drift_history.json"
RETRAIN_LOG = DRIFT_MONITOR_ROOT / "retrain_events.json"
MODEL_HISTORY_FILE = MODEL_DECAY_ROOT / "model_run_history.json"

# --- HARDWARE ---
CUDA_VISIBLE_DEVICES = "-1" # Force CPU for local dev/monitoring
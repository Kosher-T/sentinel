# This module defines the parameters for the model being monitored.
# If you want to switch backbones (e.g., from MobileNetV2 to VGG16), update this file.

from pathlib import Path
import os

# --- EMBEDDING BACKBONE SETTINGS ---
# Used by both drift_pipeline and decay_pipeline for feature extraction
EMBEDDING_MODEL_TYPE = "VGG16"
EMBEDDING_INPUT_SHAPE = (224, 224, 3) 
EMBEDDING_FEATURE_COUNT = 512 # VGG16 Global Average Pooling output size

# --- PROJECT ROOT ---
# Resolves the absolute path to the project folder
PROJECT_ROOT = Path(__file__).resolve().parent

# --- DATA SOURCE PATHS (THE GOLDEN SET) ---
# The 'Golden Set' is the ground-truth benchmark for model performance
BASE_DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_SET_DIR = BASE_DATA_DIR / "golden_set_septuplets"

# --- DRIFT SPECIFIC PATHS ---
# Where original training data and incoming production data are stored
TRAINING_DATA_PATH = BASE_DATA_DIR / "data_drift" / "training_data"
INCOMING_DATA_PATH = BASE_DATA_DIR / "data_drift" / "incoming_data"
ARCHIVED_DATA_PATH = BASE_DATA_DIR / "data_drift" / "history" # New: for archiving processed data

# --- MONITORING ROOTS ---
# Centralized locations for persistence and logs
MODEL_DECAY_ROOT = PROJECT_ROOT / "data" / "model_decay"
DRIFT_MONITOR_ROOT = PROJECT_ROOT / "data" / "data_drift"

# --- MODEL PATHS ---
# Directories where the latest 'Fresh' and 'Old' models are stored for decay comparison
MODEL_PATH = PROJECT_ROOT / "models"
OLD_MODEL_PATH = MODEL_PATH / "production"
FRESH_MODEL_PATH = MODEL_PATH / "challenger"

# --- RESULTS & EMBEDDINGS (DECAY PIPELINE) ---
# Intermediate outputs for the decay pipeline
FRESH_RESULTS_DIR = MODEL_DECAY_ROOT / "fresh_model_results"
OLD_RESULTS_DIR = MODEL_DECAY_ROOT / "old_model_results"
EMBEDDINGS_ROOT = MODEL_DECAY_ROOT / "embeddings"

# --- DRIFT & DECAY THRESHOLDS ---
DECAY_THRESHOLD = 5.0  # % drop in Golden Set performance (Wasserstein distance)
DRIFT_THRESHOLD = 26.4  # % distance limit for incoming production data

# --- CONTINUOUS MONITORING SETTINGS ---
MONITOR_SCHEDULE = "0 */6 * * *" # Cron: Every 6 hours
RETRAIN_TRIGGER_COUNT = 3        # Consecutive fails required to trigger an automated retrain
DRIFT_FAILURE_RATIO = 0.8        # 80% failure in recent window triggers system alert
TIMEFRAME_WINDOW = 5             # Number of past timeframes to consider for the failure ratio

# --- EXECUTION ENGINE SETTINGS ---
# Determines the order of failover for training jobs
EXECUTION_DRIVERS_PRIORITY = ["LOCAL"] # Options: "LOCAL", "AWS", "GCP"
RETRAINING_SCRIPT = PROJECT_ROOT / "mock_train.py"
EXECUTION_TIMEOUT = 3600               # 1 hour in seconds
EXPECTED_CHALLENGER_PATH = FRESH_MODEL_PATH / "challenger_v2.pth"

# --- SAFETY & HARDWARE GOVERNOR ---
# Forces single-GPU to avoid MirroredStrategy NaN issues
FORCE_SINGLE_GPU = True 
# Keywords to search for in training scripts during pre-flight check
DANGEROUS_STRATEGIES = ["MirroredStrategy", "MultiWorkerMirroredStrategy", "CollectiveAllReduceStrategy"]
# Fail-fast if these patterns are detected in logs
CRITICAL_LOG_ERRORS = ["nan", "inf", "Out of memory", "CUDA_ERROR_OUT_OF_MEMORY"]

# --- DATABASE & LOGGING (For Dashboard & Persistence) ---
DRIFT_HISTORY_DB = DRIFT_MONITOR_ROOT / "drift_history.db" # Updated to .db to match dashboard
RETRAIN_LOG = DRIFT_MONITOR_ROOT / "retrain_events.json"
MODEL_HISTORY_FILE = MODEL_DECAY_ROOT / "model_run_history.json"
MODEL_REGISTRY_DB = DRIFT_MONITOR_ROOT / "model_registry.db"

# --- HARDWARE ---
# Forces CPU for monitoring tasks to avoid interrupting heavy GPU training sessions
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
CUDA_VISIBLE_DEVICES = "-1"

# --- DISTILLER CONFIGURATION ---
# Directories for distilled (latent space) models
PRODUCTION_DISTILLED_DIR = MODEL_PATH / "production_distilled"
CHALLENGER_DISTILLED_DIR = MODEL_PATH / "challenger_distilled"

# Ensure they exist
PRODUCTION_DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
CHALLENGER_DISTILLED_DIR.mkdir(parents=True, exist_ok=True)

# Distiller Settings
DISTILL_SUFFIX = "_latent"
POLLING_INTERVAL = 180  # Seconds between folder scans
STABILITY_DELAY = 2   # Seconds to wait for file size stability

# Mapping of original folders to distilled folders for the Distiller to watch
DISTILL_MAP = {
    str(OLD_MODEL_PATH): str(PRODUCTION_DISTILLED_DIR),
    str(FRESH_MODEL_PATH): str(CHALLENGER_DISTILLED_DIR)
}

# --- ALERT ESCALATION SETTINGS ---
ESCALATION_TIMEOUT_MINUTES = 15        # Minutes before first escalation
ESCALATION_FINAL_TIMEOUT_MINUTES = 30  # Minutes before final escalation
SECONDARY_ONCALL_EMAIL = ""            # Secondary on-call (configured via setup)
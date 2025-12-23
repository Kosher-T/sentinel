# This module defines the parameters for the VFI model being monitored.
# The idea is that if you switch to a different model (e.g., VGG, Custom CNN),
# you only change this file and the embedding generation code.

from pathlib import Path
import os


EMBEDDING_MODEL_TYPE = "VGG16"
EMBEDDING_INPUT_SHAPE = (224, 224, 3) # Required input shape for the model
EMBEDDING_FEATURE_COUNT = 512 # Output feature vector length (MobileNetV2's final dense layer size)

# --- PROJECT ROOT CALCULATION ---
# Assumes this file is located in sentinel/detector_model_decay/
# which it definitely is and always will be.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- DATA SOURCE PATHS ---
BASE_DATA_DIR = PROJECT_ROOT / "data" / "golden_set_septuplets"

# --- MODEL PATHS ---
OLD_MODEL_PATH = BASE_DATA_DIR / "models" / "old_model" / "vfi_septuplet_epoch_31.keras"
FRESH_MODEL_PATH = BASE_DATA_DIR / "models" / "fresh_model" / "vfi_septuplet_epoch_35.keras"

# --- OUTPUT PATHS ---
MODEL_DECAY_ROOT = PROJECT_ROOT / "data" / "model_decay"
OLD_RESULTS_DIR = MODEL_DECAY_ROOT / "old_model_results"
FRESH_RESULTS_DIR = MODEL_DECAY_ROOT / "fresh_model_results"
EMBEDDINGS_ROOT = MODEL_DECAY_ROOT / "embeddings"

# --- SETTINGS ---
# Environment settings can go here too if needed later
CUDA_VISIBLE_DEVICES = "-1" # Force CPU for local dev
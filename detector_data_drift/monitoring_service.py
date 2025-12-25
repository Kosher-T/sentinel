import os
import sys
import sqlite3
import datetime
import numpy as np
from pathlib import Path

# --- DYNAMIC PATH RESOLUTION ---
# Resolves the project root where all_config.py now resides
# Structure: project_root/detector_data_drift/monitoring_service.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 1. Ensure the project root is in sys.path to find all_config.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Ensure CURRENT_DIR is in sys.path to find local modules (feature_extractor, etc.)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Attempt to load the consolidated configuration from the project root
try:
    import all_config as config
    print("✅ Successfully loaded all_config from project root")
except ImportError:
    print("⚠️ Warning: all_config.py not found in root. Using environment defaults.")
    config = None

# --- LOCAL IMPORTS ---
# We import directly since CURRENT_DIR is now in the path
try:
    import feature_extractor as detector
except ImportError:
    print("❌ Critical Error: Could not find feature_extractor.py")
    sys.exit(1)

try:
    # Look for drift_analyzer in the neighboring folder
    from drift_analyzer import analyze_drift
except ImportError:
    # Fallback if the file is moved or in the parent folder
    def analyze_drift(b, n): return 0.0

# --- CONFIGURATION MAPPING ---
NEW_DATA_PATH = Path(os.environ.get("NEW_DATA_PATH", "/app/incoming_data"))

# Use the root from all_config if available, else a default relative to script
DEFAULT_OUTPUT = getattr(config, 'DRIFT_MONITOR_ROOT', PROJECT_ROOT / "data" / "monitoring" / "drift") if config else CURRENT_DIR / "status_output"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT)))

# File paths for persistence
DB_PATH = OUTPUT_DIR / "drift_history.db"
BASELINE_PATH = OUTPUT_DIR / "baseline_embeddings.npy"
STATUS_PATH = OUTPUT_DIR / "status.txt"
SCORE_PATH = OUTPUT_DIR / "score.txt"

# Thresholds: Environment variable > all_config > hardcoded default
try:
    default_threshold = getattr(config, 'DRIFT_THRESHOLD', 30.0) if config else 30.0
    DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", default_threshold))
except (ValueError, TypeError):
    DRIFT_THRESHOLD = 30.0

MIN_SAMPLES_FOR_CHECK = 50 

def init_db():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                 (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL)''')
    conn.commit()
    conn.close()

def log_to_db(score, status, threshold):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO drift_logs VALUES (?, ?, ?, ?)", 
              (timestamp, score, status, threshold))
    conn.commit()
    conn.close()

def get_total_image_count(directory):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if not directory.exists(): return 0
    return sum(1 for p in directory.rglob('*') if p.suffix.lower() in valid_exts)

def check_for_drift():
    print(f"\n--- 🛰️ SENTINEL MONITORING JOB (Threshold: {DRIFT_THRESHOLD}%) ---")
    init_db()
    
    # 1. Baseline Handling
    if not BASELINE_PATH.exists():
        print(f"1. Baseline NOT found. Generating from {NEW_DATA_PATH}...")
        num_samples = get_total_image_count(NEW_DATA_PATH)
        if num_samples == 0:
            print(f"❌ ERROR: No data found at {NEW_DATA_PATH}.")
            return
            
        model = detector.create_embedding_model()
        image_list = [str(p) for p in NEW_DATA_PATH.rglob('*') if p.suffix.lower() in {'.webp', '.png', '.jpg'}]
        baseline = detector.extract_features(model, image_list)
        
        np.save(str(BASELINE_PATH), baseline)
        print(f"✅ Baseline saved. Samples: {baseline.shape[0]}")
        
        with open(SCORE_PATH, "w") as f: f.write("0.00")
        with open(STATUS_PATH, "w") as f: f.write("PASS")
        return

    # 2. Monitoring Flow
    baseline = np.load(str(BASELINE_PATH))
    num_samples = get_total_image_count(NEW_DATA_PATH)

    if num_samples < MIN_SAMPLES_FOR_CHECK:
        print(f"⚠️ Insufficient data ({num_samples}/{MIN_SAMPLES_FOR_CHECK}). Skipping.")
        with open(SCORE_PATH, "w") as f: f.write("0.00")
        with open(STATUS_PATH, "w") as f: f.write("PASS")
        return

    # 3. Process New Data
    print(f"2. Analyzing {num_samples} new samples...")
    model = detector.create_embedding_model()
    image_list = [str(p) for p in NEW_DATA_PATH.rglob('*') if p.suffix.lower() in {'.webp', '.png', '.jpg'}]
    new_embeddings = detector.extract_features(model, image_list)
    
    # 4. Analyze
    score = analyze_drift(baseline, new_embeddings)
    status = "FAIL" if score > DRIFT_THRESHOLD else "PASS"
    
    print(f"\n>>> 📊 DRIFT SCORE: {score:.2f}% | STATUS: {status}")

    # 5. Persistent Outputs
    with open(SCORE_PATH, "w") as f: f.write(f"{score:.2f}")
    with open(STATUS_PATH, "w") as f: f.write(status)
    log_to_db(score, status, DRIFT_THRESHOLD)

if __name__ == "__main__":
    check_for_drift()
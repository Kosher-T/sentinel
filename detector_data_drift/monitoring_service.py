import os
import sys
import sqlite3
import datetime
import numpy as np
from pathlib import Path

# --- ROBUST PATH RESOLUTION ---
# Get the absolute path of the directory containing this script
# Structure: [PROJECT_ROOT]/detector_data_drift/monitoring_service.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Priority 1: Add PROJECT_ROOT to sys.path so 'import all_config' works
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Priority 2: Add CURRENT_DIR to sys.path so local modules work without package prefix
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --- CONFIGURATION LOADING ---
try:
    import all_config as config
    print(f"✅ Loaded all_config.py from: {PROJECT_ROOT}")
except ImportError:
    print(f"⚠️ Warning: all_config.py not found at {PROJECT_ROOT}. Using environment defaults.")
    config = None

# --- LOCAL MODULE IMPORTS ---
try:
    # Import directly from the same folder
    import feature_extractor as detector
    print("✅ Loaded feature_extractor.py")
except ImportError:
    print(f"❌ Critical Error: Could not find feature_extractor.py in {CURRENT_DIR}")
    sys.exit(1)

try:
    # Try to find drift_analyzer in the neighboring folder or project root
    try:
        from detector_model_decay.drift_analyzer import analyze_drift
    except ImportError:
        from drift_analyzer import analyze_drift
    print("✅ Loaded drift_analyzer")
except ImportError:
    print("⚠️ Warning: drift_analyzer not found. Using fallback logic.")
    def analyze_drift(b, n): return 0.0

# --- PATH & THRESHOLD MAPPING ---
NEW_DATA_PATH = Path(os.environ.get("NEW_DATA_PATH", "/app/incoming_data"))

# Use paths from config if they exist, otherwise use defaults
DEFAULT_OUTPUT = getattr(config, 'DRIFT_MONITOR_ROOT', PROJECT_ROOT / "data" / "monitoring" / "drift") if config else CURRENT_DIR / "status_output"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT)))

# Persistence files
DB_PATH = OUTPUT_DIR / "drift_history.db"
BASELINE_PATH = OUTPUT_DIR / "baseline_embeddings.npy"
STATUS_PATH = OUTPUT_DIR / "status.txt"
SCORE_PATH = OUTPUT_DIR / "score.txt"

# Threshold priority: Environment Var > all_config.py > Hardcoded Default
try:
    default_threshold = getattr(config, 'DRIFT_THRESHOLD', 30.0) if config else 30.0
    DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", default_threshold))
except (ValueError, TypeError):
    DRIFT_THRESHOLD = 30.0

MIN_SAMPLES_FOR_CHECK = 50 

def init_db():
    """Initializes SQLite database for drift history."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                 (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL)''')
    conn.commit()
    conn.close()

def log_to_db(score, status, threshold):
    """Logs the monitoring result."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO drift_logs VALUES (?, ?, ?, ?)", 
              (timestamp, score, status, threshold))
    conn.commit()
    conn.close()

def get_total_image_count(directory):
    """Counts images in the target directory."""
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
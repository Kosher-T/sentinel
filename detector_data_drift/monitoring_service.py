import os
import sys
import sqlite3
import datetime
import numpy as np
from pathlib import Path

# --- DYNAMIC PATH RESOLUTION ---
# Ensures the script can find its sibling modules and parent packages
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "detector_data_drift" else CURRENT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Imports using the established package structure
try:
    import detector_data_drift.feature_extractor as detector
    from detector_model_decay.drift_analyzer import analyze_drift
except ImportError:
    # Fallback for localized execution
    import feature_extractor as detector
    from drift_analyzer import analyze_drift

# --- CONFIGURATION ---
# Use Path objects for better cross-platform compatibility
NEW_DATA_PATH = Path(os.environ.get("NEW_DATA_PATH", "/app/incoming_data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/status_output"))

# Database and file paths
DB_PATH = OUTPUT_DIR / "drift_history.db"
BASELINE_PATH = OUTPUT_DIR / "baseline_embeddings.npy"
STATUS_PATH = OUTPUT_DIR / "status.txt"
SCORE_PATH = OUTPUT_DIR / "score.txt"

# Thresholds and constraints
MIN_SAMPLES_FOR_CHECK = 50 
try:
    DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "30.0"))
except ValueError:
    DRIFT_THRESHOLD = 30.0

def init_db():
    """Creates the database table if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                 (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL)''')
    conn.commit()
    conn.close()

def log_to_db(score, status, threshold):
    """Saves the result to the history database."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO drift_logs VALUES (?, ?, ?, ?)", 
              (timestamp, score, status, threshold))
    conn.commit()
    conn.close()
    print(f"✅ Logged result to {DB_PATH}")

def get_total_image_count(directory):
    """Recursively counts image files in the directory."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    for path in directory.rglob('*'):
        if path.suffix.lower() in valid_exts:
            count += 1
    return count

def check_for_drift():
    print(f"\n--- 🛰️ SENTINEL MONITORING JOB (Threshold: {DRIFT_THRESHOLD}%) ---")
    init_db()
    
    # 1. Baseline Handling
    if not BASELINE_PATH.exists():
        print(f"1. Baseline NOT found. Generating new baseline from {NEW_DATA_PATH}...")
        
        num_samples = get_total_image_count(NEW_DATA_PATH)
        if num_samples == 0:
            print(f"❌ CRITICAL ERROR: No data found at {NEW_DATA_PATH}.")
            sys.exit(1)
            
        model = detector.create_embedding_model()
        baseline = detector.extract_features(model, [str(p) for p in NEW_DATA_PATH.rglob('*') if p.suffix.lower() in {'.webp', '.png', '.jpg'}])
        
        if baseline.size == 0:
            print(f"❌ CRITICAL ERROR: Failed to generate baseline embeddings.")
            sys.exit(1)
            
        np.save(str(BASELINE_PATH), baseline)
        print(f"✅ Baseline saved to {BASELINE_PATH}. Samples: {baseline.shape[0]}")
        
        # Write initial neutral outputs
        with open(SCORE_PATH, "w") as f: f.write("0.00")
        with open(STATUS_PATH, "w") as f: f.write("PASS")
        return

    # 2. Monitoring Flow
    print(f"1. Baseline loaded from {BASELINE_PATH}")
    baseline = np.load(str(BASELINE_PATH))

    num_samples = get_total_image_count(NEW_DATA_PATH)
    if num_samples < MIN_SAMPLES_FOR_CHECK:
        print(f"⚠️ Only {num_samples} samples found. Skipping check (Min: {MIN_SAMPLES_FOR_CHECK}).")
        log_to_db(0.0, "INSUFFICIENT_DATA", DRIFT_THRESHOLD)
        with open(SCORE_PATH, "w") as f: f.write("0.00")
        with open(STATUS_PATH, "w") as f: f.write("PASS")
        return

    # 3. Process New Data
    print(f"2. Extracting features from {num_samples} samples...")
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
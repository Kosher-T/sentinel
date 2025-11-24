import os
import sys
import sqlite3
import datetime
import numpy as np
import drift_detector as detector
import drift_analyzer as analyzer

# --- CONFIGURATION ---
NEW_DATA_PATH = os.environ.get("NEW_DATA_PATH", "/app/incoming_data")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "baseline_embeddings.npy")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/status_output")
DB_PATH = os.path.join(OUTPUT_DIR, "drift_history.db") # NEW: Database file

try:
    DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "30.0"))
except ValueError:
    DRIFT_THRESHOLD = 30.0

# Paths for legacy support (GitHub Actions still reads these)
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.txt") 
SCORE_PATH = os.path.join(OUTPUT_DIR, "score.txt")

def init_db():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                 (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL)''')
    conn.commit()
    conn.close()

def log_to_db(score, status, threshold):
    """Saves the result to the history database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO drift_logs VALUES (?, ?, ?, ?)", 
              (timestamp, score, status, threshold))
    conn.commit()
    conn.close()
    print(f"6. Logged result to {DB_PATH}")

def check_for_drift():
    print(f"--- STARTING MONITORING JOB (Threshold: {DRIFT_THRESHOLD}%) ---")
    
    # Ensure DB exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()
    
    # 1. Load Baseline
    if not os.path.exists(BASELINE_PATH):
        print(f"CRITICAL ERROR: Baseline file {BASELINE_PATH} not found.")
        sys.exit(1)
    baseline = np.load(BASELINE_PATH)

    # 2. Check New Data
    if not os.path.exists(NEW_DATA_PATH) or not os.listdir(NEW_DATA_PATH):
        print(f"Error: No data found at {NEW_DATA_PATH}.")
        sys.exit(1)

    # 3. Load Model
    model = detector.create_embedding_model()
    
    # 4. Generate Embeddings
    new_embeddings = detector.generate_embeddings_from_directory(model, NEW_DATA_PATH)
    
    # 5. Calculate Drift
    score, _ = analyzer.analyze_drift(baseline, new_embeddings)
    
    print(f"\n>>> DRIFT SCORE: {score:.2f}%")
    
    # Determine Verdict
    if score > DRIFT_THRESHOLD:
        status = "FAIL"
        print("[FAIL] HIGH DRIFT DETECTED!")
    else:
        status = "PASS"
        print("[PASS] System Normal.")

    # --- SAVE OUTPUTS ---
    # 1. Text files for GitHub Actions (The "Sticky Notes")
    with open(SCORE_PATH, "w") as f:
        f.write(f"{score:.2f}")
    with open(STATUS_PATH, "w") as f:
        f.write(status)

    # 2. Database for Dashboard (The "Diary")
    log_to_db(score, status, DRIFT_THRESHOLD)
    
    # Exit code logic
    sys.exit(0 if status == "PASS" else 0) # We exit 0 so the pipeline continues

if __name__ == "__main__":
    check_for_drift()
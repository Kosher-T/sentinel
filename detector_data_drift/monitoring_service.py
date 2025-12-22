import os
import sys
import sqlite3
import datetime
import numpy as np
import drift_detector as detector
import drift_analyzer as analyzer

# --- CONFIGURATION ---
NEW_DATA_PATH = os.environ.get("NEW_DATA_PATH", "/app/incoming_data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/status_output")
DB_PATH = os.path.join(OUTPUT_DIR, "drift_history.db")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "baseline_embeddings.npy") 
MIN_SAMPLES_FOR_CHECK = 50 # New constraint to prevent false positives on tiny batches

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

def get_total_image_count(directory):
    """
    Recursively counts image files in the directory and all subdirectories.
    Solves the issue where data inside subfolders (e.g. 1113(1)-1/) was ignored.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    count = 0
    # os.walk yields a 3-tuple (root, dirs, files) for every directory in the tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                count += 1
    return count

def check_for_drift():
    print(f"--- STARTING MONITORING JOB (Threshold: {DRIFT_THRESHOLD}%) ---")
    
    # Ensure DB exists and output directory is ready
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()
    
    # --- 1. Baseline Handling (Setup phase) ---
    if not os.path.exists(BASELINE_PATH):
        print(f"1. Baseline NOT found. Generating new baseline from {NEW_DATA_PATH}...")
        
        # Use recursive count logic
        num_baseline_samples = get_total_image_count(NEW_DATA_PATH)
        
        if num_baseline_samples == 0:
            print(f"CRITICAL ERROR: No data found at {NEW_DATA_PATH} (recursively checked). Cannot generate baseline.")
            sys.exit(1)
            
        model = detector.create_embedding_model()
        baseline = detector.generate_embeddings_from_directory(model, NEW_DATA_PATH)
        
        if baseline.size == 0:
            print(f"CRITICAL ERROR: Failed to generate baseline embeddings. Check data format.")
            sys.exit(1)
            
        # Save the generated baseline to the persistent volume
        np.save(BASELINE_PATH, baseline)
        print(f"1. Baseline embeddings successfully created and saved to {BASELINE_PATH}. Total samples: {baseline.shape[0]}")
        
        # We skip logging 0% to avoid dashboard spikes
        print("Initial baseline run complete. Monitoring will begin on next execution.")
        
        # --- SAVE OUTPUTS for initial run ---
        with open(SCORE_PATH, "w") as f:
            f.write(f"0.00")
        with open(STATUS_PATH, "w") as f:
            f.write("PASS")
        
        sys.exit(0) # Exit success after baseline creation

    # --- 2. Monitoring Flow (Check phase) ---
    
    print(f"1. Baseline found at {BASELINE_PATH}. Loading...")
    baseline = np.load(BASELINE_PATH)

    # Check for New Data files recursively
    num_samples = get_total_image_count(NEW_DATA_PATH)
    
    if num_samples == 0:
        print(f"Error: No image files found in {NEW_DATA_PATH} or its subdirectories.")
        sys.exit(1)
    
    if num_samples < MIN_SAMPLES_FOR_CHECK:
        status = "PASS"
        score = 0.0
        print(f"WARNING: Only {num_samples} samples found. Skipping drift check to prevent false positive (MIN required: {MIN_SAMPLES_FOR_CHECK}).")
        print("[PASS] Insufficient Data for Reliable Check.")
        log_to_db(score, "TOO_FEW_SAMPLES", DRIFT_THRESHOLD) # Log the skip status
        
        with open(SCORE_PATH, "w") as f:
            f.write(f"{score:.2f}")
        with open(STATUS_PATH, "w") as f:
            f.write(status)
        
        sys.exit(0) # Exit success after logging skip
    
    print(f"2. {num_samples} samples found. Proceeding with check.")

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
        print("[FAIL] HIGH DRIFT DETECTED! Remediation pipeline triggered.")
    else:
        status = "PASS"
        print("[PASS] System Normal.")

    # --- SAVE OUTPUTS ---
    # 1. Text files for GitHub Actions
    with open(SCORE_PATH, "w") as f:
        f.write(f"{score:.2f}")
    with open(STATUS_PATH, "w") as f:
        f.write(status)

    # 2. Database for Dashboard
    log_to_db(score, status, DRIFT_THRESHOLD)
    
    # Exit code logic
    sys.exit(0 if status == "PASS" else 0) # We exit 0 so the pipeline continues

if __name__ == "__main__":
    check_for_drift()
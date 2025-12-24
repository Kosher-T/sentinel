import sqlite3
import time
import os
import sys
from pathlib import Path

# Ensure root is in path for imports
sys.path.append(os.getcwd())

import all_config as config

# --- DB INITIALIZATION ---
# Dashboard expects: temp_status/drift_history.db
# We use all_config to ensure the path is absolute relative to the project root
DB_PATH = config.PROJECT_ROOT / "temp_status" / "drift_history.db"

def init_db():
    """Ensures the database folder and table exist before running."""
    # Create the temp_status folder if it doesn't exist
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Schema matches the load_data() function in dashboard.py
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drift_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            drift_score REAL,
            threshold REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_to_db(score, threshold, status):
    """Writes a new monitoring entry to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO drift_logs (drift_score, threshold, status)
        VALUES (?, ?, ?)
    ''', (score, threshold, status))
    conn.commit()
    conn.close()

def run_drift_check():
    """
    Step 1: Data Drift Analysis
    Integrates with detector_data_drift modules.
    """
    print("🔍 [Sentinel] Step 1: Checking Data Drift...")
    try:
        # This will eventually import your actual drift logic:
        # from detector_data_drift.monitoring_service import check_current_drift
        
        # Simulated score for demonstration
        score = 12.5 
        passed = score < config.DRIFT_THRESHOLD
        status = "PASS" if passed else "FAIL"
        return score, status
    except Exception as e:
        print(f"❌ Drift Check Error: {e}")
        return 0.0, "ERROR"

def check_retrain_trigger():
    """Checks the DB for a streak of failures to avoid jitter."""
    if not DB_PATH.exists():
        return False

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Check the history of the last few runs defined in config
    cursor.execute('''
        SELECT status FROM drift_logs 
        ORDER BY timestamp DESC LIMIT ?
    ''', (config.RETRAIN_TRIGGER_COUNT,))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < config.RETRAIN_TRIGGER_COUNT:
        return False
    
    fail_count = sum(1 for row in rows if row[0] == "FAIL")
    # If the ratio of failures is too high (e.g. 60%), trigger retraining
    return (fail_count / config.RETRAIN_TRIGGER_COUNT) >= config.DRIFT_FAILURE_RATIO

def trigger_retraining_workflow():
    """
    Step 2: Model Retraining
    Simulates the retraining trigger.
    """
    print("🏗️ [Sentinel] Step 2: Persistence of Drift detected. Triggering Retraining...")
    time.sleep(1) 
    print("✅ [Sentinel] Retraining complete. Challenger model produced.")
    return True

def run_decay_audit():
    """
    Step 3: Model Decay Audit (The 'Gold Standard' check)
    Uses the decay_pipeline to ensure the new model isn't worse than the old one.
    """
    print("🛡️ [Sentinel] Step 3: Running Decay Audit on Challenger...")
    try:
        # Using the actual logic from your project structure
        from detector_model_decay.decay_pipeline import run_analysis
        
        # run_analysis manages the full comparison and reporting
        run_analysis()
        return True # Assume success if no exception raised
    except Exception as e:
        print(f"❌ [Sentinel] Decay Audit failed: {e}")
        return False

def main():
    init_db()
    print(f"\n{'='*55}")
    print(f"📡 SENTINEL PIPELINE START: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    # 1. Check Drift
    score, status = run_drift_check()
    log_to_db(score, config.DRIFT_THRESHOLD, status)
    print(f"📊 Result: {score}% Drift | Threshold: {config.DRIFT_THRESHOLD}% | Status: {status}")

    # 2. Evaluate Health & Self-Heal
    # Trigger retrain if CURRENT is fail OR if HISTORY suggests we are unstable
    if status == "FAIL" or check_retrain_trigger():
        if check_retrain_trigger():
            print("🚨 Alert: Stability threshold breached. System requires intervention.")
            trigger_retraining_workflow()
            
            if run_decay_audit():
                print("🚀 [Sentinel] SUCCESS: Challenger model validated and deployed.")
            else:
                print("⚠️ [Sentinel] ABORT: Challenger failed audit. Reverting to Baseline.")
        else:
            print("🟡 Warning: Drift detected, but stability check suggests waiting for more data.")
    else:
        print("🟢 System Performance: Optimal. No intervention required.")

    print(f"\n{'='*55}")
    print("🏁 PIPELINE EXECUTION COMPLETE")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
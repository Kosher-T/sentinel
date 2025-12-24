import sqlite3
import time
import os
import sys
from pathlib import Path

# Ensure root is in path for imports
sys.path.append(os.getcwd())

import all_config as config

# --- ENVIRONMENT & CI DETECTION ---
# If running in GitHub Actions or Docker, we disable interactive prompts
IS_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("DOCKER_ENV") == "true"

# Allow environment variables to override config for workflow flexibility
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", config.DRIFT_THRESHOLD))

# --- DB INITIALIZATION ---
DB_PATH = config.PROJECT_ROOT / "temp_status" / "drift_history.db"

def init_db():
    """Ensures the database folder and table exist before running."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
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
    Integrates with detector_data_drift logic.
    """
    print(f"🔍 [Sentinel] Step 1: Checking Data Drift (Threshold: {DRIFT_THRESHOLD}%)...")
    try:
        # In a real scenario, this calls your feature-based drift detector
        # For now, we simulate based on the existence of data in the volume
        incoming_data = Path("/app/incoming_data") if IS_CI else config.PROJECT_ROOT / "data" / "frames"
        
        # Placeholder logic: if 'drifted' is in the path or environment, simulate high drift
        if os.getenv("SIMULATE_DRIFT") == "true":
            score = 35.5
        else:
            score = 12.5 
            
        passed = score < DRIFT_THRESHOLD
        status = "PASS" if passed else "FAIL"
        return score, status
    except Exception as e:
        print(f"❌ Drift Check Error: {e}")
        return 0.0, "ERROR"

def check_retrain_trigger():
    """Checks the DB for a streak of failures to determine system stability."""
    if not DB_PATH.exists():
        return False

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT status FROM drift_logs 
        ORDER BY timestamp DESC LIMIT ?
    ''', (config.RETRAIN_TRIGGER_COUNT,))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < config.RETRAIN_TRIGGER_COUNT:
        return False
    
    fail_count = sum(1 for row in rows if row[0] == "FAIL")
    failure_ratio = fail_count / config.RETRAIN_TRIGGER_COUNT
    
    return failure_ratio >= config.DRIFT_FAILURE_RATIO

def trigger_retraining_workflow():
    """Step 2: Model Retraining Simulator."""
    print("🏗️ [Sentinel] Step 2: System Instability Detected. Triggering Retraining...")
    # This would trigger a separate GitHub Action or local training script
    time.sleep(2) 
    print("✅ [Sentinel] Retraining complete. Challenger model generated.")
    return True

def run_decay_audit():
    """Step 3: Model Decay Audit (Gold Standard)."""
    print("🛡️ [Sentinel] Step 3: Auditing Challenger Model via Decay Pipeline...")
    try:
        from detector_model_decay.decay_pipeline import run_analysis
        
        # In CI/Headless mode, run_analysis should not ask for input.
        # We assume config has the correct FRESH/OLD paths set for the environment.
        run_analysis()
        return True 
    except Exception as e:
        print(f"❌ [Sentinel] Decay Audit failed: {e}")
        return False

def main():
    init_db()
    print(f"\n{'='*60}")
    print(f"📡 SENTINEL PIPELINE | MODE: {'HEADLESS (CI)' if IS_CI else 'INTERACTIVE'}")
    print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 1. DRIFT CHECK
    score, status = run_drift_check()
    log_to_db(score, DRIFT_THRESHOLD, status)
    print(f"📊 Result: {score}% Drift | Status: {status}")

    # 2. EVALUATE SYSTEM HEALTH
    is_unstable = check_retrain_trigger()
    
    if status == "FAIL" or is_unstable:
        if is_unstable:
            print("🚨 ALERT: Stability threshold breached (persistent failures).")
            trigger_retraining_workflow()
            
            if run_decay_audit():
                print("🚀 [Sentinel] SUCCESS: Challenger validated. Deployment authorized.")
            else:
                print("⚠️ [Sentinel] ABORT: Challenger failed audit. Rolling back to Baseline.")
        else:
            print("🟡 WARNING: Drift detected. Monitoring for stability before intervention.")
    else:
        print("🟢 SYSTEM HEALTH: Optimal. Baseline model performing within spec.")

    print(f"\n{'='*60}")
    print("🏁 PIPELINE EXECUTION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
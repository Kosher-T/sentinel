import sqlite3
import time
import os
import sys
from pathlib import Path

# --- DYNAMIC PATH RESOLUTION ---
# Ensure project root is in sys.path so we can import internal packages
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import local project modules
try:
    import detector_model_decay.all_config as config
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(str(PROJECT_ROOT / "detector_model_decay"))
    import all_config as config

# --- ENVIRONMENT & CI DETECTION ---
IS_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("DOCKER_ENV") == "true"

# Allow environment variables to override config for workflow flexibility
# Workflow typically passes these via 'docker run -e'
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", config.DECAY_THRESHOLD))

# --- DB & LOGGING PATHS ---
# If in Docker, we use the volume-mapped path for persistence
if IS_CI:
    DB_DIR = Path("/app/status_output")
else:
    DB_DIR = PROJECT_ROOT / "temp_status"

DB_PATH = DB_DIR / "drift_history.db"

def init_db():
    """Ensures the database folder and table exist before running."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
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
    
    # Also write a simple text file for the GitHub Workflow 'cat' command
    with open(DB_DIR / "status.txt", "w") as f:
        f.write(status)
    with open(DB_DIR / "score.txt", "w") as f:
        f.write(f"{score:.2f}")

def run_drift_check():
    """
    Step 1: Data Drift Analysis
    Integrates with detector_data_drift logic.
    """
    print(f"🔍 [Sentinel] Step 1: Checking Data Drift (Threshold: {DRIFT_THRESHOLD}%)...")
    try:
        # Placeholder logic: if 'SIMULATE_DRIFT' is set, we simulate a breach
        if os.getenv("SIMULATE_DRIFT") == "true":
            score = 35.5
        else:
            # Baseline simulated score
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
    # Pull the last 5 runs (configurable via config)
    count = getattr(config, 'RETRAIN_TRIGGER_COUNT', 5)
    ratio = getattr(config, 'DRIFT_FAILURE_RATIO', 0.6)
    
    cursor.execute('''
        SELECT status FROM drift_logs 
        ORDER BY timestamp DESC LIMIT ?
    ''', (count,))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < count:
        return False
    
    fail_count = sum(1 for row in rows if row[0] == "FAIL")
    failure_ratio = fail_count / count
    
    return failure_ratio >= ratio

def trigger_retraining_workflow():
    """Step 2: Model Retraining Simulator."""
    print("🏗️ [Sentinel] Step 2: System Instability Detected. Triggering Retraining...")
    # In a real environment, this would call a training script or API
    time.sleep(1) 
    print("✅ [Sentinel] Retraining complete. Challenger model generated.")
    return True

def run_decay_audit():
    """Step 3: Model Decay Audit (Gold Standard)."""
    print("🛡️ [Sentinel] Step 3: Auditing Challenger Model via Decay Pipeline...")
    try:
        # Import the latest version of the analysis runner
        from detector_model_decay.decay_pipeline import run_analysis
        
        # This will run the comparison between Baseline and Challenger
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
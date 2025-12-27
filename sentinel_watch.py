import sys
import os
import sqlite3
import shutil
import time
from datetime import datetime
from pathlib import Path
import logging
import all_config as config

project_root = config.PROJECT_ROOT
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Internal Imports
import detector_data_drift.drift_pipeline as drift_pipeline
import detector_model_decay.decay_analyzer as decay_analyzer
import detector_data_drift.feature_extractor as detector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL: %(message)s')

class SentinelWatch:
    def __init__(self):
        self.db_path = config.DRIFT_HISTORY_DB
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database if it doesn't exist."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Added data_path column to track the folder in history
        c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, data_path TEXT)''')
        
        # Migration: Check if data_path exists, if not, add it (for existing DBs)
        c.execute("PRAGMA table_info(drift_logs)")
        columns = [column[1] for column in c.fetchall()]
        if 'data_path' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN data_path TEXT")
            
        conn.commit()
        conn.close()

    # --- SIMULATED UTILITIES ---
    
    def simulate_cloud_connection(self):
        logging.info("☁️  Connecting to Cloud Environment...")
        time.sleep(1) # Simulating latency
        return True

    def simulate_alert(self, level, message):
        """
        Levels: INFO, WARNING, CRITICAL
        """
        print(f"\n🚨 [ALERT - {level}] {message}\n")

    def simulate_retraining(self, original_data, new_data_folder):
        logging.info("🛠️  Starting Retraining Loop (Challenger Model)...")
        logging.info(f"   -> Mixing {original_data} + {new_data_folder}")
        time.sleep(3) 
        new_model_path = config.DATA_PATH / "challenger_model_v2.keras"
        return new_model_path

    def simulate_deployment(self, new_model_path):
        logging.info(f"🚀 Deploying Challenger Model ({new_model_path.name}) to Production...")
        time.sleep(2)
        logging.info("✅ Deployment Complete. New model is live.")

    def archive_incoming_data(self, score, status):
        """
        Moves processed incoming data to history with a recognizable name.
        Naming Convention: YYYYMMDD_HHMMSS_[STATUS]_[SCORE]%
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Human-recognizable folder name showing drift status and score
        folder_name = f"{timestamp}_{status}_{score:.2f}pct"
        dest_dir = config.ARCHIVED_DATA_PATH / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🗄️  Archiving incoming data to {dest_dir}...")
        
        try:
            if config.INCOMING_DATA_PATH.exists():
                shutil.copytree(config.INCOMING_DATA_PATH, dest_dir, dirs_exist_ok=True)
                logging.info(f"   -> Data archived successfully as {status}.")
                return str(dest_dir)
            else:
                logging.warning(f"   -> No data found in {config.INCOMING_DATA_PATH} to archive.")
                return None
        except Exception as e:
            logging.error(f"   -> Archive failed: {e}")
            return None

    # --- CORE PIPELINES ---

    def record_drift_result(self, score, status, folder_path=None):
        """Writes the drift result and the path to its corresponding data to the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO drift_logs (timestamp, drift_score, status, threshold, data_path) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, score, status, config.DRIFT_THRESHOLD, folder_path))
        conn.commit()
        conn.close()
        logging.info(f"📝 Result recorded: {status} ({score:.2f}%) mapped to {folder_path}")

    def check_drift_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT status FROM drift_logs ORDER BY timestamp DESC LIMIT ?", (config.TIMEFRAME_WINDOW,))
        rows = c.fetchall()
        conn.close()

        if not rows:
            return False, 0, 0

        statuses = [r[0] for r in rows]
        total = len(statuses)
        fails = statuses.count("FAIL")
        
        is_triggered = ((fails == total) and (total == config.TIMEFRAME_WINDOW)) or ((fails / total) >= config.DRIFT_FAILURE_RATIO)
        return is_triggered, fails, total

    def run_decay_pipeline(self, challenger_model_path):
        logging.info("📉 Running Decay Pipeline (Gatekeeper Check)...")
        try:
            golden_files = detector.get_recursive_image_paths(config.GOLDEN_SET_DIR)
            if not golden_files:
                logging.error("❌ Golden Set empty! Cannot verify decay.")
                return False

            logging.info(f"   -> Testing on {len(golden_files)} Golden Set images.")
            
            import random
            simulated_decay_score = random.uniform(0, 10.0) 
            logging.info(f"   -> Calculated Decay Score on Golden Set: {simulated_decay_score:.2f}%")
            
            if simulated_decay_score > config.DECAY_THRESHOLD:
                logging.error(f"⛔ DECAY CHECK FAILED. Score {simulated_decay_score:.2f}% > Threshold {config.DECAY_THRESHOLD}%")
                return False
            else:
                logging.info(f"✅ Decay Check Passed. Score {simulated_decay_score:.2f}% < Threshold {config.DECAY_THRESHOLD}%")
                return True

        except Exception as e:
            logging.error(f"❌ Decay Pipeline Error: {e}")
            return False

    # --- MAIN WORKFLOW ---

    def watch(self):
        self.simulate_cloud_connection()
        
        logging.info("--- STEP 1: MONITOR DATA DRIFT ---")
        drift_score, status = drift_pipeline.run_drift_check()
        
        if drift_score is None:
            logging.error("Drift Pipeline returned no result.")
            return

        # 1. Archive first to get the folder path with status-based naming
        archived_path = self.archive_incoming_data(drift_score, status)

        # 2. Record Result with the mapping to the archive folder
        self.record_drift_result(drift_score, status, archived_path)

        # 3. Check Logic
        is_triggered, fails, total = self.check_drift_history()
        
        if status == "PASS":
            if not is_triggered:
                logging.info("✅ Drift Status: OK. Archiving and sleeping.")
                return
            else:
                logging.warning(f"⚠️ Current result PASS, but history shows instability ({fails}/{total} fails).")
                return

        logging.warning(f"⚠️ Drift Detected. Historical Window: {fails}/{total} failures.")

        if is_triggered:
            logging.info("--- STEP 2: TRIGGER RETRAINING ---")
            self.simulate_alert("WARNING", f"Drift threshold exceeded ({fails}/{total} in window). Initiating Retraining.")
            
            challenger_model = self.simulate_retraining(config.ORIGINAL_DATA_PATH, config.INCOMING_DATA_PATH)
            
            logging.info("--- STEP 3: DECAY CHECK (GATEKEEPER) ---")
            decay_passed = self.run_decay_pipeline(challenger_model)
            
            if decay_passed:
                self.simulate_deployment(challenger_model)
                self.simulate_alert("INFO", "Self-healing complete. New model deployed.")
            else:
                logging.critical("STOP! Retrained model failed Decay Check.")
                self.simulate_alert("CRITICAL", "Retrained model failed Decay Check. Deployment Aborted.")
                
        else:
            logging.info("ℹ️  Drift detected but threshold not yet met. Recorded and waiting.")

if __name__ == "__main__":
    sentinel = SentinelWatch()
    sentinel.watch()
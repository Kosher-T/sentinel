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
import detector_data_drift.pipeline as pipeline
import detector_model_decay.analyzer as analyzer
import detector_data_drift.extractor as detector

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
        c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, data_path TEXT)''')
        
        # Migration check for data_path
        c.execute("PRAGMA table_info(drift_logs)")
        columns = [column[1] for column in c.fetchall()]
        if 'data_path' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN data_path TEXT")
            
        conn.commit()
        conn.close()

    # --- SIMULATED UTILITIES ---
    
    def simulate_cloud_connection(self):
        logging.info("Connecting to Cloud Environment...")
        time.sleep(1) 
        return True

    def simulate_alert(self, level, message):
        print(f"\n🚨 [ALERT - {level}] {message}\n")

    def simulate_retraining(self, original_data, new_data_folder):
        """Simulates retraining and saving a new model file."""
        logging.info("Starting Retraining Loop (Challenger Model)...")
        logging.info(f"   -> Mixing {original_data} + {new_data_folder}")
        time.sleep(3) 
        # In a real scenario, this would be saved into the CHALLENGER folder
        new_model_name = f"challenger_v{int(time.time())}.keras"
        new_model_path = config.MODEL_PATH / "golden_set_septuplets" / "models" / "challenger" / new_model_name
        
        # Simulate creating the file so Distiller sees it
        new_model_path.touch()
        logging.info(f"💾 Challenger model saved to: {new_model_path.name}")
        return new_model_path

    def simulate_deployment(self, new_model_path):
        logging.info(f"Deploying Challenger Model ({new_model_path.name}) to Production...")
        time.sleep(2)
        logging.info("🟢 Deployment Complete. New model is live.")

    def archive_incoming_data(self, score, status):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{status}_{score:.2f}pct"
        dest_dir = config.ARCHIVED_DATA_PATH / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🗄️ Archiving incoming data to {dest_dir}...")
        
        try:
            if config.INCOMING_DATA_PATH.exists():
                shutil.copytree(config.INCOMING_DATA_PATH, dest_dir, dirs_exist_ok=True)
                return str(dest_dir)
            else:
                logging.warning(f"   -> No data found in {config.INCOMING_DATA_PATH} to archive.")
                return None
        except Exception as e:
            logging.error(f"   -> Archive failed: {e}")
            return None

    # --- CORE PIPELINES ---

    def wait_for_distillation(self, original_model_path):
        """
        Polls the distilled directory to see if the Distiller service 
        has finished creating the latent-space version.
        """
        distilled_name = original_model_path.stem + config.DISTILL_SUFFIX + original_model_path.suffix
        # Determine if it's production or challenger based on path
        if "production" in str(original_model_path):
            target_dir = config.PRODUCTION_DISTILLED_DIR
        else:
            target_dir = config.CHALLENGER_DISTILLED_DIR
            
        distilled_path = target_dir / distilled_name
        
        logging.info(f"⏳ Waiting for Distiller to process {original_model_path.name}...")
        
        max_attempts = 20 # 100 seconds total
        for _ in range(max_attempts):
            if distilled_path.exists():
                logging.info(f"✨ Distilled asset found: {distilled_name}")
                return distilled_path
            time.sleep(5)
            
        logging.error(f"🔴 Timeout: Distiller did not produce {distilled_name} in time.")
        return None

    def record_drift_result(self, score, status, folder_path=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO drift_logs (timestamp, drift_score, status, threshold, data_path) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, score, status, config.DRIFT_THRESHOLD, folder_path))
        conn.commit()
        conn.close()

    def check_drift_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT status FROM drift_logs ORDER BY timestamp DESC LIMIT ?", (config.TIMEFRAME_WINDOW,))
        rows = c.fetchall()
        conn.close()
        if not rows: return False, 0, 0
        statuses = [r[0] for r in rows]
        total = len(statuses)
        fails = statuses.count("FAIL")
        is_triggered = ((fails == total) and (total == config.TIMEFRAME_WINDOW)) or ((fails / total) >= config.DRIFT_FAILURE_RATIO)
        return is_triggered, fails, total

    def run_decay_pipeline(self, challenger_model_path):
        logging.info("Running Decay Pipeline (Gatekeeper Check)...")
        
        # PHASE TWO: Instead of the full model, we need the DISTILLED version for analysis
        distilled_path = self.wait_for_distillation(challenger_model_path)
        if not distilled_path:
            return False

        try:
            golden_files = detector.get_recursive_image_paths(config.GOLDEN_SET_DIR)
            if not golden_files:
                logging.error("🔴 Golden Set empty! Cannot verify decay.")
                return False

            logging.info(f"   -> Testing on {len(golden_files)} Golden Set images using {distilled_path.name}")
            
            # Simulation for now, but in reality, analyzer.py would load distilled_path
            import random
            simulated_decay_score = random.uniform(0, 10.0) 
            
            if simulated_decay_score > config.DECAY_THRESHOLD:
                logging.error(f"🔴 DECAY CHECK FAILED. Score {simulated_decay_score:.2f}% > Threshold {config.DECAY_THRESHOLD}%")
                return False
            else:
                logging.info(f"🟢 Decay Check Passed. Score {simulated_decay_score:.2f}% < Threshold {config.DECAY_THRESHOLD}%")
                return True

        except Exception as e:
            logging.error(f"🔴 Decay Pipeline Error: {e}")
            return False

    # --- MAIN WORKFLOW ---

    def watch(self):
        self.simulate_cloud_connection()
        
        logging.info("--- STEP 1: MONITOR DATA DRIFT ---")
        drift_score, status = pipeline.run_drift_check()
        
        if drift_score is None: return

        archived_path = self.archive_incoming_data(drift_score, status)
        self.record_drift_result(drift_score, status, archived_path)
        is_triggered, fails, total = self.check_drift_history()
        
        if status == "PASS":
            if not is_triggered:
                logging.info("🟢 Drift Status: OK. Archiving and sleeping.")
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
            logging.info("ℹ️ Drift detected but threshold not yet met. Recorded and waiting.")

if __name__ == "__main__":
    sentinel = SentinelWatch()
    sentinel.watch()
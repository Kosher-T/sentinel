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
        c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL)''')
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
        # In prod: Send Slack/Email/PagerDuty notification

    def simulate_retraining(self, original_data, new_data_folder):
        """
        Simulates the retraining of the 'Challenger' model.
        Returns path to the new model artifact.
        """
        logging.info("🛠️  Starting Retraining Loop (Challenger Model)...")
        logging.info(f"   -> Mixing {original_data} + {new_data_folder}")
        time.sleep(3) # Simulate training time
        
        # Simulate producing a new model artifact
        new_model_path = config.DATA_PATH / "challenger_model_v2.keras"
        # In reality, this would actually run a training script
        return new_model_path

    def simulate_deployment(self, new_model_path):
        logging.info(f"🚀 Deploying Challenger Model ({new_model_path.name}) to Production...")
        time.sleep(2)
        logging.info("✅ Deployment Complete. New model is live.")

    def archive_incoming_data(self):
        """Moves processed incoming data to history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = config.ARCHIVED_DATA_PATH / timestamp
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🗄️  Archiving incoming data to {dest_dir}...")
        
        # Activated copying logic to ensure history is populated
        try:
            if config.INCOMING_DATA_PATH.exists():
                shutil.copytree(config.INCOMING_DATA_PATH, dest_dir, dirs_exist_ok=True)
                logging.info("   -> Archive complete.")
            else:
                logging.warning(f"   -> No data found in {config.INCOMING_DATA_PATH} to archive.")
        except Exception as e:
            logging.error(f"   -> Archive failed: {e}")

    # --- CORE PIPELINES ---

    def record_drift_result(self, score, status):
        """Writes the drift result to the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO drift_logs (timestamp, drift_score, status, threshold) VALUES (?, ?, ?, ?)",
                  (timestamp, score, status, config.DRIFT_THRESHOLD))
        conn.commit()
        conn.close()
        logging.info(f"📝 Result recorded: {status} ({score:.2f}%)")

    def check_drift_history(self):
        """
        Checks the failure rate in the configured window.
        Returns: (is_triggered, fail_count, total_count)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Get the last N records
        c.execute("SELECT status FROM drift_logs ORDER BY timestamp DESC LIMIT ?", (config.TIMEFRAME_WINDOW,))
        rows = c.fetchall()
        conn.close()

        if not rows:
            return False, 0, 0

        statuses = [r[0] for r in rows]
        total = len(statuses)
        fails = statuses.count("FAIL")
        
        # Check conditions
        consecutive_fail = (fails == total) and (total == config.TIMEFRAME_WINDOW)
        failure_ratio = (fails / total) >= config.DRIFT_FAILURE_RATIO
        
        is_triggered = consecutive_fail or failure_ratio
        return is_triggered, fails, total

    def run_decay_pipeline(self, challenger_model_path):
        """
        The Decay Check (Gatekeeper).
        Runs the Challenger model against the Golden Set and compares with the Old Model.
        """
        logging.info("📉 Running Decay Pipeline (Gatekeeper Check)...")
        
        try:
            # 1. Load Models (Simulated for Sentinel if files don't exist, else use detector)
            # In a real run, we would load the actual Keras models here.
            # model_old = detector.load_model(config.OLD_MODEL_PATH)
            # model_new = detector.load_model(challenger_model_path)
            
            # 2. Get Golden Set Images
            golden_files = detector.get_recursive_image_paths(config.GOLDEN_SET_DIR)
            if not golden_files:
                logging.error("❌ Golden Set empty! Cannot verify decay.")
                return False

            logging.info(f"   -> Testing on {len(golden_files)} Golden Set images.")

            # 3. Feature Extraction (Simulated here to avoid heavy load in this script, 
            #    or we call detector.extract_features if we want real execution)
            
            # --- SIMULATION OF DECAY CHECK FOR SENTINEL LOGIC ---
            # Ideally, we extract embeddings:
            # old_emb = detector.extract_features_from_list(model_old, golden_files)
            # new_emb = detector.extract_features_from_list(model_new, golden_files)
            
            # For this script's purpose, we will simulate the result based on randomness 
            # or assume a "successful" retrain usually passes, but sometimes fails.
            import random
            simulated_decay_score = random.uniform(0, 10.0) # 0 to 10% decay
            
            # Using decay_analyzer logic (just for the score calculation demonstration)
            # score = decay_analyzer.calculate_decay_score(new_emb, old_emb)
            
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
        
        # 1. RUN Drift Pipeline
        logging.info("--- STEP 1: MONITOR DATA DRIFT ---")
        drift_score, status = drift_pipeline.run_drift_check()
        
        if drift_score is None:
            logging.error("Drift Pipeline returned no result.")
            return

        # 2. Record Result
        self.record_drift_result(drift_score, status)

        # 3. Archive incoming data immediately after check
        self.archive_incoming_data()

        # 4. Check Logic
        is_triggered, fails, total = self.check_drift_history()
        
        if status == "PASS":
            if not is_triggered:
                logging.info("✅ Drift Status: OK. Sleeping.")
                return
            else:
                logging.warning(f"⚠️ Current result PASS, but history shows instability ({fails}/{total} fails). Proceeding with caution (or could trigger check).")
                # For now, we only trigger on confirmed failure trend.
                return

        # Status is FAIL
        logging.warning(f"⚠️ Drift Detected. Historical Window: {fails}/{total} failures.")

        if is_triggered:
            # 5. Trigger Retraining Loop
            logging.info("--- STEP 2: TRIGGER RETRAINING ---")
            self.simulate_alert("WARNING", f"Drift threshold exceeded ({fails}/{total} in window). Initiating Retraining.")
            
            challenger_model = self.simulate_retraining(config.ORIGINAL_DATA_PATH, config.INCOMING_DATA_PATH)
            
            # 6. Run Decay Check
            logging.info("--- STEP 3: DECAY CHECK (GATEKEEPER) ---")
            decay_passed = self.run_decay_pipeline(challenger_model)
            
            if decay_passed:
                self.simulate_deployment(challenger_model)
                self.simulate_alert("INFO", "Self-healing complete. New model deployed.")
            else:
                logging.critical("STOP! Retrained model failed Decay Check.")
                self.simulate_alert("CRITICAL", "Retrained model failed Decay Check. Deployment Aborted. Engineer intervention required.")
                # Option i: Create copy, route data (Simulated log)
                logging.info("-> [Fallback] Keeping old model for Standard Data. Routing New Data to secondary pipeline.")
                
        else:
            logging.info("ℹ️  Drift detected but threshold (consecutive/ratio) not yet met. Recording and waiting.")

if __name__ == "__main__":
    sentinel = SentinelWatch()
    sentinel.watch()
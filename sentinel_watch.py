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

# Service Imports
try:
    from services.execution_engine import ExecutionEngine
    from services.alert_utils import SentinelAlert
    from services.golden_set_curator import GoldenSetCurator
    from services.data_rotator import DataRotator
except ImportError:
    # Fallback if running from a different context
    from execution_engine import ExecutionEngine  # type: ignore
    from alert_utils import SentinelAlert  # type: ignore
    from golden_set_curator import GoldenSetCurator  # type: ignore
    from data_rotator import DataRotator  # type: ignore

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL: %(message)s')

class SentinelWatch:
    def __init__(self):
        self.db_path = config.DRIFT_HISTORY_DB
        self._init_db()
        self.alert_engine = SentinelAlert()

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

    # --- UTILITIES ---
    
    def simulate_cloud_connection(self):
        logging.info("Connecting to Cloud Environment...")
        time.sleep(1) 
        return True

    def send_alert(self, level, message, event_type="generic", metrics=None):
        """
        Triggers the multi-channel alert system (Email + System Notification).
        Maps textual levels to integer priorities for alert_utils.
        """
        level_map = {"INFO": 1, "WARNING": 2, "CRITICAL": 3}
        int_level = level_map.get(level, 1)
        
        # Log to console for persistent record
        metrics_str = f" | Metrics: {metrics}" if metrics else ""
        print(f"\n🚨 [ALERT - {level}] {message}{metrics_str}\n")
        
        # Fire actual alert
        try:
            self.alert_engine.fire(int_level, event_type, message, metrics)
        except Exception as e:
            logging.error(f"Failed to dispatch alert: {e}")

    def simulate_deployment(self, new_model_path):
        logging.info(f"Deploying Challenger Model ({new_model_path.name}) to Production...")
        # In a real scenario, this would move files:
        # shutil.move(new_model_path, config.OLD_MODEL_PATH / "production_vX.pth")
        time.sleep(2)
        logging.info("🟢 Deployment Complete. New model is live.")

    def archive_incoming_data(self, score, status):
        """
        Moves the current batch of incoming data to the archive history.
        Clears the incoming folder so data isn't processed twice.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{status}_{score:.2f}pct"
        dest_dir = config.ARCHIVED_DATA_PATH / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🗄️ Archiving incoming data to {dest_dir}...")
        
        try:
            # Check if there is data to move
            if any(config.INCOMING_DATA_PATH.iterdir()):
                # Move contents, not the folder itself, to preserve the mount point/folder structure
                for item in config.INCOMING_DATA_PATH.iterdir():
                    shutil.move(str(item), str(dest_dir))
                return str(dest_dir)
            else:
                logging.warning(f"   -> No data found in {config.INCOMING_DATA_PATH} to archive.")
                return None
        except Exception as e:
            logging.error(f"   -> Archive failed: {e}")
            return None

    def purge_drift_history(self):
        """
        Cleans up the archived data after a successful retraining and deployment.
        This prevents retraining on data that has already been accounted for.
        """
        logging.info("🧹 Purging drift data history...")
        try:
            if config.ARCHIVED_DATA_PATH.exists():
                shutil.rmtree(config.ARCHIVED_DATA_PATH)
                config.ARCHIVED_DATA_PATH.mkdir(parents=True, exist_ok=True)
                logging.info("🟢 History purged. Ready for new cycle.")
        except Exception as e:
            logging.error(f"🔴 Failed to purge history: {e}")

    def update_baselines(self, deployed_model_path: Path):
        """
        Post-deployment baseline updates.
        
        After a successful model deployment, updates:
        1. Golden Set: Run curator with drifted data to update baselines
        2. ORIGINAL_DATA_PATH: Rotate drifted data into reference dataset
        
        This ensures:
        - Golden Set reflects the new model's outputs
        - Future drift checks won't re-trigger on already-processed data
        
        Args:
            deployed_model_path: Path to the newly deployed model
        """
        logging.info("--- STEP 4: UPDATE BASELINES ---")
        
        # 1. Update Golden Set with the new model's predictions
        try:
            logging.info("📦 Updating Golden Set with new model baselines...")
            curator = GoldenSetCurator(
                input_dirs=[config.ARCHIVED_DATA_PATH],
                model_path=deployed_model_path,
                sample_size=100  # Maintain ~100 sample Golden Set
            )
            exit_code = curator.curate()
            if exit_code == 0:
                logging.info("✅ Golden Set updated successfully")
            else:
                logging.warning("⚠️ Golden Set update had issues, check logs")
        except Exception as e:
            logging.error(f"🔴 Golden Set update failed: {e}")
        
        # 2. Rotate drifted data into ORIGINAL_DATA_PATH
        try:
            logging.info("🔄 Rotating drifted data into ORIGINAL_DATA_PATH...")
            rotator = DataRotator(rotation_percentage=0.20)
            success = rotator.rotate(
                source_dir=config.ARCHIVED_DATA_PATH,
                target_dir=config.ORIGINAL_DATA_PATH,
                sample_prefix="0_"
            )
            if success:
                logging.info("✅ ORIGINAL_DATA_PATH updated successfully")
            else:
                logging.warning("⚠️ ORIGINAL_DATA_PATH rotation had issues")
        except Exception as e:
            logging.error(f"🔴 ORIGINAL_DATA_PATH rotation failed: {e}")

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
        logging.info("ℹ️ Running Drift Check on CPU (this may take a moment for large batches)...")
        drift_score, status = pipeline.run_drift_check()
        
        if drift_score is None: return

        # 1. Archive the data first so it is included in the history
        archived_path = self.archive_incoming_data(drift_score, status)
        self.record_drift_result(drift_score, status, archived_path)
        
        # 2. Check triggers
        is_triggered, fails, total = self.check_drift_history()
        
        if status == "PASS":
            if not is_triggered:
                logging.info("🟢 Drift Status: OK. Data archived.")
                return
            else:
                logging.warning(f"⚠️ Current result PASS, but history shows instability ({fails}/{total} fails).")
                return

        logging.warning(f"⚠️ Drift Detected. Historical Window: {fails}/{total} failures.")

        if is_triggered:
            logging.info("--- STEP 2: TRIGGER RETRAINING (EXECUTION ENGINE) ---")
            self.send_alert("WARNING", f"Drift threshold exceeded ({fails}/{total} in window). Initiating Retraining.", event_type="retraining")
            
            # Initialize Engine
            engine = ExecutionEngine()
            
            # We point the engine to ARCHIVED_DATA_PATH to include all recent failures + the current batch
            logging.info(f"🚀 Dispatching Training Job using data from: {config.ARCHIVED_DATA_PATH}")
            success, challenger_model_path, result_payload = engine.run_training(data_path=config.ARCHIVED_DATA_PATH)
            
            if not success:
                logging.critical(f"🔴 Retraining Failed: {result_payload}")
                self.send_alert("CRITICAL", f"Automated Retraining Failed: {result_payload}", event_type="retraining_error")
                return

            # If success, result_payload contains metrics (Loss/Accuracy)
            logging.info(f"✅ Retraining Complete. Metrics: {result_payload}")
            self.send_alert("INFO", "Retraining Success. Proceeding to Validation.", event_type="retraining", metrics=result_payload)
            
            logging.info("--- STEP 3: DECAY CHECK (GATEKEEPER) ---")
            decay_passed = self.run_decay_pipeline(Path(challenger_model_path))  # type: ignore
            
            if decay_passed:
                self.simulate_deployment(Path(challenger_model_path))  # type: ignore
                self.send_alert("INFO", "Self-healing complete. New model deployed.", event_type="deployment")
                
                # Update baselines BEFORE purging history
                self.update_baselines(Path(challenger_model_path))  # type: ignore
                
                # Cleanup History
                self.purge_drift_history()
            else:
                logging.critical("STOP! Retrained model failed Decay Check.")
                self.send_alert("CRITICAL", "Retrained model failed Decay Check. Deployment Aborted.", event_type="decay_fail", metrics=result_payload)
                
        else:
            logging.info("ℹ️ Drift detected but threshold not yet met. Recorded and waiting.")

if __name__ == "__main__":
    sentinel = SentinelWatch()
    sentinel.watch()
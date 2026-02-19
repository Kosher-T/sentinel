import sys
import os
import json
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
    from services.system_state_tracker import SystemStateTracker
    from services.audit_log import SentinelAuditLog
    from services.model_registry import ModelRegistry
except ImportError:
    # Fallback if running from a different context
    from execution_engine import ExecutionEngine  # type: ignore
    from alert_utils import SentinelAlert  # type: ignore
    from golden_set_curator import GoldenSetCurator  # type: ignore
    from data_rotator import DataRotator  # type: ignore
    from system_state_tracker import SystemStateTracker  # type: ignore
    from audit_log import SentinelAuditLog  # type: ignore
    from model_registry import ModelRegistry  # type: ignore

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL: %(message)s')

import schedule

class SentinelWatch:
    def __init__(self):
        self.db_path = config.DRIFT_HISTORY_DB
        self._init_db()
        self.alert_engine = SentinelAlert()
        self.state_tracker = SystemStateTracker()
        self.audit = SentinelAuditLog()
        self.registry = ModelRegistry()
        
        # Start alert escalation watchdog
        self.alert_engine.start_escalation_watchdog()

    def _init_db(self):
        """Initializes the SQLite database if it doesn't exist."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS drift_logs
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, data_path TEXT)''')
        
        # Migration checks
        c.execute("PRAGMA table_info(drift_logs)")
        columns = [column[1] for column in c.fetchall()]
        if 'data_path' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN data_path TEXT")
        if 'root_cause_json' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN root_cause_json TEXT")
        if 'ci_low' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN ci_low REAL")
        if 'ci_high' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN ci_high REAL")
            
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
        
        # Audit trail
        self.audit.log("alert", "fire", {"level": level, "event_type": event_type, "message": message})
        
        # Fire actual alert
        try:
            self.alert_engine.fire(int_level, event_type, message, metrics)
        except Exception as e:
            logging.error(f"Failed to dispatch alert: {e}")

    def simulate_deployment(self, new_model_path):
        logging.info(f"Deploying Challenger Model ({new_model_path.name}) to Production...")
        self.audit.log("deployment", "start", {"model": new_model_path.name})
        # In a real scenario, this would move files:
        # shutil.move(new_model_path, config.OLD_MODEL_PATH / "production_vX.pth")
        time.sleep(2)
        logging.info("🟢 Deployment Complete. New model is live.")
        self.audit.log("deployment", "success", {"model": new_model_path.name})

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
                self.audit.log("data", "archive", {"dest_dir": str(dest_dir)})
                return str(dest_dir)
            else:
                logging.warning(f"   -> No data found in {config.INCOMING_DATA_PATH} to archive.")
                return None
        except Exception as e:
            logging.error(f"   -> Archive failed: {e}")
            self.audit.log("data", "archive", {"error": str(e)}, status="error")
            return None

    def discard_incoming_data(self):
        """
        Removes the current batch of incoming data without archiving.
        Used when drift check passes and data doesn't need to be preserved.
        """
        logging.info("🗑️ Discarding incoming data (PASS - no archive needed)...")
        try:
            if any(config.INCOMING_DATA_PATH.iterdir()):
                for item in config.INCOMING_DATA_PATH.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                logging.info("✅ Incoming data discarded successfully.")
                self.audit.log("data", "discard")
            else:
                logging.info("   -> No data found in incoming folder.")
        except Exception as e:
            logging.error(f"🔴 Failed to discard incoming data: {e}")
            self.audit.log("data", "discard", {"error": str(e)}, status="error")

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
                self.audit.log("data", "purge_history")
        except Exception as e:
            logging.error(f"🔴 Failed to purge history: {e}")
            self.audit.log("data", "purge_history", {"error": str(e)}, status="error")

    def update_baselines(self, deployed_model_path: Path):
        """
        Post-deployment baseline updates.
        
        After a successful model deployment, updates:
        1. Golden Set: Run curator with drifted data to update baselines
        2. TRAINING_DATA_PATH: Rotate drifted data into reference dataset
        
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
                self.audit.log("baseline", "golden_set_update")
            else:
                logging.warning("⚠️ Golden Set update had issues, check logs")
                self.audit.log("baseline", "golden_set_update", {"exit_code": exit_code}, status="failure")
        except Exception as e:
            logging.error(f"🔴 Golden Set update failed: {e}")
            self.audit.log("baseline", "golden_set_update", {"error": str(e)}, status="error")
        
        # 2. Rotate drifted data into TRAINING_DATA_PATH
        try:
            logging.info("🔄 Rotating drifted data into TRAINING_DATA_PATH...")
            rotator = DataRotator(rotation_percentage=0.20)
            success = rotator.rotate(
                source_dir=config.ARCHIVED_DATA_PATH,
                target_dir=config.TRAINING_DATA_PATH,
                sample_prefix="0_"
            )
            if success:
                logging.info("✅ TRAINING_DATA_PATH updated successfully")
                self.audit.log("baseline", "training_data_update")
            else:
                logging.warning("⚠️ TRAINING_DATA_PATH rotation had issues")
                self.audit.log("baseline", "training_data_update", status="failure")
        except Exception as e:
            logging.error(f"🔴 TRAINING_DATA_PATH rotation failed: {e}")
            self.audit.log("baseline", "training_data_update", {"error": str(e)}, status="error")

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

    def record_drift_result(self, score, status, folder_path=None, root_cause=None, confidence_interval=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rc_json = json.dumps(root_cause) if root_cause else None
        ci_low = confidence_interval.get("low") * 100 if confidence_interval else None
        ci_high = confidence_interval.get("high") * 100 if confidence_interval else None
        c.execute("INSERT INTO drift_logs (timestamp, drift_score, status, threshold, data_path, root_cause_json, ci_low, ci_high) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, score, status, config.DRIFT_THRESHOLD, folder_path, rc_json, ci_low, ci_high))
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
        
        # Check for consecutive failures
        consecutive_fails = 0
        for status in statuses:
            if status == "FAIL":
                consecutive_fails += 1
            else:
                break
        
        is_consecutive = consecutive_fails >= config.RETRAIN_TRIGGER_COUNT
        is_ratio = (total > 0) and ((fails / total) >= config.DRIFT_FAILURE_RATIO)
        
        is_triggered = is_consecutive or is_ratio
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
                self.audit.log("decay", "check_fail", {"score": round(simulated_decay_score, 2), "threshold": config.DECAY_THRESHOLD}, status="failure")
                return False
            else:
                logging.info(f"🟢 Decay Check Passed. Score {simulated_decay_score:.2f}% < Threshold {config.DECAY_THRESHOLD}%")
                self.audit.log("decay", "check_pass", {"score": round(simulated_decay_score, 2), "threshold": config.DECAY_THRESHOLD})
                return True

        except Exception as e:
            logging.error(f"🔴 Decay Pipeline Error: {e}")
            return False

    def check_retraining_trigger(self, fails, total, root_cause):
        """Checks if retraining should be triggered (Drift or Schedule)."""
        # 1. Check Drift Trigger
        is_consecutive = fails >= config.RETRAIN_TRIGGER_COUNT
        is_ratio = (total > 0) and ((fails / total) >= config.DRIFT_FAILURE_RATIO)
        is_drift_triggered = is_consecutive or is_ratio

        if is_drift_triggered:
            return True, "Drift Threshold Exceeded"

        return False, None

    def execute_retraining(self, trigger_reason, root_cause=None):
        """Executes the retraining pipeline."""
        logging.info(f"--- TRIGGERING RETRAINING: {trigger_reason} ---")
        
        alert_msg = f"Retraining Initiated. Reason: {trigger_reason}"
        if root_cause:
             pattern = root_cause.get('drift_pattern', 'unknown')
             alert_msg += f" | Drift Pattern: {pattern}"
        self.send_alert("WARNING", alert_msg, event_type="retraining_start")

        # Initialize Engine
        engine = ExecutionEngine()
        
        # Use ARCHIVED data (recent failures) + TRAINING data (if needed)
        # For scheduled retraining, we might want to use the entire dataset 
        # For now, we stick to the existing data path logic or default
        data_path = config.ARCHIVED_DATA_PATH
        if trigger_reason == "Scheduled":
             # If scheduled, we might want to train on everything or just re-verify
             logging.info("🗓️ Scheduled run: Using accumulated history + training data")
             
        logging.info(f"🚀 Dispatching Training Job using data from: {data_path}")
        self.audit.log("training", "start", {"data_path": str(data_path), "trigger": trigger_reason})
        
        success, challenger_model_path, result_payload = engine.run_training(data_path=data_path)
        
        if not success:
            logging.critical(f"🔴 Retraining Failed: {result_payload}")
            self.audit.log("training", "failure", {"error": str(result_payload)}, status="failure")
            self.send_alert("CRITICAL", f"Automated Retraining Failed: {result_payload}", event_type="retraining_error")
            self.state_tracker.update_from_event("retraining", success=False, details=str(result_payload))
            return

        # If success, result_payload contains metrics (Loss/Accuracy)
        logging.info(f"✅ Retraining Complete. Metrics: {result_payload}")
        self.audit.log("training", "success", {"metrics": result_payload})
        self.send_alert("INFO", "Retraining Success. Proceeding to Validation.", event_type="retraining", metrics=result_payload)
        self.state_tracker.update_from_event("retraining", success=True)
        
        # --- MODEL REGISTRY: Register new version ---
        current_prod = self.registry.get_current_production()
        parent_version = current_prod["version"] if current_prod else None
        model_version = self.registry.register_model(
            model_path=str(challenger_model_path),
            source="retrain",
            trigger_reason=trigger_reason,
            training_metrics=result_payload if isinstance(result_payload, dict) else None,
            parent_version=parent_version,
        )
        
        logging.info("--- STEP 3: DECAY CHECK (GATEKEEPER) ---")
        decay_passed = self.run_decay_pipeline(Path(challenger_model_path))  # type: ignore
        
        # --- MODEL REGISTRY: Record validation result ---
        self.registry.update_validation(model_version, passed=decay_passed)
        
        if decay_passed:
            self.simulate_deployment(Path(challenger_model_path))  # type: ignore
            self.send_alert("INFO", "Self-healing complete. New model deployed.", event_type="deployment")
            self.state_tracker.update_from_event("deployment", success=True)
            
            # --- MODEL REGISTRY: Record deployment ---
            self.registry.record_deployment(model_version)
            
            # Update baselines BEFORE purging history
            self.update_baselines(Path(challenger_model_path))  # type: ignore
            
            # Cleanup History
            self.purge_drift_history()
        else:
            logging.critical("STOP! Retrained model failed Decay Check.")
            self.send_alert("CRITICAL", "Retrained model failed Decay Check. Deployment Aborted.", event_type="decay_fail", metrics=result_payload)
            self.state_tracker.update_from_event("decay_check", success=False)


    # --- JOBS ---

    def job_monitor(self):
        """
        Scheduled Job: Runs the drift monitoring pipeline.
        """
        logging.info("\n🔎 --- STARTING MONITORING JOB ---")
        self.simulate_cloud_connection()
        
        logging.info("--- STEP 1: MONITOR DATA DRIFT ---")
        drift_score, status, root_cause, confidence_interval = pipeline.run_drift_check()
        
        if drift_score is None: 
            logging.info("ℹ️ No new data to check.")
            return

        # Audit the drift check result
        drift_details = {"score": round(drift_score, 2), "threshold": config.DRIFT_THRESHOLD}
        if confidence_interval:
            drift_details["ci_low"] = round(confidence_interval.get("low", 0) * 100, 2)
            drift_details["ci_high"] = round(confidence_interval.get("high", 0) * 100, 2)
            drift_details["ci_margin"] = round(confidence_interval.get("margin", 0) * 100, 2)
        if root_cause and status != "PASS":
            drift_details.update(root_cause) # Add root cause details if available

        self.audit.log(
            "drift", f"check_{status.lower()}",
            drift_details,
            status="success" if status == "PASS" else "failure"
        )

        # 1. Handle data based on drift status
        if status == "PASS":
            # PASS: Record to DB only, discard the incoming data
            self.record_drift_result(drift_score, status, None, root_cause, confidence_interval)
            self.discard_incoming_data()
        else:
            # FAIL: Archive the data and record to DB
            archived_path = self.archive_incoming_data(drift_score, status)
            self.record_drift_result(drift_score, status, archived_path, root_cause, confidence_interval)
        
        # 2. Check triggers
        is_triggered, fails, total = self.check_drift_history()
        
        # 3. Update system state
        self.state_tracker.update_from_drift(drift_score, status, is_triggered)
        
        if status == "PASS":
            if not is_triggered:
                logging.info("🟢 Drift Status: OK. Data discarded.")
                # Auto-acknowledge pending escalations when system is stable
                self.alert_engine.acknowledge_all_alerts()
                return
            else:
                logging.warning(f"⚠️ Current result PASS, but history shows instability ({fails}/{total} fails).")

        if is_triggered:
            self.audit.log("drift", "threshold_triggered", {"fails": fails, "total": total})
            self.execute_retraining("Drift Threshold Exceeded", root_cause)
        else:
            logging.info("ℹ️ Drift detected (or history unstable) but threshold not yet met.")

    def job_retrain(self):
        """
        Scheduled Job: Force retraining to keep model fresh.
        """
        logging.info("\n🗓️ --- STARTING SCHEDULED RETRAINING JOB ---")
        self.execute_retraining("Scheduled Retraining")


    def start_service(self):
        """
        Main Service Loop using 'schedule' library.
        """
        logging.info("🛡️ SENTINEL WATCH SERVICE STARTED")
        logging.info(f"   • Monitoring Interval: {config.MONITOR_INTERVAL_MINUTES} minutes")
        logging.info(f"   • Retraining Interval: {config.RETRAINING_INTERVAL_DAYS} days")

        # Schedule Jobs
        schedule.every(config.MONITOR_INTERVAL_MINUTES).minutes.do(self.job_monitor)
        schedule.every(config.RETRAINING_INTERVAL_DAYS).days.do(self.job_retrain)

        # Run once on startup
        self.job_monitor()

        logging.info("⏳ Waiting for next scheduled job...")
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    sentinel = SentinelWatch()
    sentinel.start_service()

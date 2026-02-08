import subprocess
import sys
import time
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import global configs
from all_config import (
    EXECUTION_DRIVERS_PRIORITY,
    RETRAINING_SCRIPT,
    EXECUTION_TIMEOUT,
    EXPECTED_CHALLENGER_PATH,
    INCOMING_DATA_PATH,
    FORCE_SINGLE_GPU,
    DANGEROUS_STRATEGIES,
    CRITICAL_LOG_ERRORS
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL_EXEC: %(message)s')

class BaseDriver(ABC):
    @abstractmethod
    def start(self, script_path, data_path, config):
        pass

    @abstractmethod
    def is_running(self):
        pass

    @abstractmethod
    def get_logs(self):
        pass

    @abstractmethod
    def finalize(self):
        pass

class LocalDriver(BaseDriver):
    def __init__(self):
        self.process = None
        self.artifact_path = None

    def start(self, script_path, data_path, config): # type: ignore
        logging.info("🟢 Initializing Local Execution Environment...")
        
        data_p = Path(data_path)
        if data_p.is_dir():
            logging.info("📂 Data source detected as directory. Engaging recursive ingestion mode.")
            data_arg = ["--data_dir", str(data_p), "--recursive", "True"]
        else:
            data_arg = ["--data_file", str(data_p)]

        try:
            cmd = ["python", str(script_path)] + data_arg
            # Explicitly set encoding to utf-8 to handle emojis and match mock_train output
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8', 
                bufsize=1,
                universal_newlines=True
            )
            self.artifact_path = config.get("expected_output")
            return True
        except Exception as e:
            logging.error(f"🔴 Local Start Failed: {e}")
            return False

    def is_running(self): # type:ignore
        if not self.process: return False
        return self.process.poll() is None

    def get_logs(self): # type:ignore
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line: return line.strip()
        return None

    def finalize(self): # type:ignore
        if not self.process:
            return False, None, "Process never started."
        
        exit_code = self.process.wait()
        if exit_code == 0:
            if Path(self.artifact_path).exists(): # type: ignore
                return True, self.artifact_path, None
            else:
                return False, None, f"Exit Code 0 but artifact missing at {self.artifact_path}"
        
        return False, None, f"Process exited with non-zero code: {exit_code}"

class ExecutionEngine:
    def __init__(self, drivers_priority=EXECUTION_DRIVERS_PRIORITY):
        self.drivers_priority = drivers_priority
        self.active_driver = None
        self.config = {
            "expected_output": str(EXPECTED_CHALLENGER_PATH),
            "timeout": EXECUTION_TIMEOUT
        }

    def _preflight_safety_check(self, script_path):
        """Scans script for forbidden distributed strategies if hardware governor is active."""
        if not FORCE_SINGLE_GPU:
            return True, None

        logging.info(f"🛡️ Safety Governor: Enforcing single-GPU mode (CUDA_VISIBLE_DEVICES=0).")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for strategy in DANGEROUS_STRATEGIES:
                    if strategy in content:
                        logging.warning(f"💡 Strategy '{strategy}' will be forced to single-device stability.")
            return True, None
        except Exception as e:
            return False, f"Safety Scan Failed: {e}"

    def _parse_metrics(self, log_line):
        """Extracts Loss and Accuracy from trainer logs for real-time tracking."""
        # Check for critical errors first
        for err in CRITICAL_LOG_ERRORS:
            if err.lower() in log_line.lower():
                return {"error": f"Critical Trainer Error Detected: {err}"}

        # Regex for Loss=X.XXX, Accuracy=X.XX
        loss_match = re.search(r"Loss=([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", log_line)
        acc_match = re.search(r"Accuracy=([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", log_line)
        
        if loss_match or acc_match:
            metrics = {}
            if loss_match: metrics["loss"] = float(loss_match.group(1))
            if acc_match: metrics["accuracy"] = float(acc_match.group(1))
            return metrics
        return None

    def run_training(self, script_path=RETRAINING_SCRIPT, data_path=INCOMING_DATA_PATH):
        # Step 1: Safety Check
        safe, err = self._preflight_safety_check(script_path)
        if not safe:
            return False, None, err

        # Step 2: Driver Selection & Launch
        for driver_name in self.drivers_priority:
            driver = self._get_driver(driver_name)
            if not driver: continue

            logging.info(f"🔄 Attempting Training via {driver_name}...")
            if driver.start(script_path, data_path, self.config):
                self.active_driver = driver
                return self._monitor_execution()
            
            logging.warning(f"⚠️ {driver_name} failed to provision. Trying next failover...")

        return False, None, "All execution drivers failed."

    def _get_driver(self, name):
        if name == "LOCAL": return LocalDriver()
        # Plugs for future cloud implementations
        if name == "AWS": return None # To be implemented with Boto3
        if name == "GCP": return None # To be implemented with Vertex AI SDK
        return None

    def _monitor_execution(self):
        """Polls the driver until completion."""
        start_time = time.time()
        latest_metrics = {}
        found_error = None
        
        # Keep polling as long as it's running OR as long as there are logs to read
        while True:
            is_running = self.active_driver.is_running() # type: ignore
            log = self.active_driver.get_logs() # type: ignore
            
            if log:
                logging.info(f"[TRAINER]: {log}")
                metrics = self._parse_metrics(log)
                if metrics:
                    latest_metrics = metrics
                    if "error" in metrics:
                        found_error = metrics["error"]
                        logging.error(f"🔴 Fail-Fast Triggered: {found_error}")
                        # If running, try to terminate
                        if is_running and hasattr(self.active_driver, 'process') and self.active_driver.process: # type: ignore
                            self.active_driver.process.terminate() # type: ignore
                        return False, None, found_error
            
            if not is_running and not log:
                break
                
            if time.time() - start_time > self.config["timeout"]:
                if hasattr(self.active_driver, 'process') and self.active_driver.process: # type: ignore
                    self.active_driver.process.terminate() # type: ignore
                return False, None, "Execution timed out."
            
            if not log:
                time.sleep(0.1)

        # Process has finished
        success, path, driver_error = self.active_driver.finalize() # type: ignore
        
        # Priority 1: If we caught an error in the metrics during the loop
        if found_error:
            return False, None, found_error
            
        # Priority 2: If the driver finalization failed, check metrics one last time
        if not success:
            if "error" in latest_metrics:
                return False, None, latest_metrics["error"]
            return False, None, driver_error

        return success, path, latest_metrics

if __name__ == "__main__":
    engine = ExecutionEngine()
    result = engine.run_training()
    print(f"Final Result: {result}")
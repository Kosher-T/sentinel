import subprocess
import time
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

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
        self.last_metrics = {}

    def _pre_flight_check(self, script_path):
        """Scans the training script for dangerous strategies."""
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
                for strategy in DANGEROUS_STRATEGIES:
                    if strategy in content:
                        logging.warning(f"⚠️ DANGER: '{strategy}' detected in training script.")
                        return True, strategy
            return False, None
        except Exception as e:
            logging.error(f"Failed to perform pre-flight scan: {e}")
            return False, None

    def start(self, script_path, data_path, config): # type: ignore
        logging.info("🟢 Initializing Local Execution Environment...")
        
        # 1. Keyword Scan
        has_dangerous, strategy_name = self._pre_flight_check(script_path)
        
        # 2. Hardware Override
        env = os.environ.copy()
        if FORCE_SINGLE_GPU:
            logging.info("🛡️ Safety Governor: Enforcing single-GPU mode (CUDA_VISIBLE_DEVICES=0).")
            env["CUDA_VISIBLE_DEVICES"] = "0"
            if has_dangerous:
                logging.info(f"💡 Strategy '{strategy_name}' will be forced to single-device stability.")

        data_p = Path(data_path)
        if data_p.is_dir():
            data_arg = ["--data_dir", str(data_p), "--recursive", "True"]
            logging.info(f"📂 Data source detected as directory. Engaging recursive ingestion mode.")
        else:
            data_arg = ["--data_file", str(data_p)]
            logging.info(f"📄 Data source detected as single file.")

        try:
            cmd = ["python", str(script_path)] + data_arg
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout for parsing
                text=True,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            self.artifact_path = config.get("expected_output")
            return True
        except Exception as e:
            logging.error(f"🔴 Local Start Failed: {e}")
            return False

    def is_running(self): # type: ignore
        if not self.process: return False
        return self.process.poll() is None

    def get_logs(self): # type: ignore
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line: return line.strip()
        return None

    def finalize(self): # type: ignore
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
        # Regex for capturing metrics from logs
        # Pattern looks for: Metrics: Loss=0.1234, Accuracy=0.85
        self.metric_pattern = re.compile(r"Loss=([+-]?\d*\.\d+|nan|inf).*Accuracy=([+-]?\d*\.\d+|nan|inf)", re.IGNORECASE)

    def run_training(self, script_path=RETRAINING_SCRIPT, data_path=INCOMING_DATA_PATH):
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
        return None

    def _parse_metrics(self, log_line):
        """Extracts metrics and checks for numerical instability."""
        # 1. Look for NaNs/Infs or specific error keywords
        lower_line = log_line.lower()
        for err in CRITICAL_LOG_ERRORS:
            if err.lower() in lower_line:
                return {"error": f"Numerical Instability/Error Detected: {err}"}

        # 2. Extract standard metrics
        match = self.metric_pattern.search(log_line)
        if match:
            loss_val = match.group(1)
            acc_val = match.group(2)
            return {"loss": loss_val, "accuracy": acc_val}
        
        return None

    def _monitor_execution(self):
        start_time = time.time()
        latest_metrics = {}
        
        while self.active_driver.is_running(): # type: ignore
            log = self.active_driver.get_logs() # type: ignore
            if log:
                logging.info(f"[TRAINER]: {log}")
                
                # Metric Parsing & Fail-Fast
                metrics = self._parse_metrics(log)
                if metrics:
                    if "error" in metrics:
                        logging.error(f"🔴 Fail-Fast Triggered: {metrics['error']}")
                        # Terminate the process if possible (Driver dependent)
                        if hasattr(self.active_driver, 'process') and self.active_driver.process: # type: ignore
                            self.active_driver.process.terminate() # type: ignore
                        return False, None, metrics["error"]
                    
                    latest_metrics = metrics
            
            if time.time() - start_time > self.config["timeout"]:
                if hasattr(self.active_driver, 'process') and self.active_driver.process: # type: ignore
                    self.active_driver.process.terminate() # type: ignore
                return False, None, "Execution timed out."
            
            time.sleep(0.1)

        success, path, error = self.active_driver.finalize() # type: ignore
        # Attach the latest metrics to the result for Sentinel's alerts
        return success, path, error if error else latest_metrics

if __name__ == "__main__":
    engine = ExecutionEngine()
    result = engine.run_training()
    print(f"Final Result: {result}")
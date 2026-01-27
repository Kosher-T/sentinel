import subprocess
import time
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL_EXEC: %(message)s')

class BaseDriver(ABC):
    """
    The Universal Contract. Every Cloud/Local driver must implement these methods.
    """
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
        """Should return (success_bool, artifact_path, error_msg)"""
        pass

class LocalDriver(BaseDriver):
    """The 'Home Lab' driver using local terminal subprocesses."""
    def __init__(self):
        self.process = None
        self.stdout = ""
        self.artifact_path = None

    def start(self, script_path, data_path, config):  #type: ignore
        logging.info("🟢 Initializing Local Execution Environment...")
        
        data_p = Path(data_path)
        # Determine if we are dealing with a single file or a directory for recursive ingestion
        if data_p.is_dir():
            data_arg = ["--data_dir", str(data_p), "--recursive", "True"]
            logging.info(f"📂 Data source detected as directory. Engaging recursive ingestion mode.")
        else:
            data_arg = ["--data_file", str(data_p)]
            logging.info(f"📄 Data source detected as single file.")

        try:
            # Construct command with dynamic data arguments
            cmd = ["python", str(script_path)] + data_arg
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.artifact_path = config.get("expected_output", "new_model.pth")
            return True
        except Exception as e:
            logging.error(f"🔴 Local Start Failed: {e}")
            return False

    def is_running(self):  #type: ignore
        if not self.process: return False
        return self.process.poll() is None

    def get_logs(self):  #type: ignore
        # Capturing non-blocking output
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line: return line.strip()
        return None

    def finalize(self):  #type: ignore
        if not self.process:
            return False, None, "Process never started."
        
        exit_code = self.process.wait()
        if exit_code == 0:
            # Check if the model actually exists
            if Path(self.artifact_path).exists():  #type: ignore
                return True, self.artifact_path, None
            else:
                return False, None, "Training finished but no model file found."
        
        _, stderr = self.process.communicate()
        return False, None, f"Exit Code {exit_code}: {stderr}"

class ExecutionEngine:
    """The Traffic Controller / Dispatcher."""
    def __init__(self, drivers_priority=["LOCAL"]):
        self.drivers_priority = drivers_priority
        self.active_driver = None
        self.config = {
            "expected_output": "models/challenger_v2.pth",
            "timeout": 3600 # 1 hour
        }

    def run_training(self, script_path, data_path):
        """
        The main entry point for Sentinel. Handles Failover logic.
        """
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
        
        while self.active_driver.is_running():  #type: ignore
            log = self.active_driver.get_logs()  #type: ignore
            if log:
                logging.info(f"[TRAINER]: {log}")
            
            # Simple timeout check
            if time.time() - start_time > self.config["timeout"]:
                return False, None, "Execution timed out."
            
            time.sleep(1) # Don't burn the CPU polling

        return self.active_driver.finalize()  #type: ignore

# --- MOCK TESTER ---
if __name__ == "__main__":
    # Ensure a local directory 'data/drift_shard' exists for testing directory mode
    # or a file 'data/drifted_data.csv' exists for testing file mode.
    engine = ExecutionEngine(drivers_priority=["LOCAL"])
    success, model, error = engine.run_training("mock_train.py", "C:\\Code\\Code\\Python\\frame_generation_engine\\sentinel\\data\\data_drift\\incoming_data")
    if success:
        print(f"✅ SUCCESS: New model at {model}")
    else:
        print(f"❌ FAILED: {error}")
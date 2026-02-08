# system_state_tracker.py - Sentinel System State Tracker
# Tracks global system state (NOMINAL / WARNING / RED) based on drift checks and pipeline events.
# State is persisted to disk and exposed to the dashboard.

import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] STATE_TRACKER: %(message)s')

try:
    import all_config as config
    STATE_FILE = config.SYSTEM_STATE_FILE
except (ImportError, AttributeError):
    # Fallback if running from a different context or before config update
    STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "data_drift" / "system_state.json"


class SystemState(Enum):
    """Global system state enumeration."""
    NOMINAL = "NOMINAL"  # Green - All systems operating normally
    WARNING = "WARNING"  # Yellow - Drift detected but not critical
    RED = "RED"          # Red - Critical failure (retraining failed, decay check failed)


class SystemStateTracker:
    """
    Tracks and persists global system state.
    
    State transitions:
    - NOMINAL: Drift PASS or successful deployment
    - WARNING: Drift FAIL (not triggered), or retraining in progress
    - RED: Retraining failed or decay check failed
    """
    
    # Maximum number of drift history entries to keep
    MAX_HISTORY_SIZE = 50
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize the state tracker.
        
        Args:
            state_file: Optional path override for state persistence file.
        """
        self.state_file = state_file or STATE_FILE
        self._ensure_parent_dir()
        self._data = self._load_state()
    
    def _ensure_parent_dir(self) -> None:
        """Ensure the parent directory exists."""
        if not self.state_file.parent.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self) -> Dict[str, Any]:
        """Load existing state from disk or return default."""
        default_state = {
            "state": SystemState.NOMINAL.value,
            "last_updated": datetime.now().isoformat(),
            "last_reason": "System initialized",
            "drift_history": []
        }
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Validate state value
                    if data.get("state") not in [s.value for s in SystemState]:
                        data["state"] = SystemState.NOMINAL.value
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load state file: {e}. Using defaults.")
                return default_state
        return default_state
    
    def _persist(self) -> None:
        """Save current state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self._data, f, indent=4)
        except IOError as e:
            logging.error(f"Failed to persist state to {self.state_file}: {e}")
    
    def get_state(self) -> str:
        """
        Get the current system state as a string.
        
        Returns:
            One of: "NOMINAL", "WARNING", "RED"
        """
        return self._data["state"]
    
    def get_state_details(self) -> Dict[str, Any]:
        """
        Get full state details including history.
        
        Returns:
            Dict containing: state, last_updated, last_reason, drift_history
        """
        return self._data.copy()
    
    def _update_state(self, new_state: SystemState, reason: str) -> None:
        """
        Internal method to update state with tracking.
        
        Args:
            new_state: The new SystemState
            reason: Human-readable reason for the change
        """
        old_state = self._data["state"]
        self._data["state"] = new_state.value
        self._data["last_updated"] = datetime.now().isoformat()
        self._data["last_reason"] = reason
        
        if old_state != new_state.value:
            logging.info(f"🔄 State transition: {old_state} → {new_state.value} | Reason: {reason}")
        
        self._persist()
    
    def _add_drift_history(self, score: float, status: str) -> None:
        """Add a drift result to history, maintaining max size."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "status": status
        }
        self._data["drift_history"].insert(0, entry)
        
        # Trim history to max size
        if len(self._data["drift_history"]) > self.MAX_HISTORY_SIZE:
            self._data["drift_history"] = self._data["drift_history"][:self.MAX_HISTORY_SIZE]
    
    def update_from_drift(self, score: float, status: str, is_triggered: bool = False) -> None:
        """
        Update state based on a drift check result.
        
        Args:
            score: The drift score percentage
            status: "PASS" or "FAIL"
            is_triggered: True if drift threshold triggered retraining
        """
        self._add_drift_history(score, status)
        
        if status == "PASS":
            self._update_state(
                SystemState.NOMINAL,
                f"Drift check passed (score: {score:.2f}%)"
            )
        elif is_triggered:
            self._update_state(
                SystemState.WARNING,
                f"Drift triggered retraining (score: {score:.2f}%)"
            )
        else:
            # FAIL but not triggered yet
            self._update_state(
                SystemState.WARNING,
                f"Drift detected (score: {score:.2f}%), monitoring..."
            )
    
    def update_from_event(self, event_type: str, success: bool, details: Optional[str] = None) -> None:
        """
        Update state based on a pipeline event.
        
        Args:
            event_type: One of "retraining", "decay_check", "deployment"
            success: Whether the event succeeded
            details: Optional additional context
        """
        detail_str = f" - {details}" if details else ""
        
        if event_type == "retraining":
            if success:
                self._update_state(
                    SystemState.WARNING,
                    f"Retraining completed, awaiting validation{detail_str}"
                )
            else:
                self._update_state(
                    SystemState.RED,
                    f"Retraining failed{detail_str}"
                )
        
        elif event_type == "decay_check":
            if success:
                self._update_state(
                    SystemState.WARNING,
                    f"Decay check passed, proceeding to deployment{detail_str}"
                )
            else:
                self._update_state(
                    SystemState.RED,
                    f"Decay check failed - deployment aborted{detail_str}"
                )
        
        elif event_type == "deployment":
            if success:
                self._update_state(
                    SystemState.NOMINAL,
                    f"Deployment successful - system healed{detail_str}"
                )
            else:
                self._update_state(
                    SystemState.RED,
                    f"Deployment failed{detail_str}"
                )
        
        else:
            logging.warning(f"Unknown event type: {event_type}")


# Convenience function for reading state without instantiating the class
def get_current_state() -> Dict[str, Any]:
    """
    Read the current system state from disk.
    
    Returns:
        Dict with state details, or default NOMINAL state if file doesn't exist.
    """
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    
    return {
        "state": SystemState.NOMINAL.value,
        "last_updated": None,
        "last_reason": "No state file found"
    }


if __name__ == "__main__":
    print("\n🔍 SYSTEM STATE TRACKER TEST\n")
    print("-" * 40)
    
    # Initialize tracker
    tracker = SystemStateTracker()
    print(f"Current State: {tracker.get_state()}")
    print(f"Details: {tracker.get_state_details()}")
    
    # Simulate drift FAIL
    print("\n📉 Simulating drift FAIL...")
    tracker.update_from_drift(score=15.5, status="FAIL", is_triggered=False)
    print(f"State after drift FAIL: {tracker.get_state()}")
    
    # Simulate triggered retraining
    print("\n🔄 Simulating triggered retraining...")
    tracker.update_from_drift(score=18.2, status="FAIL", is_triggered=True)
    print(f"State after trigger: {tracker.get_state()}")
    
    # Simulate successful retraining
    print("\n✅ Simulating successful retraining...")
    tracker.update_from_event("retraining", success=True)
    print(f"State after retraining: {tracker.get_state()}")
    
    # Simulate decay check pass
    print("\n✅ Simulating decay check pass...")
    tracker.update_from_event("decay_check", success=True)
    print(f"State after decay check: {tracker.get_state()}")
    
    # Simulate deployment success
    print("\n🚀 Simulating deployment success...")
    tracker.update_from_event("deployment", success=True)
    print(f"Final State: {tracker.get_state()}")
    
    print("\n" + "-" * 40)
    print("✅ State Tracker Test Complete")

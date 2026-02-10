# rebase_workflow.py - Sentinel System Rebase Orchestration
# Handles engineer-initiated system rebase after external changes.
#
# Usage:
#   from services.rebase_workflow import RebaseWorkflow
#   workflow = RebaseWorkflow()
#   options = workflow.get_options_for_change("new_model")
#   workflow.start(change_type="new_model", method="new_training_data", config={...})

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] REBASE: %(message)s')

try:
    from services.system_state_tracker import SystemStateTracker
    from services.audit_log import SentinelAuditLog
except ImportError:
    from system_state_tracker import SystemStateTracker
    from audit_log import SentinelAuditLog  # type: ignore

try:
    import all_config as config
except ImportError:
    config = None


class ChangeType(Enum):
    """What changed that requires a rebase."""
    NEW_MODEL = "new_model"              # New model deployed
    DATA_PIPELINE = "data_pipeline"      # Data source/pipeline changed
    TRANSIENT_FIX = "transient_fix"      # Temporary issue fixed, nothing structural


class RebaseMethod(Enum):
    """How to perform the rebase."""
    NEW_TRAINING_DATA = "new_training_data"      # Engineer provides fresh training data
    STREAM_CALIBRATION = "stream_calibration"    # Trust next N incoming samples
    KEEP_BASELINE = "keep_baseline"              # Just clear failure state


# Valid methods for each change type
VALID_METHODS = {
    ChangeType.NEW_MODEL: [
        RebaseMethod.NEW_TRAINING_DATA,
        RebaseMethod.STREAM_CALIBRATION
    ],
    ChangeType.DATA_PIPELINE: [
        RebaseMethod.NEW_TRAINING_DATA,
        RebaseMethod.STREAM_CALIBRATION
    ],
    ChangeType.TRANSIENT_FIX: [
        RebaseMethod.KEEP_BASELINE,
        RebaseMethod.STREAM_CALIBRATION
    ]
}


# Human-readable descriptions for the UI
CHANGE_TYPE_LABELS = {
    ChangeType.NEW_MODEL: {
        "label": "New model deployed",
        "description": "A new or updated model has been deployed to production"
    },
    ChangeType.DATA_PIPELINE: {
        "label": "Data pipeline changed",
        "description": "The data source, preprocessing, or pipeline has changed"
    },
    ChangeType.TRANSIENT_FIX: {
        "label": "Transient issue fixed",
        "description": "A temporary issue was fixed (no structural changes)"
    }
}

REBASE_METHOD_LABELS = {
    RebaseMethod.NEW_TRAINING_DATA: {
        "label": "Provide new training data",
        "description": "Recalibrate Golden Set from engineer-provided data"
    },
    RebaseMethod.STREAM_CALIBRATION: {
        "label": "Trust incoming stream",
        "description": "Use the next N production samples as the new baseline"
    },
    RebaseMethod.KEEP_BASELINE: {
        "label": "Keep current baseline",
        "description": "Just clear failure state without changing the Golden Set"
    }
}


class RebaseWorkflow:
    """
    Orchestrates the system rebase process.
    
    The rebase workflow allows engineers to reset Sentinel after external
    interventions or proactive improvements. The workflow:
    
    1. Captures what changed (reason)
    2. Presents valid options based on the change type
    3. Executes the chosen rebase method
    4. Returns the system to NOMINAL state
    """
    
    # Number of samples to collect in stream calibration mode
    DEFAULT_CALIBRATION_SAMPLES = 50
    
    def __init__(self, state_tracker: Optional[SystemStateTracker] = None):
        """
        Initialize the rebase workflow.
        
        Args:
            state_tracker: Optional SystemStateTracker instance (creates one if not provided)
        """
        self.state_tracker = state_tracker or SystemStateTracker()
        self.audit = SentinelAuditLog()
        self._progress = {
            "phase": None,
            "message": "",
            "percent": 0,
            "details": {}
        }
    
    @staticmethod
    def get_change_types() -> List[Dict[str, Any]]:
        """
        Get all available change types for the UI.
        
        Returns:
            List of dicts with value, label, and description
        """
        return [
            {
                "value": ct.value,
                "label": CHANGE_TYPE_LABELS[ct]["label"],
                "description": CHANGE_TYPE_LABELS[ct]["description"]
            }
            for ct in ChangeType
        ]
    
    @staticmethod
    def get_options_for_change(change_type: str) -> List[Dict[str, Any]]:
        """
        Get valid rebase methods for a given change type.
        
        Args:
            change_type: One of "new_model", "data_pipeline", "transient_fix"
            
        Returns:
            List of valid method dicts with value, label, and description
        """
        try:
            ct = ChangeType(change_type)
        except ValueError:
            logging.warning(f"Unknown change type: {change_type}")
            return []
        
        valid_methods = VALID_METHODS.get(ct, [])
        return [
            {
                "value": method.value,
                "label": REBASE_METHOD_LABELS[method]["label"],
                "description": REBASE_METHOD_LABELS[method]["description"]
            }
            for method in valid_methods
        ]
    
    def start(
        self,
        change_type: str,
        method: str,
        config_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Begin a rebase operation.
        
        Args:
            change_type: What changed (new_model, data_pipeline, transient_fix)
            method: Chosen approach (new_training_data, stream_calibration, keep_baseline)
            config_data: Additional configuration for the chosen method
            
        Returns:
            True if rebase started successfully
        """
        # Validate inputs
        try:
            ct = ChangeType(change_type)
            rm = RebaseMethod(method)
        except ValueError as e:
            logging.error(f"Invalid rebase parameters: {e}")
            return False
        
        # Check if method is valid for change type
        if rm not in VALID_METHODS.get(ct, []):
            logging.error(f"Method '{method}' not valid for change type '{change_type}'")
            return False
        
        # Start the rebase in state tracker
        self.state_tracker.start_rebase(reason=change_type, method=method)
        
        self._progress = {
            "phase": "started",
            "message": "Rebase initiated",
            "percent": 10,
            "details": {
                "change_type": change_type,
                "method": method,
                "config": config_data or {}
            }
        }
        
        logging.info(f"🔄 Starting rebase: {change_type} → {method}")
        self.audit.log("rebase", "start", {"change_type": change_type, "method": method, "config": config_data or {}})
        
        # Execute based on method
        if rm == RebaseMethod.KEEP_BASELINE:
            return self._execute_keep_baseline()
        elif rm == RebaseMethod.NEW_TRAINING_DATA:
            return self._execute_new_training_data(config_data or {})
        elif rm == RebaseMethod.STREAM_CALIBRATION:
            return self._execute_stream_calibration(config_data or {})
        
        return False
    
    def _execute_keep_baseline(self) -> bool:
        """
        Execute 'keep baseline' rebase - just clear failure state.
        
        Returns:
            True on success
        """
        self._progress["phase"] = "executing"
        self._progress["message"] = "Clearing failure state..."
        self._progress["percent"] = 50
        
        # Nothing to do except complete the rebase
        self.state_tracker.complete_rebase(
            success=True,
            details="Baseline preserved, failure state cleared"
        )
        
        self._progress["phase"] = "complete"
        self._progress["message"] = "Rebase complete - baseline preserved"
        self._progress["percent"] = 100
        
        logging.info("✅ Keep baseline rebase complete")
        return True
    
    def _execute_new_training_data(self, config_data: Dict[str, Any]) -> bool:
        """
        Execute rebase with new training data.
        
        The engineer must provide:
        - training_data_path: Path to new training data
        - model_path: Path to the production model (optional, uses config default)
        
        Args:
            config_data: Configuration including training_data_path
            
        Returns:
            True if recalibration started (actual work happens async)
        """
        training_data_path = config_data.get("training_data_path")
        model_path = config_data.get("model_path")
        
        if not training_data_path:
            self._progress["phase"] = "error"
            self._progress["message"] = "Training data path required"
            logging.error("Training data path not provided for new_training_data rebase")
            return False
        
        # Validate path exists
        if not Path(training_data_path).exists():
            self._progress["phase"] = "error"
            self._progress["message"] = f"Training data not found: {training_data_path}"
            logging.error(f"Training data path does not exist: {training_data_path}")
            return False
        
        self._progress["phase"] = "executing"
        self._progress["message"] = "Recalibrating Golden Set..."
        self._progress["percent"] = 30
        
        # TODO: Integrate with setup.py calibration functions
        # For now, mark as requiring manual completion
        self._progress["phase"] = "awaiting_completion"
        self._progress["message"] = "Recalibration in progress - run setup.py to complete"
        self._progress["percent"] = 50
        self._progress["details"]["training_data_path"] = training_data_path
        self._progress["details"]["model_path"] = model_path
        
        logging.info(f"📊 New training data rebase started with: {training_data_path}")
        return True
    
    def _execute_stream_calibration(self, config_data: Dict[str, Any]) -> bool:
        """
        Execute stream calibration rebase.
        
        Sentinel will collect the next N incoming samples and use them
        as the new baseline.
        
        Args:
            config_data: Configuration including sample_count (optional)
            
        Returns:
            True if calibration mode started
        """
        sample_count = config_data.get("sample_count", self.DEFAULT_CALIBRATION_SAMPLES)
        
        # Get calibration sample count from config if available
        if config and hasattr(config, 'REBASE_CALIBRATION_SAMPLES'):
            sample_count = config_data.get("sample_count", config.REBASE_CALIBRATION_SAMPLES)
        
        self._progress["phase"] = "calibrating"
        self._progress["message"] = f"Collecting {sample_count} samples for new baseline..."
        self._progress["percent"] = 20
        self._progress["details"]["target_samples"] = sample_count
        self._progress["details"]["collected_samples"] = 0
        
        # TODO: Implement actual stream collection
        # This requires integration with the drift check pipeline
        # For now, mark as in-progress
        
        logging.info(f"📊 Stream calibration started - collecting {sample_count} samples")
        return True
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current rebase progress.
        
        Returns:
            Dict with phase, message, percent, and details
        """
        return self._progress.copy()
    
    def complete(self, success: bool = True, details: Optional[str] = None) -> None:
        """
        Manually complete the rebase (for methods that require external completion).
        
        Args:
            success: Whether rebase was successful
            details: Optional message
        """
        self.state_tracker.complete_rebase(success=success, details=details)
        self.audit.log("rebase", "complete", {"success": success, "details": details}, status="success" if success else "failure")
        
        self._progress["phase"] = "complete" if success else "failed"
        self._progress["message"] = details or ("Rebase complete" if success else "Rebase failed")
        self._progress["percent"] = 100 if success else 0
    
    def cancel(self) -> None:
        """Cancel the current rebase operation."""
        self.state_tracker.cancel_rebase()
        self.audit.log("rebase", "cancel")
        
        self._progress["phase"] = "cancelled"
        self._progress["message"] = "Rebase cancelled"
        self._progress["percent"] = 0


if __name__ == "__main__":
    print("\n🔄 REBASE WORKFLOW TEST\n")
    print("-" * 40)
    
    # Show available change types
    print("\n📋 Available Change Types:")
    for ct in RebaseWorkflow.get_change_types():
        print(f"  • {ct['label']} ({ct['value']})")
        print(f"    {ct['description']}")
    
    # Show options for new_model
    print("\n📋 Options for 'new_model':")
    for opt in RebaseWorkflow.get_options_for_change("new_model"):
        print(f"  • {opt['label']} ({opt['value']})")
        print(f"    {opt['description']}")
    
    # Show options for transient_fix
    print("\n📋 Options for 'transient_fix':")
    for opt in RebaseWorkflow.get_options_for_change("transient_fix"):
        print(f"  • {opt['label']} ({opt['value']})")
        print(f"    {opt['description']}")
    
    print("\n" + "-" * 40)
    print("✅ Rebase Workflow Module Loaded Successfully")

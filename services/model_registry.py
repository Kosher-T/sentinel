# model_registry.py - Sentinel Model Registry Service
# Tracks model versions, training metrics, validation results, and deployment history.
#
# Usage:
#   from services.model_registry import ModelRegistry
#   registry = ModelRegistry()
#   version = registry.register_model(model_path, source="retrain", ...)
#   registry.update_validation(version, passed=True, decay_metrics={...})
#   registry.record_deployment(version)

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import all_config as config
except ImportError:
    config = None


class ModelStatus:
    """Model lifecycle status enumeration."""
    REGISTERED = "registered"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class ModelRegistry:
    """
    SQLite-backed model version registry.

    Tracks every model through its lifecycle:
      registered → validated/rejected → deployed

    Each version records:
    - Source (retrain, manual, rebase) and trigger reason
    - Training metrics (loss, accuracy, epochs)
    - Decay validation metrics (golden set score, wasserstein)
    - Deployment timestamp and parent version lineage
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path:
            self.db_path = db_path
        elif config and hasattr(config, 'MODEL_REGISTRY_DB'):
            self.db_path = config.MODEL_REGISTRY_DB
        else:
            self.db_path = Path(__file__).resolve().parent.parent / "data" / "data_drift" / "model_registry.db"

        self._init_db()

    def _init_db(self):
        """Creates the model_versions table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            registered_at TEXT NOT NULL,
            model_path TEXT,
            source TEXT,
            trigger_reason TEXT,
            parent_version TEXT,
            status TEXT NOT NULL DEFAULT 'registered',
            training_metrics_json TEXT,
            decay_metrics_json TEXT,
            deployment_timestamp TEXT,
            notes TEXT
        )''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_mv_status ON model_versions(status)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_mv_registered ON model_versions(registered_at)''')
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def register_model(
        self,
        model_path: str,
        source: str = "retrain",
        trigger_reason: str = "",
        training_metrics: dict = None,
        parent_version: str = None,
        notes: str = None,
    ) -> str:
        """
        Register a new model version after training.

        Args:
            model_path: Path to the model artifact
            source: How it was created (retrain, manual, rebase)
            trigger_reason: What triggered the training (Drift, Scheduled, Manual)
            training_metrics: Dict of training metrics (loss, accuracy, epochs, etc.)
            parent_version: The version this model replaces (if any)
            notes: Optional free-text notes

        Returns:
            The assigned version string (e.g. "v3")
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Auto-increment version
        c.execute("SELECT MAX(id) FROM model_versions")
        row = c.fetchone()
        next_num = (row[0] or 0) + 1
        version = f"v{next_num}"

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_json = json.dumps(training_metrics) if training_metrics else None

        c.execute(
            """INSERT INTO model_versions
               (version, registered_at, model_path, source, trigger_reason,
                parent_version, status, training_metrics_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (version, timestamp, str(model_path), source, trigger_reason,
             parent_version, ModelStatus.REGISTERED, metrics_json, notes)
        )
        conn.commit()
        conn.close()

        logging.info(f"📦 Model Registry: Registered {version} from {source} ({trigger_reason})")
        return version

    def update_validation(
        self,
        version: str,
        passed: bool,
        decay_metrics: dict = None,
    ):
        """
        Update a model version with decay validation results.

        Args:
            version: The version string (e.g. "v3")
            passed: Whether the decay check passed
            decay_metrics: Dict of decay/golden-set metrics
        """
        new_status = ModelStatus.VALIDATED if passed else ModelStatus.REJECTED
        decay_json = json.dumps(decay_metrics) if decay_metrics else None

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """UPDATE model_versions
               SET status = ?, decay_metrics_json = ?
               WHERE version = ?""",
            (new_status, decay_json, version)
        )
        conn.commit()
        conn.close()

        result_str = "✅ VALIDATED" if passed else "🔴 REJECTED"
        logging.info(f"📦 Model Registry: {version} → {result_str}")

    def record_deployment(self, version: str):
        """
        Mark a model version as deployed to production.

        Args:
            version: The version string (e.g. "v3")
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """UPDATE model_versions
               SET status = ?, deployment_timestamp = ?
               WHERE version = ?""",
            (ModelStatus.DEPLOYED, timestamp, version)
        )
        conn.commit()
        conn.close()

        logging.info(f"📦 Model Registry: {version} → 🚀 DEPLOYED at {timestamp}")

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_current_production(self) -> Optional[dict]:
        """
        Get the latest deployed model version.

        Returns:
            Dict of version details, or None if no model has been deployed.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT * FROM model_versions
               WHERE status = ?
               ORDER BY deployment_timestamp DESC, id DESC LIMIT 1""",
            (ModelStatus.DEPLOYED,)
        )
        row = c.fetchone()
        conn.close()

        if row:
            return self._row_to_dict(row)
        return None

    def get_version(self, version: str) -> Optional[dict]:
        """
        Get details for a specific version.

        Args:
            version: The version string (e.g. "v3")

        Returns:
            Dict of version details, or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM model_versions WHERE version = ?", (version,))
        row = c.fetchone()
        conn.close()

        if row:
            return self._row_to_dict(row)
        return None

    def get_history(self, limit: int = 50) -> list:
        """
        Get version history, newest first.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of version dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM model_versions ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_metrics_trend(self, limit: int = 20) -> list:
        """
        Get training metrics across versions for charting.

        Returns only versions that have training metrics,
        ordered chronologically (oldest first for plotting).

        Args:
            limit: Maximum number of entries

        Returns:
            List of dicts with version, registered_at, status,
            and flattened training metric fields (loss, accuracy, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT version, registered_at, status, training_metrics_json
               FROM model_versions
               WHERE training_metrics_json IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()

        results = []
        for row in reversed(rows):  # Reverse to get chronological order
            entry = {
                "version": row["version"],
                "registered_at": row["registered_at"],
                "status": row["status"],
            }
            try:
                metrics = json.loads(row["training_metrics_json"])
                entry.update(metrics)
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(entry)

        return results

    def get_deployment_count(self) -> int:
        """Get total number of deployments."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM model_versions WHERE status = ?", (ModelStatus.DEPLOYED,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_version_count(self) -> int:
        """Get total number of registered versions."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM model_versions")
        count = c.fetchone()[0]
        conn.close()
        return count

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _row_to_dict(self, row) -> dict:
        """Convert a sqlite3.Row to a dict with parsed JSON fields."""
        d = dict(row)

        # Parse JSON fields
        for json_field in ("training_metrics_json", "decay_metrics_json"):
            raw = d.get(json_field)
            if raw:
                try:
                    d[json_field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass  # Leave as string if unparseable

        return d

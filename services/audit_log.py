# audit_log.py - Sentinel Audit Log Service
# Records every action Sentinel takes in a structured, queryable SQLite database.
#
# Usage:
#   from services.audit_log import SentinelAuditLog
#   audit = SentinelAuditLog()
#   audit.log("drift", "check_pass", {"score": 1.5, "threshold": 3.2})
#
# Categories: drift, data, training, decay, deployment, baseline, alert, state, rebase

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


class SentinelAuditLog:
    """
    Append-only audit log for all Sentinel actions.
    
    Every action is recorded with:
    - category: The subsystem (drift, data, training, deployment, etc.)
    - action: The specific action (check_pass, archive, start, etc.)
    - status: Outcome (success, failure, error)
    - details: Action-specific data as a JSON dict
    - metadata: Optional extra context (e.g., system state at time of action)
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path:
            self.db_path = db_path
        elif config and hasattr(config, "AUDIT_LOG_DB"):
            self.db_path = config.AUDIT_LOG_DB
        else:
            self.db_path = Path(__file__).resolve().parent.parent / "data" / "data_drift" / "audit_log.db"
        
        self._init_db()

    def _init_db(self):
        """Creates the audit_log table and index if they don't exist."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'success',
            details TEXT,
            metadata TEXT
        )''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_audit_category_time 
                     ON audit_log (category, timestamp)''')
        conn.commit()
        conn.close()

    def log(self, category: str, action: str, details: dict = None,
            status: str = "success", metadata: dict = None) -> int:
        """
        Record an action in the audit log.
        
        Args:
            category: Subsystem (drift, data, training, decay, deployment, baseline, alert, state, rebase)
            action: Specific action (check_pass, archive, start, success, etc.)
            details: Action-specific data dict
            status: Outcome - "success", "failure", or "error"
            metadata: Optional extra context
            
        Returns:
            The ID of the inserted audit entry
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details_json = json.dumps(details) if details else None
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO audit_log (timestamp, category, action, status, details, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, category, action, status, details_json, metadata_json)
            )
            entry_id = c.lastrowid
            conn.commit()
            conn.close()
            logging.debug(f"📝 Audit: [{category}] {action} ({status})")
            return entry_id
        except Exception as e:
            logging.error(f"🔴 Audit log write failed: {e}")
            return -1

    def query(self, category: str = None, action: str = None, status: str = None,
              since: str = None, until: str = None, limit: int = 50) -> list:
        """
        Query audit log entries with optional filters.
        
        Args:
            category: Filter by category
            action: Filter by action
            status: Filter by status
            since: ISO timestamp lower bound (inclusive)
            until: ISO timestamp upper bound (inclusive)
            limit: Maximum entries to return (default 50)
            
        Returns:
            List of dicts with id, timestamp, category, action, status, details, metadata
        """
        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT id, timestamp, category, action, status, details, metadata FROM audit_log{where_clause} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(query, params)
            rows = c.fetchall()
            conn.close()

            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "category": row["category"],
                    "action": row["action"],
                    "status": row["status"],
                    "details": json.loads(row["details"]) if row["details"] else None,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                }
                for row in rows
            ]
        except Exception as e:
            logging.error(f"🔴 Audit log query failed: {e}")
            return []

    def get_last_deployment(self) -> dict | None:
        """
        Get the most recent deployment entry.
        
        Returns:
            Dict of the last deployment entry, or None if no deployments recorded.
        """
        results = self.query(category="deployment", action="success", limit=1)
        return results[0] if results else None

    def get_timeline(self, limit: int = 100) -> list:
        """
        Get recent audit entries for dashboard display.
        
        Returns:
            List of recent audit entries, newest first.
        """
        return self.query(limit=limit)

    def get_category_counts(self) -> dict:
        """
        Get count of entries per category for dashboard summary.
        
        Returns:
            Dict mapping category to entry count.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT category, COUNT(*) FROM audit_log GROUP BY category ORDER BY COUNT(*) DESC")
            rows = c.fetchall()
            conn.close()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logging.error(f"🔴 Audit log count query failed: {e}")
            return {}


if __name__ == "__main__":
    import tempfile
    import os

    print("\n📝 AUDIT LOG SERVICE TEST\n")
    print("-" * 40)

    # Use a temporary database for testing
    tmp_db = Path(tempfile.mkdtemp()) / "test_audit.db"
    audit = SentinelAuditLog(db_path=tmp_db)
    print(f"✅ Initialized audit log at: {tmp_db}")

    # Insert entries across all categories
    test_entries = [
        ("drift", "check_pass", {"score": 1.5, "threshold": 3.2}),
        ("drift", "check_fail", {"score": 4.8, "threshold": 3.2}),
        ("drift", "threshold_triggered", {"fails": 3, "total": 5}),
        ("data", "archive", {"dest_dir": "/data/history/20260210_FAIL_4.80pct"}),
        ("data", "discard", None),
        ("data", "purge_history", None),
        ("training", "start", {"data_path": "/data/history"}),
        ("training", "success", {"loss": 0.023, "accuracy": 0.97}),
        ("decay", "check_pass", {"score": 1.2, "threshold": 3.0}),
        ("deployment", "start", {"model": "challenger_v2.pth"}),
        ("deployment", "success", {"model": "challenger_v2.pth"}),
        ("baseline", "golden_set_update", {"status": "success"}),
        ("baseline", "training_data_update", {"status": "success"}),
        ("alert", "fire", {"level": "WARNING", "message": "Drift threshold exceeded"}),
        ("state", "transition", {"from": "WARNING", "to": "NOMINAL"}),
        ("rebase", "start", {"change_type": "new_model", "method": "new_training_data"}),
        ("rebase", "complete", {"success": True}),
    ]

    for category, action, details in test_entries:
        entry_id = audit.log(category, action, details)
        print(f"  📌 [{category}] {action} -> ID {entry_id}")

    print(f"\n✅ Inserted {len(test_entries)} entries")

    # Test queries
    print("\n--- Query Tests ---")

    drift_entries = audit.query(category="drift")
    print(f"  Drift entries: {len(drift_entries)} (expected 3)")
    assert len(drift_entries) == 3, f"Expected 3 drift entries, got {len(drift_entries)}"

    deployment_entries = audit.query(category="deployment")
    print(f"  Deployment entries: {len(deployment_entries)} (expected 2)")
    assert len(deployment_entries) == 2, f"Expected 2 deployment entries, got {len(deployment_entries)}"

    last_deploy = audit.get_last_deployment()
    print(f"  Last deployment: {last_deploy['details']['model'] if last_deploy else 'None'}")
    assert last_deploy is not None, "Expected a deployment entry"

    timeline = audit.get_timeline(limit=5)
    print(f"  Timeline (limit 5): {len(timeline)} entries")
    assert len(timeline) == 5, f"Expected 5 timeline entries, got {len(timeline)}"

    counts = audit.get_category_counts()
    print(f"  Category counts: {counts}")
    assert counts["drift"] == 3
    assert counts["deployment"] == 2

    # Cleanup
    os.remove(tmp_db)
    os.rmdir(tmp_db.parent)

    print("\n" + "-" * 40)
    print("✅ Audit Log Service Test Complete")

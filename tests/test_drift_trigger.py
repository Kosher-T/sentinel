import unittest
import sqlite3
import tempfile
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import all_config as config
# Patch config constants for the test to matches our expectations
config.RETRAIN_TRIGGER_COUNT = 3
config.DRIFT_FAILURE_RATIO = 0.8
config.TIMEFRAME_WINDOW = 5

import sentinel_watch

class TestTriggerLogic(unittest.TestCase):
    def setUp(self):
        # Create a temporary database
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.test_dir.name) / "test_drift.db"
        
        # Initialize DB schema
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE drift_logs
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, data_path TEXT, root_cause_json TEXT)''')
        conn.commit()
        conn.close()
        
        # Initialize SentinelWatch with patched DB path
        self.watcher = sentinel_watch.SentinelWatch()
        self.watcher.db_path = self.db_path
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def insert_logs(self, statuses):
        """Insert logs with timestamps decreasing from now."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now()
        
        for i, status in enumerate(statuses):
            # i=0 is newest (t=now), i=1 is older (t=now-1), ...
            # We want statuses[0] to be the NEWEST.
            ts = (now - timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S')
            c.execute("INSERT INTO drift_logs (timestamp, drift_score, status) VALUES (?, ?, ?)",
                      (ts, 50.0 if status == "FAIL" else 10.0, status))
        conn.commit()
        conn.close()

    def test_consecutive_failures_trigger(self):
        # 3 FAILs (Newest -> Oldest)
        self.insert_logs(["FAIL", "FAIL", "FAIL"])
        triggered, fails, total = self.watcher.check_drift_history()
        self.assertTrue(triggered, "Should trigger on 3 consecutive failures")
        self.assertEqual(fails, 3)

    def test_ratio_trigger(self):
        # 4 FAILs out of 5 (80% >= 0.8)
        # Order: FAIL, FAIL, PASS, FAIL, FAIL
        self.insert_logs(["FAIL", "FAIL", "PASS", "FAIL", "FAIL"])
        triggered, fails, total = self.watcher.check_drift_history()
        self.assertTrue(triggered, "Should trigger on 4/5 failures (80% check)")
        self.assertEqual(fails, 4)

    def test_mixed_no_trigger(self):
        # 3 FAILs out of 5 (60% < 0.8), not consecutive (max 2)
        # Order: FAIL, FAIL, PASS, FAIL, PASS
        self.insert_logs(["FAIL", "FAIL", "PASS", "FAIL", "PASS"])
        triggered, fails, total = self.watcher.check_drift_history()
        self.assertFalse(triggered, "Should NOT trigger on 3/5 mixed (60%)")
        self.assertEqual(fails, 3)

    def test_consecutive_break(self):
        # 3 FAILs but interrupted by recent PASS
        # Order: PASS, FAIL, FAIL, FAIL
        self.insert_logs(["PASS", "FAIL", "FAIL", "FAIL"])
        triggered, fails, total = self.watcher.check_drift_history()
        # Ratio: 3/4 = 75% < 80%. Consecutive: 0 (starts with PASS).
        self.assertFalse(triggered, "Should NOT trigger if recent pass breaks consecutive chain")

    def test_exact_ratio_boundary(self):
        # 4 FAILs in 5 = 80%. Just enough.
        self.insert_logs(["FAIL", "PASS", "FAIL", "FAIL", "FAIL"]) # 4 fails
        triggered, fails, total = self.watcher.check_drift_history()
        self.assertTrue(triggered, "Should trigger exactly on boundary (4/5)")

if __name__ == '__main__':
    unittest.main()

import unittest
import tempfile
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.alert_escalation import AlertEscalationTracker


class TestAlertEscalationTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.test_dir.name) / "test_escalation.json"
        self.callback = MagicMock()
        self.tracker = AlertEscalationTracker(
            state_file=self.state_file,
            escalation_callback=self.callback,
            timeout_minutes=1,       # 1 minute for faster tests
            final_timeout_minutes=2,  # 2 minutes for faster tests
        )

    def tearDown(self):
        self.tracker.stop_watchdog()
        self.test_dir.cleanup()

    # ----- Tracking -----

    def test_track_warning_alert(self):
        """Level 2 (WARNING) alerts are tracked."""
        alert_id = self.tracker.track(level=2, event_type="retraining_start", message="Test warning")
        self.assertIsNotNone(alert_id)
        self.assertEqual(len(self.tracker.get_pending()), 1)

    def test_track_critical_alert(self):
        """Level 3 (CRITICAL) alerts are tracked."""
        alert_id = self.tracker.track(level=3, event_type="decay_fail", message="Test critical")
        self.assertIsNotNone(alert_id)
        self.assertEqual(len(self.tracker.get_pending()), 1)

    def test_ignore_info_alert(self):
        """Level 1 (INFO) alerts are NOT tracked."""
        alert_id = self.tracker.track(level=1, event_type="deployment", message="Test info")
        self.assertIsNone(alert_id)
        self.assertEqual(len(self.tracker.get_pending()), 0)

    def test_track_stores_details(self):
        """Tracked entries contain the expected fields."""
        self.tracker.track(level=2, event_type="retraining_start", message="Drift detected")
        pending = self.tracker.get_pending()
        self.assertEqual(len(pending), 1)
        entry = pending[0]
        self.assertEqual(entry["level"], 2)
        self.assertEqual(entry["event_type"], "retraining_start")
        self.assertEqual(entry["message"], "Drift detected")
        self.assertEqual(entry["escalation_level"], 0)
        self.assertFalse(entry["acknowledged"])
        self.assertIn("elapsed_seconds", entry)
        self.assertIn("elapsed_display", entry)

    # ----- Acknowledge -----

    def test_acknowledge_single(self):
        """Acknowledging removes a specific alert from pending."""
        alert_id = self.tracker.track(level=2, event_type="test", message="Test")
        self.assertEqual(self.tracker.get_pending_count(), 1)

        result = self.tracker.acknowledge(alert_id)
        self.assertTrue(result)
        self.assertEqual(self.tracker.get_pending_count(), 0)

    def test_acknowledge_nonexistent(self):
        """Acknowledging a non-existent alert returns False."""
        result = self.tracker.acknowledge("nonexistent_id")
        self.assertFalse(result)

    def test_acknowledge_all(self):
        """Bulk acknowledge clears all pending alerts."""
        self.tracker.track(level=2, event_type="test1", message="One")
        self.tracker.track(level=3, event_type="test2", message="Two")
        self.tracker.track(level=2, event_type="test3", message="Three")
        self.assertEqual(self.tracker.get_pending_count(), 3)

        count = self.tracker.acknowledge_all()
        self.assertEqual(count, 3)
        self.assertEqual(self.tracker.get_pending_count(), 0)

    def test_acknowledge_all_empty(self):
        """Bulk acknowledge on empty list returns 0."""
        count = self.tracker.acknowledge_all()
        self.assertEqual(count, 0)

    # ----- Escalation Logic -----

    def test_no_escalation_before_timeout(self):
        """Alerts within the timeout are not escalated."""
        self.tracker.track(level=2, event_type="test", message="Fresh alert")
        self.tracker.check_escalations()
        self.callback.assert_not_called()

        # Verify escalation level is still 0
        pending = self.tracker.get_pending()
        self.assertEqual(pending[0]["escalation_level"], 0)

    def test_escalation_level_1(self):
        """Alert past first timeout triggers level 1 escalation callback."""
        self.tracker.track(level=2, event_type="test", message="Old alert")

        # Manually backdate the fired_at timestamp past the timeout
        with self.tracker._lock:
            self.tracker.pending[0]["fired_at"] = (
                datetime.now() - timedelta(minutes=2)
            ).isoformat()

        self.tracker.check_escalations()

        # Callback should have been called with escalation level 1
        self.callback.assert_called_once()
        call_args = self.callback.call_args
        self.assertEqual(call_args[0][1], 1)  # escalation_level

        # Entry should be at escalation level 1
        with self.tracker._lock:
            self.assertEqual(self.tracker.pending[0]["escalation_level"], 1)

    def test_escalation_level_2(self):
        """Alert past final timeout triggers level 2 escalation callback."""
        self.tracker.track(level=3, event_type="critical_test", message="Very old alert")

        # Backdate past the final timeout
        with self.tracker._lock:
            self.tracker.pending[0]["fired_at"] = (
                datetime.now() - timedelta(minutes=3)
            ).isoformat()
            # Simulate already being escalated to level 1
            self.tracker.pending[0]["escalation_level"] = 1

        self.tracker.check_escalations()

        self.callback.assert_called_once()
        call_args = self.callback.call_args
        self.assertEqual(call_args[0][1], 2)  # escalation_level = FINAL

    def test_no_double_escalation(self):
        """An already-escalated alert at max level is not re-escalated."""
        self.tracker.track(level=3, event_type="test", message="Already maxed")

        with self.tracker._lock:
            self.tracker.pending[0]["fired_at"] = (
                datetime.now() - timedelta(minutes=5)
            ).isoformat()
            self.tracker.pending[0]["escalation_level"] = 2  # Already at final

        self.tracker.check_escalations()
        self.callback.assert_not_called()

    def test_acknowledged_alerts_not_escalated(self):
        """Acknowledged alerts are not escalated even if past timeout."""
        alert_id = self.tracker.track(level=2, event_type="test", message="Acked")
        self.tracker.acknowledge(alert_id)

        # Force check (should have no pending to escalate)
        self.tracker.check_escalations()
        self.callback.assert_not_called()

    # ----- Persistence -----

    def test_state_persists_across_instances(self):
        """Pending alerts survive tracker restart."""
        self.tracker.track(level=2, event_type="persist_test", message="Should survive")
        self.assertEqual(self.tracker.get_pending_count(), 1)

        # Create new tracker pointing to same state file
        tracker2 = AlertEscalationTracker(
            state_file=self.state_file,
            escalation_callback=self.callback,
        )
        self.assertEqual(tracker2.get_pending_count(), 1)
        pending = tracker2.get_pending()
        self.assertEqual(pending[0]["event_type"], "persist_test")

    # ----- Counts -----

    def test_pending_count(self):
        """get_pending_count returns accurate count."""
        self.assertEqual(self.tracker.get_pending_count(), 0)
        self.tracker.track(level=2, event_type="t1", message="One")
        self.assertEqual(self.tracker.get_pending_count(), 1)
        aid = self.tracker.track(level=3, event_type="t2", message="Two")
        self.assertEqual(self.tracker.get_pending_count(), 2)
        self.tracker.acknowledge(aid)
        self.assertEqual(self.tracker.get_pending_count(), 1)

    # ----- Elapsed Formatting -----

    def test_elapsed_display_seconds(self):
        """Elapsed display shows seconds for short intervals."""
        self.tracker.track(level=2, event_type="test", message="Just now")
        pending = self.tracker.get_pending()
        # Should be a short interval like "0s" or "1s"
        self.assertIn("s", pending[0]["elapsed_display"])

    def test_elapsed_display_minutes(self):
        """Elapsed display shows minutes for longer intervals."""
        self.tracker.track(level=2, event_type="test", message="Old")
        with self.tracker._lock:
            self.tracker.pending[0]["fired_at"] = (
                datetime.now() - timedelta(minutes=5)
            ).isoformat()
        pending = self.tracker.get_pending()
        self.assertIn("m", pending[0]["elapsed_display"])


if __name__ == '__main__':
    unittest.main()

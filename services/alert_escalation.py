"""
Alert Escalation Tracker for Sentinel.

Monitors unacknowledged WARNING/CRITICAL alerts and escalates them
to a secondary on-call contact after configurable timeouts.

Escalation Levels:
  0 = Initial alert fired (pending)
  1 = First escalation  → email secondary on-call
  2 = Final escalation  → email BOTH contacts urgently
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL_ESCALATION: %(message)s')

try:
    import all_config as config
    PROJECT_ROOT = config.PROJECT_ROOT
    ESCALATION_TIMEOUT_MINUTES = getattr(config, 'ESCALATION_TIMEOUT_MINUTES', 15)
    ESCALATION_FINAL_TIMEOUT_MINUTES = getattr(config, 'ESCALATION_FINAL_TIMEOUT_MINUTES', 30)
    SECONDARY_ONCALL_EMAIL = getattr(config, 'SECONDARY_ONCALL_EMAIL', '')
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ESCALATION_TIMEOUT_MINUTES = 15
    ESCALATION_FINAL_TIMEOUT_MINUTES = 30
    SECONDARY_ONCALL_EMAIL = ''

STATE_FILE = PROJECT_ROOT / "alert_escalation_state.json"
WATCHDOG_INTERVAL_SECONDS = 30


class AlertEscalationTracker:
    """Tracks pending alerts and escalates unacknowledged ones."""

    def __init__(self, state_file=None, escalation_callback=None,
                 timeout_minutes=None, final_timeout_minutes=None):
        """
        Args:
            state_file: Path for persisting state. Defaults to PROJECT_ROOT/alert_escalation_state.json
            escalation_callback: Callable(alert_dict, escalation_level) invoked on escalation.
                                 The caller (SentinelAlert) provides this to send the actual email.
            timeout_minutes: Minutes before first escalation.
            final_timeout_minutes: Minutes before final escalation.
        """
        self.state_file = Path(state_file) if state_file else STATE_FILE
        self.escalation_callback = escalation_callback
        self.timeout = timedelta(minutes=timeout_minutes or ESCALATION_TIMEOUT_MINUTES)
        self.final_timeout = timedelta(minutes=final_timeout_minutes or ESCALATION_FINAL_TIMEOUT_MINUTES)
        self._lock = threading.Lock()
        self._watchdog_thread = None
        self._stop_event = threading.Event()

        # Load or initialize state
        self.pending = self._load_state()

    # ---- Persistence ----

    def _load_state(self):
        """Load pending alerts from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return data.get("pending", [])
            except Exception:
                return []
        return []

    def _save_state(self):
        """Persist current pending alerts to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({"pending": self.pending}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save escalation state: {e}")

    # ---- Core API ----

    def track(self, level, event_type, message):
        """
        Start tracking an alert for escalation.
        Only level >= 2 (WARNING, CRITICAL) are tracked.

        Returns:
            str: alert_id if tracked, None if skipped.
        """
        if level < 2:
            return None

        alert_id = str(uuid.uuid4())[:8]
        entry = {
            "alert_id": alert_id,
            "level": level,
            "event_type": event_type,
            "message": message,
            "fired_at": datetime.now().isoformat(),
            "escalation_level": 0,
            "acknowledged": False,
        }

        with self._lock:
            self.pending.append(entry)
            self._save_state()

        logging.info(f"🔔 Tracking alert {alert_id} ({event_type}) for escalation")
        return alert_id

    def acknowledge(self, alert_id):
        """
        Acknowledge a specific alert, stopping its escalation.

        Returns:
            bool: True if found and acknowledged.
        """
        with self._lock:
            for entry in self.pending:
                if entry["alert_id"] == alert_id and not entry["acknowledged"]:
                    entry["acknowledged"] = True
                    logging.info(f"✅ Alert {alert_id} acknowledged")
                    self._cleanup_acknowledged()
                    self._save_state()
                    return True
        return False

    def acknowledge_all(self):
        """
        Acknowledge all pending alerts. Used when system returns to NOMINAL.

        Returns:
            int: Number of alerts acknowledged.
        """
        with self._lock:
            count = sum(1 for e in self.pending if not e["acknowledged"])
            for entry in self.pending:
                entry["acknowledged"] = True
            self._cleanup_acknowledged()
            self._save_state()

        if count > 0:
            logging.info(f"✅ Bulk acknowledged {count} pending alert(s)")
        return count

    def get_pending(self):
        """Return list of unacknowledged alerts with time-since-fired."""
        now = datetime.now()
        with self._lock:
            result = []
            for entry in self.pending:
                if entry["acknowledged"]:
                    continue
                fired_at = datetime.fromisoformat(entry["fired_at"])
                elapsed = now - fired_at
                result.append({
                    **entry,
                    "elapsed_seconds": int(elapsed.total_seconds()),
                    "elapsed_display": self._format_elapsed(elapsed),
                })
            return result

    def get_pending_count(self):
        """Return count of unacknowledged alerts."""
        with self._lock:
            return sum(1 for e in self.pending if not e["acknowledged"])

    # ---- Escalation Logic ----

    def check_escalations(self):
        """
        Check all pending alerts and escalate any that have exceeded their timeout.
        Called periodically by the watchdog thread.
        """
        now = datetime.now()
        escalated = []

        with self._lock:
            for entry in self.pending:
                if entry["acknowledged"]:
                    continue

                fired_at = datetime.fromisoformat(entry["fired_at"])
                elapsed = now - fired_at

                if entry["escalation_level"] == 0 and elapsed >= self.timeout:
                    # First escalation
                    entry["escalation_level"] = 1
                    escalated.append((dict(entry), 1))
                    logging.warning(
                        f"⚠️ Alert {entry['alert_id']} escalated to level 1 "
                        f"(unacknowledged for {self._format_elapsed(elapsed)})"
                    )

                elif entry["escalation_level"] == 1 and elapsed >= self.final_timeout:
                    # Final escalation
                    entry["escalation_level"] = 2
                    escalated.append((dict(entry), 2))
                    logging.critical(
                        f"🔴 Alert {entry['alert_id']} FINAL escalation (level 2) "
                        f"(unacknowledged for {self._format_elapsed(elapsed)})"
                    )

            if escalated:
                self._save_state()

        # Fire callbacks outside the lock to avoid deadlocks
        for alert_dict, esc_level in escalated:
            if self.escalation_callback:
                try:
                    self.escalation_callback(alert_dict, esc_level)
                except Exception as e:
                    logging.error(f"Escalation callback failed for {alert_dict['alert_id']}: {e}")

    # ---- Watchdog Thread ----

    def start_watchdog(self, interval_seconds=None):
        """Start the background watchdog thread that checks for escalations."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return  # Already running

        interval = interval_seconds or WATCHDOG_INTERVAL_SECONDS
        self._stop_event.clear()

        def _watchdog_loop():
            logging.info(f"🐕 Escalation watchdog started (interval: {interval}s)")
            while not self._stop_event.is_set():
                try:
                    self.check_escalations()
                except Exception as e:
                    logging.error(f"Watchdog error: {e}")
                self._stop_event.wait(interval)
            logging.info("🐕 Escalation watchdog stopped")

        self._watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            name="EscalationWatchdog",
            daemon=True
        )
        self._watchdog_thread.start()

    def stop_watchdog(self):
        """Stop the background watchdog thread."""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)

    # ---- Helpers ----

    def _cleanup_acknowledged(self):
        """Remove acknowledged entries from the pending list."""
        self.pending = [e for e in self.pending if not e["acknowledged"]]

    @staticmethod
    def _format_elapsed(delta):
        """Format a timedelta into human-readable string."""
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        remaining_min = minutes % 60
        return f"{hours}h {remaining_min}m"

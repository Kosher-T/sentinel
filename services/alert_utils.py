import sys
import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] SENTINEL_ALERT: %(message)s')

try:
    import all_config as config
    PROJECT_ROOT = config.PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent

# Constants
STATE_FILE = PROJECT_ROOT / "sentinel_alert_state.json"
TEST_STATE_FILE = PROJECT_ROOT / "sentinel_alert_test_cycle.json"

TRUST_THRESHOLDS = {
    "drift_pass": 3,
    "retraining": 1,
    "deployment": 1
}

# Email Configuration
SMTP_SERVER = "sandbox.smtp.mailtrap.io"
SMTP_PORT = 587
SMTP_USER = "29b057e9b13807"
SMTP_PASSWORD = "22a33396821f85"
RECIPIENT_EMAIL = "itorousa@gmail.com"
SECONDARY_ONCALL_EMAIL = getattr(config, 'SECONDARY_ONCALL_EMAIL', '') if 'config' in dir() else ''

# Brand Assets
BRAND_COLOR = "#00E5FF" # Electric Cyan
FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO_STACK = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"

# Escalation Service
try:
    from services.alert_escalation import AlertEscalationTracker
    ESCALATION_AVAILABLE = True
except ImportError:
    try:
        from alert_escalation import AlertEscalationTracker  # type: ignore
        ESCALATION_AVAILABLE = True
    except ImportError:
        ESCALATION_AVAILABLE = False


def _is_headless():
    """Detect if running on a headless/cloud system where desktop notifications are useless."""
    # No display server = no desktop notifications
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Running inside Docker
    if Path("/.dockerenv").exists():
        return True
    return False


IS_HEADLESS = _is_headless()

class SentinelAlert:
    def __init__(self):
        self.state = self._load_state(STATE_FILE, default={
            "drift_pass_count": 0,
            "retraining_count": 0,
            "deployment_count": 0
        })
        
        # Initialize escalation tracker with callback
        self.escalation_tracker = None
        if ESCALATION_AVAILABLE:
            self.escalation_tracker = AlertEscalationTracker(
                escalation_callback=self._handle_escalation
            )

    def _load_state(self, filepath, default):
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_state(self, filepath, data):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save state to {filepath}: {e}")

    def _get_emoji(self, level, event_type):
        if level == 3: return "🔴"
        if level == 2: return "⚠️"
        if level == 1:
            if event_type == "archive": return "💾"
            return "🟢"
        return ""

    def _get_theme(self, level):
        """Returns (AccentColor, GlowColor, Label)"""
        if level == 3: return "#FF3B30", "rgba(255, 59, 48, 0.1)", "CRITICAL_GATE_FAILURE"
        if level == 2: return "#FFCC00", "rgba(255, 204, 0, 0.1)", "ANOMALY_DETECTED"
        if level == 1: return BRAND_COLOR, "rgba(0, 229, 255, 0.1)", "SYSTEM_NOMINAL"
        return "#8E8E93", "rgba(142, 142, 147, 0.1)", "SYSTEM_LOG"

    def _send_system_notification(self, title, message):
        if IS_HEADLESS:
            logging.debug(f"Headless environment detected. Skipping system notification: {title}")
            return
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="SentinelWatch",
                timeout=10
            ) # type: ignore
        except ImportError:
            logging.warning("Plyer not installed. Skipping system notification.")
        except Exception as e:
            logging.error(f"System notification failed: {e}")

    def _generate_email_html(self, level, title, message, metrics):
        accent_color, glow_bg, label = self._get_theme(level)
        
        # Build Terminal Metrics
        metrics_html = ""
        if metrics:
            rows = ""
            for k, v in metrics.items():
                rows += f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 0; color: #8e8e93; font-size: 11px; text-transform: uppercase; text-align: left;">{k.replace('_', ' ')}</td>
                    <td style="padding: 12px 0; color: {accent_color}; font-family: {MONO_STACK}; font-weight: bold; text-align: right;">{v}</td>
                </tr>
                """
            metrics_html = f"""
            <div style="background: rgba(28, 28, 30, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; margin-top: 25px; overflow: hidden;">
                <div style="padding: 10px 15px; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 10px; color: #636366; letter-spacing: 1px; font-weight: bold; text-transform: uppercase;">
                    Terminal Data Output
                </div>
                <div style="padding: 0 15px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        {rows}
                    </table>
                </div>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body style="background-color: #050505; padding: 20px 0; font-family: {FONT_STACK}; margin: 0;">
                <div style="max-width: 550px; width: 92%; margin: 20px auto; background-color: #0f0f0f; border-radius: 16px; overflow: hidden; border: 1px solid #222; box-shadow: 0 20px 40px rgba(0,0,0,0.8);">
                    
                    <!-- Aperture Gate Header -->
                    <div style="padding: 40px 20px 25px 20px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <div style="font-family: {MONO_STACK}; color: #444; font-size: 11px; letter-spacing: 5px; margin-bottom: 12px; text-transform: uppercase;">
                            |— Sentinel —|
                        </div>
                        <div style="display: inline-block; padding: 5px 14px; border-radius: 4px; background-color: {glow_bg}; color: {accent_color}; font-size: 10px; font-family: {MONO_STACK}; font-weight: bold; border: 1px solid {accent_color}; letter-spacing: 0.5px;">
                            {label}
                        </div>
                        <h1 style="color: #ffffff; margin-top: 25px; margin-bottom: 0; font-size: 24px; font-weight: 300; letter-spacing: -0.5px; line-height: 1.2;">
                            {title.split('] ')[-1] if ']' in title else title}
                        </h1>
                    </div>
                    
                    <!-- Content Area (Glass Content) -->
                    <div style="padding: 30px 10% 40px 10%; background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0) 100%);">
                        <p style="color: #a1a1aa; font-size: 15px; line-height: 1.6; text-align: center; margin: 0 auto; max-width: 400px;">
                            {message}
                        </p>
                        {metrics_html}
                    </div>
                    
                    <!-- Footer -->
                    <div style="padding: 25px; text-align: center; background: #0a0a0a; border-top: 1px solid rgba(255,255,255,0.05);">
                        <div style="font-family: {MONO_STACK}; font-size: 9px; color: #444; letter-spacing: 1px;">
                            SIGNAL_ENCRYPTION_ID :: {datetime.now().strftime('%H:%M:%S_UTC')}
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """
        return html

    def _send_email(self, level, title, message, metrics):
        html_content = self._generate_email_html(level, title, message, metrics)
        
        if SMTP_SERVER == "smtp.example.com":
            logging.info(f"📧 [SIMULATION] To: {RECIPIENT_EMAIL} | Subject: {title}")
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = f"Sentinel Engine <{SMTP_USER}>"
            msg["To"] = RECIPIENT_EMAIL
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, RECIPIENT_EMAIL, msg.as_string())
            logging.info(f"📧 Signal Broadcast Successful: {title}")
        except Exception as e:
            logging.error(f"❌ Broadcast Interrupted: {e}")

    def fire(self, level, event_type, message, metrics=None):
        emoji = self._get_emoji(level, event_type)
        title_prefix = {1: "OK", 2: "WARN", 3: "CRIT"}.get(level, "LOG")
        title = f"{emoji} [SENTINEL:{title_prefix}]"
        
        notify_msg = message
        if metrics:
            first_key = list(metrics.keys())[0]
            notify_msg += f" ({first_key}: {metrics[first_key]})"
            
        self._send_system_notification(title, notify_msg)

        should_email = True
        if level == 1:
            count_key = f"{event_type}_count"
            if count_key in self.state:
                if self.state[count_key] >= TRUST_THRESHOLDS.get(event_type, 999):
                    should_email = False
                    logging.info(f"🔕 Channel suppressed for {event_type} (Nominal Threshold Reached).")
                else:
                    self.state[count_key] += 1
                    self._save_state(STATE_FILE, self.state)
        
        if should_email:
            # Reverted to original multi-info format, but removed the "- " hyphen
            email_subject = f"{title} {event_type.replace('_', ' ').title()}"
            self._send_email(level, email_subject, message, metrics)
        
        # Track for escalation (WARNING and CRITICAL only)
        if self.escalation_tracker and level >= 2:
            self.escalation_tracker.track(level, event_type, message)

    def _handle_escalation(self, alert_dict, escalation_level):
        """
        Callback invoked by AlertEscalationTracker when an alert needs escalation.
        
        Level 1: Send to secondary on-call
        Level 2: Send urgent to both contacts
        """
        alert_id = alert_dict.get('alert_id', 'unknown')
        event_type = alert_dict.get('event_type', 'unknown')
        original_msg = alert_dict.get('message', '')
        original_level = alert_dict.get('level', 2)
        
        if escalation_level == 1:
            esc_msg = (
                f"⚠️ ESCALATION: The following alert has been unacknowledged "
                f"for {getattr(config, 'ESCALATION_TIMEOUT_MINUTES', 15)} minutes.\n\n"
                f"Original Alert ({event_type}): {original_msg}"
            )
            subject = f"🔴 [SENTINEL:ESCALATION] Unacknowledged Alert - {event_type.replace('_', ' ').title()}"
            
            # Send to secondary on-call if configured
            target = SECONDARY_ONCALL_EMAIL or RECIPIENT_EMAIL
            logging.warning(f"📧 Escalation Level 1 → {target} for alert {alert_id}")
            self._send_email(3, subject, esc_msg, None)
            
        elif escalation_level == 2:
            esc_msg = (
                f"🔴 FINAL ESCALATION: Alert still unacknowledged after "
                f"{getattr(config, 'ESCALATION_FINAL_TIMEOUT_MINUTES', 30)} minutes.\n\n"
                f"Original Alert ({event_type}): {original_msg}\n\n"
                f"Immediate attention required."
            )
            subject = f"🔴🔴 [SENTINEL:FINAL ESCALATION] URGENT - {event_type.replace('_', ' ').title()}"
            
            logging.critical(f"📧 FINAL Escalation → ALL contacts for alert {alert_id}")
            self._send_email(3, subject, esc_msg, None)

    def start_escalation_watchdog(self):
        """Start the background escalation watchdog thread."""
        if self.escalation_tracker:
            self.escalation_tracker.start_watchdog()

    def acknowledge_alert(self, alert_id):
        """Acknowledge a specific alert to stop its escalation."""
        if self.escalation_tracker:
            return self.escalation_tracker.acknowledge(alert_id)
        return False

    def acknowledge_all_alerts(self):
        """Acknowledge all pending alerts (e.g., when system returns to NOMINAL)."""
        if self.escalation_tracker:
            return self.escalation_tracker.acknowledge_all()
        return 0

    def get_pending_alerts(self):
        """Get list of unacknowledged pending alerts."""
        if self.escalation_tracker:
            return self.escalation_tracker.get_pending()
        return []

    def get_pending_alert_count(self):
        """Get count of unacknowledged alerts."""
        if self.escalation_tracker:
            return self.escalation_tracker.get_pending_count()
        return 0


if __name__ == "__main__":
    print("\n📡 INITIALIZING SIGNAL BROADCAST TEST 📡")
    print("-----------------------------------------")
    
    cycle_idx = 0
    if TEST_STATE_FILE.exists():
        try:
            with open(TEST_STATE_FILE, 'r') as f:
                data = json.load(f)
                cycle_idx = data.get("index", 0)
        except: pass

    scenarios = [
        {
            "level": 1, 
            "event": "deployment", 
            "msg": "Challenger model verification successful. Production instance swap completed.", 
            "metrics": {"loss": 0.045, "variance": "1.2%", "node_id": "SN-09"}
        },
        {
            "level": 2, 
            "event": "retraining", 
            "msg": "Significant performance drift detected. Triggering automated retraining sequence.", 
            "metrics": {"fail_ratio": "4/5", "drift": 0.45}
        },
        {
            "level": 3, 
            "event": "decay_fail", 
            "msg": "Gatekeeper Rejection: Challenger model shows unacceptable output variance. Deployment halted.", 
            "metrics": {"decay": "8.5%", "threshold": "5.0%"}
        }
    ]

    current_scenario = scenarios[cycle_idx % len(scenarios)]
    print(f"Scenario: {current_scenario['event'].upper()}")
    
    alert = SentinelAlert()
    alert.fire(
        level=current_scenario['level'],
        event_type=current_scenario['event'],
        message=current_scenario['msg'],
        metrics=current_scenario['metrics']
    )

    with open(TEST_STATE_FILE, 'w') as f:
        json.dump({"index": cycle_idx + 1}, f)

    print("-----------------------------------------")
    print("✅ Broadcast Test Concluded.")
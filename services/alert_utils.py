import sys
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

class SentinelAlert:
    def __init__(self):
        self.state = self._load_state(STATE_FILE, default={
            "drift_pass_count": 0,
            "retraining_count": 0,
            "deployment_count": 0
        })

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
        """Returns (AccentColor, SecondaryColor, Label)"""
        if level == 3: return "#FF3B30", "#451a1a", "CRITICAL FAILURE"
        if level == 2: return "#FFCC00", "#3d361c", "SYSTEM WARNING"
        if level == 1: return "#34C759", "#1a301f", "SYSTEM NOMINAL"
        return "#8E8E93", "#2c2c2e", "SYSTEM LOG"

    def _send_system_notification(self, title, message):
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
        accent_color, bg_accent, label = self._get_theme(level)
        
        # Build Metrics Cards
        metrics_html = ""
        if metrics:
            cards = ""
            for k, v in metrics.items():
                cards += f"""
                <div style="display: inline-block; width: 45%; margin: 5px; padding: 10px; background-color: #1c1c1e; border-radius: 4px; border-left: 2px solid {accent_color};">
                    <div style="font-size: 10px; color: #8e8e93; text-transform: uppercase;">{k.replace('_', ' ')}</div>
                    <div style="font-size: 16px; color: #ffffff; font-family: 'Courier New', Courier, monospace; font-weight: bold;">{v}</div>
                </div>
                """
            metrics_html = f"<div style='margin-top: 20px; text-align: center;'>{cards}</div>"

        # Modern Font Stack (Prioritizes cleaner sans-serifs similar to Calibri/Segoe)
        font_stack = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

        html = f"""
        <html>
            <body style="background-color: #000000; padding: 20px; font-family: {font_stack};">
                <div style="max-width: 550px; margin: 0 auto; background-color: #121212; border-radius: 12px; overflow: hidden; border: 1px solid #2c2c2e; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                    
                    <!-- Top Accent Bar -->
                    <div style="height: 4px; background-color: {accent_color};"></div>
                    
                    <!-- Header Area -->
                    <div style="padding: 30px 20px; text-align: center; background-color: #121212;">
                        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; background-color: {bg_accent}; color: {accent_color}; font-size: 11px; font-weight: bold; letter-spacing: 1px; margin-bottom: 15px;">
                            {label}
                        </div>
                        <h1 style="color: #ffffff; margin: 0; font-size: 24px; letter-spacing: -0.5px;">{title.split('] ')[-1] if ']' in title else title}</h1>
                    </div>
                    
                    <!-- Message Body -->
                    <div style="padding: 0 40px 30px 40px; color: #d1d1d6; font-size: 15px; line-height: 1.6; text-align: center;">
                        {message}
                        {metrics_html}
                    </div>
                    
                    <!-- Footer -->
                    <div style="background-color: #1c1c1e; padding: 20px; text-align: center; border-top: 1px solid #2c2c2e;">
                        <div style="font-size: 11px; color: #636366; letter-spacing: 0.5px;">
                            SENTINEL_WATCH ENGINE v1.0<br>
                            {datetime.now().strftime('%d %b %Y | %H:%M:%S').upper()}
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
            msg["From"] = f"Sentinel Watch <{SMTP_USER}>"
            msg["To"] = RECIPIENT_EMAIL
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, RECIPIENT_EMAIL, msg.as_string())
            logging.info(f"📧 Transmission Successful: {title}")
        except Exception as e:
            logging.error(f"Transmission Failure: {e}")

    def fire(self, level, event_type, message, metrics=None):
        emoji = self._get_emoji(level, event_type)
        title_prefix = {1: "SUCCESS", 2: "WARNING", 3: "CRITICAL"}.get(level, "ALERT")
        title = f"{emoji} [SENTINEL: {title_prefix}]"
        
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
            email_subject = f"{title} - {event_type.replace('_', ' ').title()}"
            self._send_email(level, email_subject, message, metrics)


if __name__ == "__main__":
    print("\n⚡ SENTINEL SIGNAL TEST ⚡")
    print("-------------------------")
    
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
            "msg": "New Challenger Model has successfully replaced Production instances.", 
            "metrics": {"train_loss": 0.045, "decay": "1.2%", "version": "v2.0.4"}
        },
        {
            "level": 2, 
            "event": "retraining", 
            "msg": "Performance drift detected in recent data windows. Automated retraining is now active.", 
            "metrics": {"drift_window": "4/5", "score": 0.45}
        },
        {
            "level": 3, 
            "event": "decay_fail", 
            "msg": "Gatekeeper Alert: Challenger model failed the final decay check. Deployment has been blocked to protect production integrity.", 
            "metrics": {"decay_score": "8.5%", "limit": "5.0%"}
        }
    ]

    current_scenario = scenarios[cycle_idx % len(scenarios)]
    print(f"Broadcasting: {current_scenario['event'].upper()}")
    
    alert = SentinelAlert()
    alert.fire(
        level=current_scenario['level'],
        event_type=current_scenario['event'],
        message=current_scenario['msg'],
        metrics=current_scenario['metrics']
    )

    with open(TEST_STATE_FILE, 'w') as f:
        json.dump({"index": cycle_idx + 1}, f)

    print("-------------------------")
    print("✅ Broadcast Complete.")
"""
Alerts Module
==============
Handles notifications for pipeline events, anomalies, and failures.
Supports Slack, Email, and logging-based alerts.
"""

import json
import logging
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
from enum import Enum

import requests

from config import config

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'alerts.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_SUCCESS = "pipeline_success"
    PIPELINE_FAILURE = "pipeline_failure"
    VALIDATION_FAILURE = "validation_failure"
    ANOMALY_DETECTED = "anomaly_detected"
    BIAS_DETECTED = "bias_detected"
    DATA_QUALITY_ISSUE = "data_quality_issue"


# =============================================================================
# Alert Formatters
# =============================================================================

def format_slack_message(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None
) -> dict:
    """Format alert as Slack message with blocks."""
    
    # Color based on severity
    colors = {
        AlertSeverity.INFO: "#36a64f",      # Green
        AlertSeverity.WARNING: "#ffcc00",   # Yellow
        AlertSeverity.ERROR: "#ff6600",     # Orange
        AlertSeverity.CRITICAL: "#ff0000",  # Red
    }
    
    # Emoji based on alert type
    emojis = {
        AlertType.PIPELINE_START: "🚀",
        AlertType.PIPELINE_SUCCESS: "✅",
        AlertType.PIPELINE_FAILURE: "❌",
        AlertType.VALIDATION_FAILURE: "⚠️",
        AlertType.ANOMALY_DETECTED: "🔍",
        AlertType.BIAS_DETECTED: "⚖️",
        AlertType.DATA_QUALITY_ISSUE: "📊",
    }
    
    emoji = emojis.get(alert_type, "📢")
    color = colors.get(severity, "#808080")
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            }
        }
    ]
    
    # Add details if provided
    if details:
        detail_text = "\n".join([f"• *{k}*: {v}" for k, v in details.items()])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Details:*\n{detail_text}"
            }
        })
    
    # Add timestamp
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            }
        ]
    })
    
    return {
        "attachments": [{
            "color": color,
            "blocks": blocks
        }]
    }


def format_email_message(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None
) -> tuple:
    """Format alert as email subject and body."""
    
    severity_prefix = f"[{severity.value.upper()}]"
    subject = f"{severity_prefix} AutoMend Pipeline: {title}"
    
    body = f"""
AutoMend Pipeline Alert
========================

Type: {alert_type.value}
Severity: {severity.value}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{message}
"""
    
    if details:
        body += "\nDetails:\n"
        for key, value in details.items():
            body += f"  - {key}: {value}\n"
    
    body += """
---
This is an automated message from the AutoMend Data Pipeline.
"""
    
    return subject, body


# =============================================================================
# Alert Senders
# =============================================================================

def send_slack_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None
) -> bool:
    """Send alert to Slack webhook."""
    
    if not config.SLACK_WEBHOOK_URL:
        logger.warning("Slack webhook URL not configured, skipping Slack alert")
        return False
    
    try:
        payload = format_slack_message(alert_type, severity, title, message, details)
        
        response = requests.post(
            config.SLACK_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Slack alert sent: {title}")
            return True
        else:
            logger.error(f"Slack alert failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def send_email_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None
) -> bool:
    """Send alert via email."""
    
    if not config.ALERT_EMAIL:
        logger.warning("Alert email not configured, skipping email alert")
        return False
    
    try:
        subject, body = format_email_message(alert_type, severity, title, message, details)
        
        msg = MIMEMultipart()
        msg['From'] = config.ALERT_EMAIL
        msg['To'] = config.ALERT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            # Note: You'd need to add authentication here for production
            # server.login(username, password)
            server.send_message(msg)
        
        logger.info(f"Email alert sent: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def log_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None
) -> None:
    """Log alert to file (always executes as fallback)."""
    
    alert_record = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type.value,
        "severity": severity.value,
        "title": title,
        "message": message,
        "details": details
    }
    
    # Log to appropriate level
    log_func = {
        AlertSeverity.INFO: logger.info,
        AlertSeverity.WARNING: logger.warning,
        AlertSeverity.ERROR: logger.error,
        AlertSeverity.CRITICAL: logger.critical,
    }.get(severity, logger.info)
    
    log_func(f"ALERT [{alert_type.value}]: {title} - {message}")
    
    # Also append to alerts JSON file
    alerts_file = config.LOGS_DIR / "alerts_history.json"
    
    try:
        alerts = []
        if alerts_file.exists():
            try:
                with open(alerts_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        alerts = json.loads(content)
                        if not isinstance(alerts, list):
                            alerts = []
            except (json.JSONDecodeError, ValueError):
                # File is corrupted, start fresh
                logger.warning("alerts_history.json was corrupted, starting fresh")
                alerts = []
        
        alerts.append(alert_record)
        
        # Keep only last 1000 alerts
        alerts = alerts[-1000:]
        
        with open(alerts_file, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Failed to write alert to history: {e}")


# =============================================================================
# Main Alert Function
# =============================================================================

def send_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None,
    channels: list = None
) -> dict:
    """
    Send alert through configured channels.
    
    Args:
        alert_type: Type of alert
        severity: Severity level
        title: Alert title
        message: Alert message
        details: Additional details dict
        channels: List of channels ('slack', 'email', 'log'). Default: all configured
    
    Returns:
        dict: Results of sending through each channel
    """
    if channels is None:
        channels = ["slack", "email", "log"]
    
    results = {}
    
    # Always log
    if "log" in channels:
        log_alert(alert_type, severity, title, message, details)
        results["log"] = True
    
    # Send to Slack for warnings and above
    if "slack" in channels and severity != AlertSeverity.INFO:
        results["slack"] = send_slack_alert(alert_type, severity, title, message, details)
    
    # Send email for errors and above
    if "email" in channels and severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        results["email"] = send_email_alert(alert_type, severity, title, message, details)
    
    return results


# =============================================================================
# Convenience Functions for Common Alerts
# =============================================================================

def alert_pipeline_start(dag_id: str, run_id: str):
    """Send pipeline start notification."""
    send_alert(
        AlertType.PIPELINE_START,
        AlertSeverity.INFO,
        "Pipeline Started",
        f"DAG '{dag_id}' has started execution.",
        {"dag_id": dag_id, "run_id": run_id}
    )


def alert_pipeline_success(dag_id: str, run_id: str, duration: float, stats: dict = None):
    """Send pipeline success notification."""
    details = {
        "dag_id": dag_id,
        "run_id": run_id,
        "duration": f"{duration:.1f}s"
    }
    if stats:
        details.update(stats)
    
    send_alert(
        AlertType.PIPELINE_SUCCESS,
        AlertSeverity.INFO,
        "Pipeline Completed Successfully",
        f"DAG '{dag_id}' completed successfully.",
        details
    )


def alert_pipeline_failure(dag_id: str, run_id: str, task_id: str, error: str):
    """Send pipeline failure notification."""
    send_alert(
        AlertType.PIPELINE_FAILURE,
        AlertSeverity.CRITICAL,
        "Pipeline Failed",
        f"DAG '{dag_id}' failed at task '{task_id}'.",
        {
            "dag_id": dag_id,
            "run_id": run_id,
            "failed_task": task_id,
            "error": error[:500]  # Truncate long errors
        }
    )


def alert_validation_failure(validation_results: dict):
    """Send validation failure notification."""
    failed_checks = [f["check"] for f in validation_results.get("failed", [])]
    
    send_alert(
        AlertType.VALIDATION_FAILURE,
        AlertSeverity.ERROR,
        "Data Validation Failed",
        f"Data validation failed with {len(failed_checks)} check(s).",
        {
            "failed_checks": ", ".join(failed_checks[:5]),
            "total_failures": len(failed_checks)
        }
    )


def alert_anomalies_detected(anomaly_count: int, anomaly_types: list):
    """Send anomaly detection notification."""
    # Convert to list before slicing (sets are not subscriptable)
    unique_types = list(set(anomaly_types))[:5]
    send_alert(
        AlertType.ANOMALY_DETECTED,
        AlertSeverity.WARNING,
        "Data Anomalies Detected",
        f"Found {anomaly_count} anomalies in the data.",
        {
            "anomaly_count": anomaly_count,
            "types": ", ".join(unique_types) if unique_types else "unknown"
        }
    )


def alert_bias_detected(biased_slices: list, severity_counts: dict):
    """Send bias detection notification."""
    sev = AlertSeverity.WARNING
    if severity_counts.get("high", 0) > 0:
        sev = AlertSeverity.ERROR
    
    send_alert(
        AlertType.BIAS_DETECTED,
        sev,
        "Data Bias Detected",
        f"Bias detected in {len(biased_slices)} data slice(s).",
        {
            "biased_slices": ", ".join(biased_slices[:5]),
            "high_severity": severity_counts.get("high", 0),
            "medium_severity": severity_counts.get("medium", 0),
            "low_severity": severity_counts.get("low", 0),
        }
    )


if __name__ == "__main__":
    # Test alerts
    print("Testing alert system...")
    
    send_alert(
        AlertType.PIPELINE_START,
        AlertSeverity.INFO,
        "Test Alert",
        "This is a test alert from the AutoMend pipeline.",
        {"test_key": "test_value"}
    )
    
    print("Alert test complete. Check logs/alerts_history.json")
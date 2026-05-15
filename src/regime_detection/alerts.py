from __future__ import annotations

import json
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Any

import pandas as pd

from .io import append_table


def emit_operational_alerts(
    dashboard: pd.DataFrame,
    quality_results: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    alerts = build_operational_alerts(dashboard, quality_results, config)
    alert_cfg = config.get("alerts", {})
    if alerts.empty:
        if not alert_cfg.get("record_heartbeat", True):
            return alerts
        alerts = pd.DataFrame(
            [
                _alert_row(
                    pd.Timestamp.now(tz="UTC"),
                    "pipeline_heartbeat",
                    "info",
                    "pipeline",
                    "pipeline completed with no active operational alerts",
                )
            ]
        )

    if not alert_cfg.get("enabled", False):
        append_table(alerts, config["tables"].get("regime_operational_alerts", "regime_operational_alerts"), config)
        return alerts

    message = _format_alert_message(alerts)
    errors: list[str] = []
    for url in alert_cfg.get("webhook_urls", []) or []:
        try:
            _send_webhook(str(url), message)
        except Exception as exc:
            errors.append(f"webhook:{exc}")

    email_cfg = alert_cfg.get("email", {})
    if email_cfg.get("enabled", False):
        try:
            _send_email(email_cfg, message)
        except Exception as exc:
            errors.append(f"email:{exc}")

    if errors and alert_cfg.get("fail_on_error", False):
        raise RuntimeError(f"Alert delivery failed: {errors}")
    alerts = alerts.copy()
    alerts["delivered"] = not errors
    append_table(alerts, config["tables"].get("regime_operational_alerts", "regime_operational_alerts"), config)
    return alerts


def build_operational_alerts(
    dashboard: pd.DataFrame,
    quality_results: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    checked_at = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []
    alert_cfg = config.get("alerts", {})
    warning_levels = set(alert_cfg.get("warning_alert_levels", ["orange", "red"]))
    min_hazard = float(alert_cfg.get("min_hazard_probability", 0.5))

    if dashboard is not None and not dashboard.empty:
        for _, row in dashboard.iterrows():
            strategy = row.get("strategy_name", "")
            level = str(row.get("current_alert_level", "")).lower()
            hazard = float(row.get("hazard_of_regime_change", 0.0) or 0.0)
            if level in warning_levels:
                rows.append(
                    _alert_row(
                        checked_at,
                        "regime_alert",
                        "critical" if level == "red" else "warning",
                        strategy,
                        f"strategy={strategy} alert_level={level}",
                    )
                )
            if hazard >= min_hazard:
                rows.append(
                    _alert_row(
                        checked_at,
                        "hazard_alert",
                        "warning",
                        strategy,
                        f"strategy={strategy} hazard_of_regime_change={hazard:.4f}",
                    )
                )

    if quality_results is not None and not quality_results.empty and "status" in quality_results.columns:
        failures = quality_results.loc[quality_results["status"].astype(str).str.lower().eq("fail")]
        for _, row in failures.tail(int(alert_cfg.get("max_quality_failures_in_alert", 25))).iterrows():
            rows.append(
                _alert_row(
                    checked_at,
                    "data_quality_failure",
                    str(row.get("severity", "error")),
                    str(row.get("logical_table", "")),
                    f"{row.get('logical_table', '')}.{row.get('check_name', '')}: {row.get('details', '')}",
                )
            )

    return pd.DataFrame(rows).drop_duplicates(["alert_type", "entity", "message"]) if rows else pd.DataFrame()


def _alert_row(
    checked_at: pd.Timestamp,
    alert_type: str,
    severity: str,
    entity: object,
    message: str,
) -> dict[str, object]:
    return {
        "created_at": checked_at,
        "alert_type": alert_type,
        "severity": severity,
        "entity": entity,
        "message": message,
        "delivered": False,
    }


def _format_alert_message(alerts: pd.DataFrame) -> str:
    lines = ["Regime detection pipeline alerts:"]
    for _, row in alerts.iterrows():
        lines.append(f"- [{row.get('severity')}] {row.get('alert_type')}: {row.get('message')}")
    return "\n".join(lines)


def _send_webhook(url: str, message: str) -> None:
    payload = json.dumps({"text": message}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10):
        pass


def _send_email(email_cfg: dict[str, Any], message: str) -> None:
    smtp_host = str(email_cfg.get("smtp_host", ""))
    if not smtp_host:
        return
    smtp_port = int(email_cfg.get("smtp_port", 587))
    sender = str(email_cfg.get("from", ""))
    recipients = [str(value) for value in email_cfg.get("to", []) or []]
    if not sender or not recipients:
        return

    email = EmailMessage()
    email["Subject"] = str(email_cfg.get("subject", "Regime detection pipeline alert"))
    email["From"] = sender
    email["To"] = ", ".join(recipients)
    email.set_content(message)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as client:
        if email_cfg.get("starttls", True):
            client.starttls()
        username = email_cfg.get("username")
        password = email_cfg.get("password")
        if username and password:
            client.login(str(username), str(password))
        client.send_message(email)

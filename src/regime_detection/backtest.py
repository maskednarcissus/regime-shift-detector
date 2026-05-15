from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MANUAL_REGIME_RANGES, RISK_MULTIPLIERS


def run_backtest(signals: pd.DataFrame, model_outputs: pd.DataFrame, model_version: str, feature_version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = signals.merge(model_outputs, on=["date", "strategy_name"], how="left")
    df["risk_multiplier"] = df["regime_alert_level"].map(RISK_MULTIPLIERS).fillna(1.0)
    df["regime_adjusted_return"] = df["actual_return"] * df["risk_multiplier"]
    df["regime_adjusted_equity"] = df.groupby("strategy_name")["regime_adjusted_return"].transform(
        lambda s: 100.0 * np.exp(s.fillna(0.0).cumsum())
    )

    metrics = []
    for strategy_name, group in df.groupby("strategy_name"):
        metrics.extend(_performance_rows(group, strategy_name, "original", "actual_return", model_version, feature_version))
        metrics.extend(
            _performance_rows(group, strategy_name, "regime_adjusted", "regime_adjusted_return", model_version, feature_version)
        )
        metrics.extend(_overlay_improvement_rows(group, strategy_name, model_version, feature_version))
        metrics.extend(_detection_classification_rows(group, strategy_name, model_version, feature_version))
        metrics.extend(_prediction_lead_time_rows(group, strategy_name, model_version, feature_version))
        metrics.extend(_shock_detection_rows(group, strategy_name, model_version, feature_version))
    return df, pd.DataFrame(metrics)


def _performance_rows(df: pd.DataFrame, strategy: str, test_name: str, return_col: str, model_version: str, feature_version: str) -> list[dict[str, object]]:
    returns = df[return_col].fillna(0.0)
    annual_return = float(np.exp(returns.mean() * 252) - 1.0)
    annual_vol = float(returns.std() * np.sqrt(252))
    equity = np.exp(returns.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    sharpe = annual_return / annual_vol if annual_vol else 0.0
    downside = float(returns[returns < 0.0].std() * np.sqrt(252))
    sortino = annual_return / downside if downside and not np.isnan(downside) else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown else 0.0
    tail_loss = float(returns.quantile(0.05))
    expected_shortfall = float(returns[returns <= tail_loss].mean()) if (returns <= tail_loss).any() else tail_loss
    recovery_time = _max_recovery_time(drawdown)
    return [
        _metric(strategy, test_name, "annualized_return", annual_return, model_version, feature_version),
        _metric(strategy, test_name, "annualized_volatility", annual_vol, model_version, feature_version),
        _metric(strategy, test_name, "sharpe_ratio", sharpe, model_version, feature_version),
        _metric(strategy, test_name, "sortino_ratio", sortino, model_version, feature_version),
        _metric(strategy, test_name, "maximum_drawdown", max_drawdown, model_version, feature_version),
        _metric(strategy, test_name, "calmar_ratio", calmar, model_version, feature_version),
        _metric(strategy, test_name, "tail_loss_5pct", tail_loss, model_version, feature_version),
        _metric(strategy, test_name, "expected_shortfall_5pct", expected_shortfall, model_version, feature_version),
        _metric(strategy, test_name, "max_recovery_time_days", recovery_time, model_version, feature_version),
    ]


def _overlay_improvement_rows(df: pd.DataFrame, strategy: str, model_version: str, feature_version: str) -> list[dict[str, object]]:
    original = _performance_summary(df["actual_return"].fillna(0.0))
    adjusted = _performance_summary(df["regime_adjusted_return"].fillna(0.0))
    return [
        _metric(strategy, "overlay_comparison", "sharpe_improvement", adjusted["sharpe"] - original["sharpe"], model_version, feature_version),
        _metric(strategy, "overlay_comparison", "sortino_improvement", adjusted["sortino"] - original["sortino"], model_version, feature_version),
        _metric(
            strategy,
            "overlay_comparison",
            "max_drawdown_reduction",
            abs(original["max_drawdown"]) - abs(adjusted["max_drawdown"]),
            model_version,
            feature_version,
        ),
        _metric(strategy, "overlay_comparison", "calmar_improvement", adjusted["calmar"] - original["calmar"], model_version, feature_version),
    ]


def _detection_classification_rows(df: pd.DataFrame, strategy: str, model_version: str, feature_version: str) -> list[dict[str, object]]:
    dates = pd.to_datetime(df["date"])
    y_true = pd.Series(False, index=df.index)
    for start, end, _ in MANUAL_REGIME_RANGES:
        y_true |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    y_pred = df["regime_alert_level"].isin(["orange", "red"])

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0

    return [
        _metric(strategy, "manual_regime_detection", "true_positives", tp, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "false_positives", fp, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "true_negatives", tn, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "false_negatives", fn, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "precision", precision, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "recall", recall, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "f1_score", f1, model_version, feature_version),
        _metric(strategy, "manual_regime_detection", "false_alarm_rate", false_alarm_rate, model_version, feature_version),
    ]


def _shock_detection_rows(df: pd.DataFrame, strategy: str, model_version: str, feature_version: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(df["date"])
    for start, end, label in MANUAL_REGIME_RANGES:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        alerts = df.loc[mask & df["regime_alert_level"].isin(["orange", "red"])]
        delay = None
        if not alerts.empty:
            delay = int((pd.to_datetime(alerts["date"].iloc[0]) - pd.Timestamp(start)).days)
        rows.append(_metric(strategy, label, "detection_delay_days", delay if delay is not None else -1, model_version, feature_version))
        rows.append(_metric(strategy, label, "detected", int(delay is not None), model_version, feature_version))
    return rows


def _prediction_lead_time_rows(df: pd.DataFrame, strategy: str, model_version: str, feature_version: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(df["date"])
    for start, _, label in MANUAL_REGIME_RANGES:
        start_date = pd.Timestamp(start)
        for horizon in (30, 60, 90):
            prob_col = f"prob_regime_change_{horizon}d"
            if prob_col not in df.columns:
                continue
            window_mask = (dates >= start_date - pd.Timedelta(days=horizon)) & (dates < start_date)
            window = df.loc[window_mask, ["date", prob_col]].copy()
            if window.empty:
                rows.extend(_empty_prediction_window_rows(strategy, label, horizon, model_version, feature_version))
                continue

            window[prob_col] = pd.to_numeric(window[prob_col], errors="coerce").fillna(0.0)
            threshold_hits = window.loc[window[prob_col] >= 0.5]
            lead_time = -1
            first_signal_probability = 0.0
            if not threshold_hits.empty:
                first_signal_date = pd.to_datetime(threshold_hits["date"].iloc[0])
                lead_time = int((start_date - first_signal_date).days)
                first_signal_probability = float(threshold_hits[prob_col].iloc[0])

            rows.extend(
                [
                    _metric(strategy, label, f"prediction_detected_{horizon}d", int(lead_time >= 0), model_version, feature_version),
                    _metric(strategy, label, f"prediction_lead_time_days_{horizon}d", lead_time, model_version, feature_version),
                    _metric(strategy, label, f"prediction_first_signal_probability_{horizon}d", first_signal_probability, model_version, feature_version),
                    _metric(strategy, label, f"prediction_max_probability_{horizon}d", float(window[prob_col].max()), model_version, feature_version),
                    _metric(strategy, label, f"prediction_mean_probability_{horizon}d", float(window[prob_col].mean()), model_version, feature_version),
                ]
            )
    return rows


def _empty_prediction_window_rows(
    strategy: str,
    label: str,
    horizon: int,
    model_version: str,
    feature_version: str,
) -> list[dict[str, object]]:
    return [
        _metric(strategy, label, f"prediction_detected_{horizon}d", 0, model_version, feature_version),
        _metric(strategy, label, f"prediction_lead_time_days_{horizon}d", -1, model_version, feature_version),
        _metric(strategy, label, f"prediction_first_signal_probability_{horizon}d", 0.0, model_version, feature_version),
        _metric(strategy, label, f"prediction_max_probability_{horizon}d", 0.0, model_version, feature_version),
        _metric(strategy, label, f"prediction_mean_probability_{horizon}d", 0.0, model_version, feature_version),
    ]


def _performance_summary(returns: pd.Series) -> dict[str, float]:
    annual_return = float(np.exp(returns.mean() * 252) - 1.0)
    annual_vol = float(returns.std() * np.sqrt(252))
    equity = np.exp(returns.cumsum())
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    downside = float(returns[returns < 0.0].std() * np.sqrt(252))
    sharpe = annual_return / annual_vol if annual_vol else 0.0
    sortino = annual_return / downside if downside and not np.isnan(downside) else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown, "calmar": calmar}


def _max_recovery_time(drawdown: pd.Series) -> int:
    max_days = 0
    current = 0
    for value in drawdown:
        if value < 0.0:
            current += 1
            max_days = max(max_days, current)
        else:
            current = 0
    return max_days


def _metric(strategy: str, test_name: str, name: str, value: float | int, model_version: str, feature_version: str) -> dict[str, object]:
    return {
        "strategy_name": strategy,
        "test_name": test_name,
        "start_date": None,
        "end_date": None,
        "model_version": model_version,
        "feature_version": feature_version,
        "metric_name": name,
        "metric_value": value,
        "notes": "mvp",
    }

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .io import append_table


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "bronze_strategy_equity": ["date", "strategy_name", "actual_equity", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_macro": ["date", "column_name", "value", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_fx": ["date", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_rates": ["date", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_commodities": ["date", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_market_risk": ["date", "ingestion_ts", "source_file", "source_system"],
    "bronze_factset_news": ["date", "ingestion_ts", "source_file", "source_system"],
    "silver_strategy_returns": ["date", "strategy_name", "actual_equity", "actual_return"],
    "silver_brazil_macro": ["date", "inflation_pressure"],
    "silver_fx_rates": ["date", "brl_return"],
    "silver_interest_rates": ["date", "selic_change"],
    "silver_commodities": ["date", "commodity_return"],
    "silver_market_risk": ["date", "cds_z"],
    "silver_news_signals": ["date"],
    "gold_regime_features": ["date", "strategy_name", "actual_equity", "actual_return", "rolling_sharpe_60d", "rolling_drawdown"],
    "gold_normalized_equity_features": ["date", "strategy_name", "expected_return", "normal_equity", "gap", "gap_pct", "residual_return"],
    "normal_model_feature_importance": ["strategy_name", "normal_model_name", "feature_name", "coefficient", "normalized_abs_importance"],
    "normal_model_validation": ["strategy_name", "normal_model_name", "split", "metric_name", "metric_value"],
    "regime_detection_signals": ["date", "strategy_name", "gap_z", "cusum_signal", "change_point_signal", "hmm_regime", "hmm_regime_method", "regime_alert_level"],
    "regime_prediction_training_set": ["date", "strategy_name", "regime_break_today", "regime_change_next_30d", "regime_change_next_60d", "regime_change_next_90d"],
    "regime_model_outputs": ["date", "strategy_name", "prob_regime_change_30d", "prob_regime_change_60d", "prob_regime_change_90d", "hazard_of_regime_change", "prob_regime_survival_30d", "prob_regime_survival_60d", "prob_regime_survival_90d"],
    "regime_prediction_metrics": ["model", "metric_name", "metric_value"],
    "regime_model_feature_importance": ["model", "target", "feature_name", "importance_type", "importance_value", "normalized_importance"],
    "regime_prediction_diagnostics": ["artifact_type", "model", "target", "bucket", "row_count", "model_version", "feature_version"],
    "regime_backtest_results": ["strategy_name", "test_name", "metric_name", "metric_value"],
    "regime_backtest_timeseries": ["date", "strategy_name", "actual_return", "regime_adjusted_return", "regime_adjusted_equity"],
    "regime_backtest_performance_metrics": ["strategy_name", "test_name", "metric_name", "metric_value"],
    "regime_dashboard_current_status": ["date", "strategy_name", "current_alert_level", "model_version"],
    "regime_operational_alerts": ["created_at", "alert_type", "severity", "entity", "message"],
}

KEY_COLUMNS: dict[str, list[str]] = {
    "bronze_strategy_equity": ["date", "strategy_name"],
    "silver_strategy_returns": ["date", "strategy_name"],
    "silver_brazil_macro": ["date"],
    "silver_fx_rates": ["date"],
    "silver_interest_rates": ["date"],
    "silver_commodities": ["date"],
    "silver_market_risk": ["date"],
    "silver_news_signals": ["date"],
    "gold_regime_features": ["date", "strategy_name"],
    "gold_normalized_equity_features": ["date", "strategy_name"],
    "regime_detection_signals": ["date", "strategy_name"],
    "regime_prediction_training_set": ["date", "strategy_name"],
    "regime_model_outputs": ["date", "strategy_name"],
    "regime_backtest_timeseries": ["date", "strategy_name"],
    "regime_dashboard_current_status": ["strategy_name"],
}


def check_and_record_table(df: pd.DataFrame, logical_table: str, physical_table: str, stage: str, config: dict[str, Any]) -> None:
    quality_cfg = config.get("quality", {})
    if not quality_cfg.get("enabled", True):
        return

    results = validate_table(df, logical_table, physical_table, stage, config)
    append_table(results, config["tables"].get("data_quality_results", "data_quality_results"), config)

    if quality_cfg.get("fail_on_error", False) and (results["status"] == "fail").any():
        failures = results.loc[results["status"] == "fail", ["check_name", "details"]].to_dict("records")
        raise ValueError(f"Data quality failed for {logical_table}: {failures}")


def validate_table(df: pd.DataFrame, logical_table: str, physical_table: str, stage: str, config: dict[str, Any]) -> pd.DataFrame:
    checked_at = pd.Timestamp.utcnow()
    row_count = int(len(df))
    rows: list[dict[str, object]] = []
    required = REQUIRED_COLUMNS.get(logical_table, [])

    rows.append(_result(checked_at, stage, logical_table, physical_table, "row_count_positive", "fail" if row_count == 0 else "pass", row_count, "> 0", row_count, "table must not be empty"))

    missing_columns = [column for column in required if column not in df.columns]
    rows.append(
        _result(
            checked_at,
            stage,
            logical_table,
            physical_table,
            "required_columns_present",
            "fail" if missing_columns else "pass",
            len(missing_columns),
            "0",
            row_count,
            f"missing={missing_columns}",
        )
    )

    if row_count > 0:
        rows.extend(_required_null_checks(df, required, checked_at, stage, logical_table, physical_table, config))
        rows.extend(_duplicate_key_check(df, logical_table, checked_at, stage, physical_table, config))
        rows.extend(_date_checks(df, checked_at, stage, logical_table, physical_table))
        rows.extend(_numeric_finite_checks(df, checked_at, stage, logical_table, physical_table))

    return pd.DataFrame(rows)


def _required_null_checks(
    df: pd.DataFrame,
    required: list[str],
    checked_at: pd.Timestamp,
    stage: str,
    logical_table: str,
    physical_table: str,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    rows = []
    threshold = float(config.get("quality", {}).get("max_required_null_fraction", 0.0))
    for column in required:
        if column not in df.columns:
            continue
        null_fraction = float(df[column].isna().mean())
        rows.append(
            _result(
                checked_at,
                stage,
                logical_table,
                physical_table,
                f"required_not_null:{column}",
                "fail" if null_fraction > threshold else "pass",
                null_fraction,
                f"<= {threshold}",
                len(df),
                "required column null fraction",
            )
        )
    return rows


def _duplicate_key_check(
    df: pd.DataFrame,
    logical_table: str,
    checked_at: pd.Timestamp,
    stage: str,
    physical_table: str,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    keys = KEY_COLUMNS.get(logical_table, [])
    if not keys or any(key not in df.columns for key in keys):
        return []
    threshold = float(config.get("quality", {}).get("max_duplicate_key_fraction", 0.0))
    duplicate_fraction = float(df.duplicated(keys).mean())
    return [
        _result(
            checked_at,
            stage,
            logical_table,
            physical_table,
            "duplicate_key_fraction",
            "fail" if duplicate_fraction > threshold else "pass",
            duplicate_fraction,
            f"<= {threshold}",
            len(df),
            f"keys={keys}",
        )
    ]


def _date_checks(
    df: pd.DataFrame,
    checked_at: pd.Timestamp,
    stage: str,
    logical_table: str,
    physical_table: str,
) -> list[dict[str, object]]:
    if "date" not in df.columns:
        return []
    parsed = pd.to_datetime(df["date"], errors="coerce")
    parse_fail_fraction = float(parsed.isna().mean())
    rows = [
        _result(
            checked_at,
            stage,
            logical_table,
            physical_table,
            "date_parseable",
            "fail" if parse_fail_fraction > 0.0 else "pass",
            parse_fail_fraction,
            "0",
            len(df),
            "date column must parse",
        )
    ]
    if "strategy_name" in df.columns:
        non_monotonic_groups = 0
        for _, group in df.assign(_date=parsed).dropna(subset=["_date"]).groupby("strategy_name"):
            if not group["_date"].is_monotonic_increasing:
                non_monotonic_groups += 1
        rows.append(
            _result(
                checked_at,
                stage,
                logical_table,
                physical_table,
                "date_monotonic_by_strategy",
                "fail" if non_monotonic_groups else "pass",
                non_monotonic_groups,
                "0",
                len(df),
                "dates should be sorted within strategy",
            )
        )
    return rows


def _numeric_finite_checks(
    df: pd.DataFrame,
    checked_at: pd.Timestamp,
    stage: str,
    logical_table: str,
    physical_table: str,
) -> list[dict[str, object]]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    non_finite_count = 0
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce")
        non_finite_count += int((~np.isfinite(values.fillna(0.0))).sum())
    return [
        _result(
            checked_at,
            stage,
            logical_table,
            physical_table,
            "numeric_values_finite",
            "fail" if non_finite_count else "pass",
            non_finite_count,
            "0",
            len(df),
            "numeric columns must not contain inf/-inf",
        )
    ]


def _result(
    checked_at: pd.Timestamp,
    stage: str,
    logical_table: str,
    physical_table: str,
    check_name: str,
    status: str,
    observed_value: float | int,
    threshold: str,
    row_count: int,
    details: str,
) -> dict[str, object]:
    severity = "error" if status == "fail" else "info"
    if isinstance(observed_value, float) and (math.isnan(observed_value) or math.isinf(observed_value)):
        observed_value = 0.0
    return {
        "checked_at": checked_at,
        "stage": stage,
        "logical_table": logical_table,
        "physical_table": physical_table,
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "observed_value": observed_value,
        "threshold": threshold,
        "row_count": row_count,
        "details": details,
    }

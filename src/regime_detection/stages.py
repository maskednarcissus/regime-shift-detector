from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alerts import emit_operational_alerts
from .backtest import run_backtest
from .config import load_config
from .dashboard import build_dashboard_current_status
from .detection import run_detection_model
from .features import (
    build_gold_regime_features,
    build_silver_brazil_macro,
    build_silver_commodities,
    build_silver_fx_rates,
    build_silver_interest_rates,
    build_silver_market_risk,
    build_silver_news_signals,
    build_silver_strategy_returns,
)
from .io import load_processed_table, read_csv, save_table
from .labels import add_train_test_split, create_prediction_labels
from .models import build_prediction_diagnostics, build_prediction_feature_importance, train_prediction_models
from .normal_model import build_normal_model_diagnostics, run_macro_factor_normal_model, run_rolling_normal_model
from .observability import frame_summary_metrics, log_stage_to_mlflow, metrics_table_to_dict
from .quality import check_and_record_table


def ingest_data(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    strategy = read_csv(config["data"]["strategy_equity"])
    strategy = _with_audit_columns(strategy, config["data"]["strategy_equity"], "strategy")
    _save_and_check(strategy, config, "bronze_strategy_equity", "ingest_data")
    _ingest_optional_factset_sources(config)


def build_silver_tables(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    try:
        strategy = load_processed_table(_table(config, "bronze_strategy_equity"), config)
    except FileNotFoundError:
        strategy = read_csv(config["data"]["strategy_equity"])
    silver = build_silver_strategy_returns(strategy)
    _save_and_check(silver, config, "silver_strategy_returns", "build_silver_tables")

    try:
        bronze_macro = load_processed_table(_table(config, "bronze_factset_macro"), config)
    except FileNotFoundError:
        bronze_macro = pd.DataFrame()
    macro = build_silver_brazil_macro(bronze_macro)
    if not macro.empty:
        _save_and_check(macro, config, "silver_brazil_macro", "build_silver_tables")

    _build_optional_domain_silver_tables(config)


def build_gold_features(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    silver = load_processed_table(_table(config, "silver_strategy_returns"), config)
    try:
        macro = load_processed_table(_table(config, "silver_brazil_macro"), config)
    except FileNotFoundError:
        macro = pd.DataFrame()
    feature_tables = _load_optional_silver_feature_tables(config)
    gold = build_gold_regime_features(silver, macro, feature_tables)
    _save_and_check(gold, config, "gold_regime_features", "build_gold_features")


def run_normal_condition_model(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    gold = load_processed_table(_table(config, "gold_regime_features"), config)
    model_type = str(config["models"].get("normal_model_type", "rolling")).lower()
    diagnostics = build_normal_model_diagnostics(
        gold,
        split_date=config["models"]["train_test_split_date"],
        alpha=float(config["models"].get("normal_ridge_alpha", 1.0)),
        rolling_window=int(config["models"]["normal_window"]),
    )
    importance, validation = diagnostics
    if not importance.empty:
        _save_and_check(importance, config, "normal_model_feature_importance", "run_normal_condition_model")
    if not validation.empty:
        _save_and_check(validation, config, "normal_model_validation", "run_normal_condition_model")

    if model_type == "macro_factor_ridge":
        normalized = run_macro_factor_normal_model(
            gold,
            split_date=config["models"]["train_test_split_date"],
            alpha=float(config["models"].get("normal_ridge_alpha", 1.0)),
            fallback_window=int(config["models"]["normal_window"]),
        )
    else:
        normalized = run_rolling_normal_model(gold, window=int(config["models"]["normal_window"]))
        normalized["normal_model_name"] = "rolling_expected_return"
    _save_and_check(normalized, config, "gold_normalized_equity_features", "run_normal_condition_model")
    log_stage_to_mlflow(
        config,
        "normal_condition_model",
        metrics={
            **frame_summary_metrics(normalized, "normalized"),
            **metrics_table_to_dict(validation, "normal_validation"),
        },
        artifacts={
            "normal_model_validation": validation,
            "normal_model_feature_importance": importance,
        },
    )


def run_detection_stage(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    normalized = load_processed_table(_table(config, "gold_normalized_equity_features"), config)
    signals = run_detection_model(
        normalized,
        gap_window=int(config["models"]["gap_window"]),
        cusum_threshold=float(config["models"]["cusum_threshold"]),
        change_point_window=int(config["models"].get("change_point_window", 30)),
        change_point_z_threshold=float(config["models"].get("change_point_z_threshold", 3.0)),
        change_point_method=str(config["models"].get("change_point_method", "pelt")),
        change_point_lookback=int(config["models"].get("change_point_lookback", 180)),
        change_point_min_size=int(config["models"].get("change_point_min_size", 20)),
        change_point_penalty=_optional_float(config["models"].get("change_point_penalty")),
        change_point_confirmation_days=int(config["models"].get("change_point_confirmation_days", 5)),
        change_point_cooldown_days=int(config["models"].get("change_point_cooldown_days", 20)),
        regime_state_count=int(config["models"].get("regime_state_count", 4)),
        regime_state_model=str(config["models"].get("regime_state_model", "hmm")),
        hmm_max_iter=int(config["models"].get("hmm_max_iter", 25)),
        hmm_transition_prior=float(config["models"].get("hmm_transition_prior", 2.0)),
    )
    _save_and_check(signals, config, "regime_detection_signals", "run_detection_stage")
    log_stage_to_mlflow(
        config,
        "detection",
        metrics=_detection_summary_metrics(signals),
        artifacts={"regime_detection_signals_tail": signals.tail(500)},
    )


def create_prediction_training_set(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    signals = load_processed_table(_table(config, "regime_detection_signals"), config)
    training = create_prediction_labels(signals)
    training = add_train_test_split(training, config["models"]["train_test_split_date"])
    _save_and_check(training, config, "regime_prediction_training_set", "create_prediction_training_set")


def train_prediction_stage(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    training = load_processed_table(_table(config, "regime_prediction_training_set"), config)
    outputs, metrics = train_prediction_models(
        training,
        model_version=config["models"]["model_version"],
        feature_version=config["models"]["feature_version"],
    )
    _save_and_check(outputs, config, "regime_model_outputs", "train_prediction_stage")
    if not metrics.empty:
        _save_and_check(metrics, config, "regime_prediction_metrics", "train_prediction_stage")
    importance = build_prediction_feature_importance(
        training,
        model_version=config["models"]["model_version"],
        feature_version=config["models"]["feature_version"],
    )
    if not importance.empty:
        _save_and_check(importance, config, "regime_model_feature_importance", "train_prediction_stage")
    diagnostics = build_prediction_diagnostics(
        training,
        outputs,
        model_version=config["models"]["model_version"],
        feature_version=config["models"]["feature_version"],
    )
    if not diagnostics.empty:
        _save_and_check(diagnostics, config, "regime_prediction_diagnostics", "train_prediction_stage")
    log_stage_to_mlflow(
        config,
        "prediction",
        metrics={
            **frame_summary_metrics(outputs, "prediction_outputs"),
            **metrics_table_to_dict(metrics, "prediction"),
        },
        artifacts={
            "regime_prediction_metrics": metrics,
            "regime_model_feature_importance": importance,
            "regime_prediction_diagnostics": diagnostics,
            "regime_model_outputs_tail": outputs.tail(500),
        },
    )


def run_backtest_stage(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    signals = load_processed_table(_table(config, "regime_detection_signals"), config)
    outputs = load_processed_table(_table(config, "regime_model_outputs"), config)
    backtest_frame, metrics = run_backtest(
        signals,
        outputs,
        model_version=config["models"]["model_version"],
        feature_version=config["models"]["feature_version"],
    )
    _save_and_check(backtest_frame, config, "regime_backtest_timeseries", "run_backtest_stage")
    _save_and_check(metrics, config, "regime_backtest_results", "run_backtest_stage")
    _save_and_check(metrics, config, "regime_backtest_performance_metrics", "run_backtest_stage")
    log_stage_to_mlflow(
        config,
        "backtest",
        metrics={
            **frame_summary_metrics(backtest_frame, "backtest_timeseries"),
            **metrics_table_to_dict(metrics, "backtest"),
        },
        artifacts={
            "regime_backtest_results": metrics,
            "regime_backtest_timeseries_tail": backtest_frame.tail(500),
        },
    )


def build_dashboard_outputs(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    backtest_frame = load_processed_table(_table(config, "regime_backtest_timeseries"), config)
    dashboard = build_dashboard_current_status(backtest_frame, model_version=config["models"]["model_version"])
    _save_and_check(dashboard, config, "regime_dashboard_current_status", "build_dashboard_outputs")
    log_stage_to_mlflow(
        config,
        "dashboard",
        metrics=frame_summary_metrics(dashboard, "dashboard"),
        artifacts={"regime_dashboard_current_status": dashboard},
    )


def emit_operational_alerts_stage(config_path: str = "config/pipeline.yml") -> None:
    config = load_config(config_path)
    dashboard = load_processed_table(_table(config, "regime_dashboard_current_status"), config)
    try:
        quality = load_processed_table(_table(config, "data_quality_results"), config)
    except FileNotFoundError:
        quality = pd.DataFrame()
    alerts = emit_operational_alerts(dashboard, quality, config)
    if not alerts.empty:
        check_and_record_table(alerts, "regime_operational_alerts", _table(config, "regime_operational_alerts"), "emit_operational_alerts", config)
    log_stage_to_mlflow(
        config,
        "operational_alerts",
        metrics={"alert_count": int(len(alerts))},
        artifacts={"regime_operational_alerts": alerts},
    )


def _table(config: dict, key: str) -> str:
    return config.get("tables", {}).get(key, key)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _save_and_check(df: pd.DataFrame, config: dict, table_key: str, stage: str) -> None:
    physical_table = _table(config, table_key)
    save_table(df, physical_table, config)
    check_and_record_table(df, table_key, physical_table, stage, config)


def _detection_summary_metrics(signals: pd.DataFrame) -> dict[str, float | int]:
    metrics: dict[str, float | int] = frame_summary_metrics(signals, "detection")
    if "change_point_signal" in signals.columns:
        metrics["change_point_signal_count"] = int(pd.to_numeric(signals["change_point_signal"], errors="coerce").fillna(0).sum())
    if "regime_alert_level" in signals.columns:
        for level, count in signals["regime_alert_level"].fillna("unknown").value_counts().items():
            metrics[f"alert_level_{level}_count"] = int(count)
    if "hmm_regime_method" in signals.columns:
        metrics["hmm_method_count"] = int(signals["hmm_regime_method"].nunique())
    return metrics


def _build_optional_domain_silver_tables(config: dict) -> None:
    builders = (
        ("bronze_factset_fx", "silver_fx_rates", build_silver_fx_rates),
        ("bronze_factset_rates", "silver_interest_rates", build_silver_interest_rates),
        ("bronze_factset_commodities", "silver_commodities", build_silver_commodities),
        ("bronze_factset_market_risk", "silver_market_risk", build_silver_market_risk),
        ("bronze_factset_news", "silver_news_signals", build_silver_news_signals),
    )
    for bronze_key, silver_key, builder in builders:
        try:
            bronze = load_processed_table(_table(config, bronze_key), config)
        except FileNotFoundError:
            continue
        silver = builder(bronze)
        if not silver.empty:
            _save_and_check(silver, config, silver_key, "build_silver_tables")


def _load_optional_silver_feature_tables(config: dict) -> list[pd.DataFrame]:
    tables = []
    for key in (
        "silver_fx_rates",
        "silver_interest_rates",
        "silver_commodities",
        "silver_market_risk",
        "silver_news_signals",
    ):
        try:
            tables.append(load_processed_table(_table(config, key), config))
        except FileNotFoundError:
            continue
    return tables


def _ingest_optional_factset_sources(config: dict) -> None:
    source_map = {
        "factset_fx": "bronze_factset_fx",
        "factset_rates": "bronze_factset_rates",
        "factset_commodities": "bronze_factset_commodities",
        "factset_market_risk": "bronze_factset_market_risk",
        "factset_news": "bronze_factset_news",
    }
    for data_key, table_key in source_map.items():
        path = config["data"].get(data_key)
        df = _read_optional(path)
        if not df.empty:
            _save_and_check(_with_audit_columns(df, path, data_key), config, table_key, "ingest_data")

    macro = _read_macro_source(config)
    if not macro.empty:
        _save_and_check(_with_audit_columns(macro, _macro_source_path(config), "factset_macro"), config, "bronze_factset_macro", "ingest_data")


def _read_macro_source(config: dict) -> pd.DataFrame:
    long_path = config["data"].get("factset_macro_long")
    long_df = _read_optional(long_path)
    if not long_df.empty:
        return long_df

    wide_path = config["data"].get("factset_macro_wide")
    wide_df = _read_optional(wide_path)
    if wide_df.empty:
        return pd.DataFrame()
    return _wide_macro_to_long(wide_df)


def _macro_source_path(config: dict) -> str:
    long_path = str(config["data"].get("factset_macro_long", ""))
    if Path(long_path).exists():
        return long_path
    return str(config["data"].get("factset_macro_wide", ""))


def _wide_macro_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in wide_df.columns:
        return pd.DataFrame()
    value_columns = [column for column in wide_df.columns if column != "date"]
    return wide_df.melt(
        id_vars=["date"],
        value_vars=value_columns,
        var_name="column_name",
        value_name="value",
    )


def _read_optional(path: str | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return read_csv(path, required=False)


def _with_audit_columns(df: pd.DataFrame, source_file: str | None, source_system: str) -> pd.DataFrame:
    out = df.copy()
    out["ingestion_ts"] = pd.Timestamp.utcnow()
    out["source_file"] = source_file or ""
    out["source_system"] = source_system
    return out

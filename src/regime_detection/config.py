from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "strategy_equity": "data/raw/strategy_equity.csv",
        "factset_macro_wide": "brazil_macro_monthly_wide.csv",
        "factset_macro_long": "brazil_macro_monthly_long.csv",
        "factset_fx": "data/raw/factset_fx.csv",
        "factset_rates": "data/raw/factset_rates.csv",
        "factset_commodities": "data/raw/factset_commodities.csv",
        "factset_market_risk": "data/raw/factset_market_risk.csv",
        "factset_news": "data/raw/factset_news.csv",
    },
    "tables": {
        "bronze_strategy_equity": "bronze_strategy_equity",
        "bronze_factset_macro": "bronze_factset_macro",
        "bronze_factset_fx": "bronze_factset_fx",
        "bronze_factset_rates": "bronze_factset_rates",
        "bronze_factset_commodities": "bronze_factset_commodities",
        "bronze_factset_market_risk": "bronze_factset_market_risk",
        "bronze_factset_news": "bronze_factset_news",
        "silver_strategy_returns": "silver_strategy_returns",
        "silver_brazil_macro": "silver_brazil_macro",
        "silver_fx_rates": "silver_fx_rates",
        "silver_interest_rates": "silver_interest_rates",
        "silver_commodities": "silver_commodities",
        "silver_market_risk": "silver_market_risk",
        "silver_news_signals": "silver_news_signals",
        "gold_regime_features": "gold_regime_features",
        "gold_normalized_equity_features": "gold_normalized_equity_features",
        "normal_model_feature_importance": "normal_model_feature_importance",
        "normal_model_validation": "normal_model_validation",
        "regime_detection_signals": "regime_detection_signals",
        "regime_prediction_training_set": "regime_prediction_training_set",
        "regime_model_outputs": "regime_model_outputs",
        "regime_prediction_metrics": "regime_prediction_metrics",
        "regime_model_feature_importance": "regime_model_feature_importance",
        "regime_prediction_diagnostics": "regime_prediction_diagnostics",
        "regime_backtest_results": "regime_backtest_results",
        "regime_backtest_timeseries": "regime_backtest_timeseries",
        "regime_dashboard_current_status": "regime_dashboard_current_status",
        "regime_backtest_performance_metrics": "regime_backtest_performance_metrics",
        "regime_operational_alerts": "regime_operational_alerts",
        "data_quality_results": "data_quality_results",
    },
    "quality": {
        "enabled": True,
        "fail_on_error": False,
        "max_required_null_fraction": 0.0,
        "max_duplicate_key_fraction": 0.0,
    },
    "mlflow": {
        "enabled": True,
        "fail_on_error": False,
        "tracking_uri": None,
        "experiment_name": "/Shared/regime_detection",
    },
    "alerts": {
        "enabled": False,
        "fail_on_error": False,
        "warning_alert_levels": ["orange", "red"],
        "min_hazard_probability": 0.5,
        "max_quality_failures_in_alert": 25,
        "record_heartbeat": True,
        "webhook_urls": [],
        "email": {
            "enabled": False,
            "smtp_host": "",
            "smtp_port": 587,
            "starttls": True,
            "from": "",
            "to": [],
            "username": "",
            "password": "",
            "subject": "Regime detection pipeline alert",
        },
    },
    "dashboard": {
        "sql_warehouse_id": "",
        "dashboard_id": "",
        "definition_path": "dashboards/regime_detection_dashboard.sql",
    },
    "models": {
        "normal_window": 252,
        "gap_window": 60,
        "cusum_threshold": 0.08,
        "normal_model_type": "rolling",
        "normal_ridge_alpha": 1.0,
        "change_point_method": "pelt",
        "change_point_window": 30,
        "change_point_z_threshold": 3.0,
        "change_point_lookback": 180,
        "change_point_min_size": 20,
        "change_point_penalty": None,
        "change_point_confirmation_days": 5,
        "change_point_cooldown_days": 20,
        "regime_state_count": 4,
        "regime_state_model": "hmm",
        "hmm_max_iter": 25,
        "hmm_transition_prior": 2.0,
        "train_test_split_date": "2022-01-01",
        "model_version": "mvp-0.1",
        "feature_version": "mvp-0.1",
    },
}


def load_config(path: str | Path = "config/pipeline.yml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(deepcopy(DEFAULT_CONFIG), loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base

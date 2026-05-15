from __future__ import annotations

import pandas as pd


def build_dashboard_current_status(backtest_frame: pd.DataFrame, model_version: str) -> pd.DataFrame:
    rows = []
    for strategy_name, group in backtest_frame.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        latest = group.iloc[-1]
        rows.append(
            {
                "date": latest["date"],
                "strategy_name": strategy_name,
                "actual_equity": latest["actual_equity"],
                "normal_equity": latest["normal_equity"],
                "gap_pct": latest["gap_pct"],
                "gap_z": latest["gap_z"],
                "rolling_sharpe_60d": latest["rolling_sharpe_60d"],
                "rolling_drawdown": latest["rolling_drawdown"],
                "current_alert_level": latest["regime_alert_level"],
                "change_point_score": latest.get("change_point_score", 0.0),
                "change_point_signal": latest.get("change_point_signal", 0),
                "change_point_method": latest.get("change_point_method", ""),
                "change_point_date": latest.get("change_point_date", None),
                "prob_regime_change_30d": latest.get("prob_regime_change_30d", 0.0),
                "prob_regime_change_60d": latest.get("prob_regime_change_60d", 0.0),
                "prob_regime_change_90d": latest.get("prob_regime_change_90d", 0.0),
                "prob_regime_survival_30d": latest.get("prob_regime_survival_30d", 1.0),
                "prob_regime_survival_60d": latest.get("prob_regime_survival_60d", 1.0),
                "prob_regime_survival_90d": latest.get("prob_regime_survival_90d", 1.0),
                "survival_model_method": latest.get("survival_model_method", ""),
                "hazard_of_regime_change": latest.get("hazard_of_regime_change", 0.0),
                "regime_change_60d_signal": latest.get("regime_change_60d_signal", 0),
                "high_precision_regime_change_60d_signal": latest.get("high_precision_regime_change_60d_signal", 0),
                "prediction_decision_threshold": latest.get("prediction_decision_threshold", 0.5),
                "high_precision_threshold": latest.get("high_precision_threshold", 0.8),
                "dominant_stress_driver": _dominant_driver(latest),
                "recommended_risk_multiplier": latest["risk_multiplier"],
                "model_version": model_version,
            }
        )
    return pd.DataFrame(rows)


def _dominant_driver(row: pd.Series) -> str:
    candidates = {
        "gap_z": abs(float(row.get("gap_z", 0.0))),
        "drawdown": abs(float(row.get("rolling_drawdown", 0.0))) * 10,
        "volatility": abs(float(row.get("rolling_vol_60d", 0.0))) * 100,
        "cds": abs(float(row.get("cds_z", 0.0))),
        "commodities": abs(float(row.get("commodity_z", 0.0))),
    }
    return max(candidates, key=candidates.get)

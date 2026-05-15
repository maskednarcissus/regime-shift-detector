from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def log_stage_to_mlflow(
    config: dict[str, Any],
    stage: str,
    metrics: dict[str, float | int | str | None] | None = None,
    artifacts: dict[str, pd.DataFrame] | None = None,
) -> None:
    mlflow_cfg = config.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return

    try:
        import mlflow
    except Exception:
        return

    try:
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))
        experiment_name = mlflow_cfg.get("experiment_name")
        if experiment_name:
            mlflow.set_experiment(str(experiment_name))

        with mlflow.start_run(run_name=f"{stage}_{config['models'].get('model_version', 'unknown')}", nested=True):
            mlflow.set_tags(
                {
                    "stage": stage,
                    "model_version": str(config["models"].get("model_version", "")),
                    "feature_version": str(config["models"].get("feature_version", "")),
                }
            )
            for key, value in (metrics or {}).items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    mlflow.log_metric(_safe_metric_name(key), float(value))
                elif value is not None:
                    mlflow.set_tag(_safe_metric_name(key), str(value))

            with tempfile.TemporaryDirectory() as temp_dir:
                for name, frame in (artifacts or {}).items():
                    if frame is None or frame.empty:
                        continue
                    path = Path(temp_dir) / f"{_safe_metric_name(name)}.csv"
                    frame.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path=stage)
    except Exception:
        if mlflow_cfg.get("fail_on_error", False):
            raise


def metrics_table_to_dict(metrics: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    if metrics.empty or not {"metric_name", "metric_value"}.issubset(metrics.columns):
        return {}
    out: dict[str, float] = {}
    model_col = "model" if "model" in metrics.columns else "test_name" if "test_name" in metrics.columns else None
    for _, row in metrics.iterrows():
        value = row.get("metric_value")
        if not isinstance(value, (int, float)) or pd.isna(value):
            continue
        parts = [prefix] if prefix else []
        if model_col:
            parts.append(str(row.get(model_col, "all")))
        parts.append(str(row["metric_name"]))
        out[_safe_metric_name("_".join(parts))] = float(value)
    return out


def frame_summary_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {f"{prefix}_rows": int(len(frame)), f"{prefix}_columns": int(len(frame.columns))}
    numeric = frame.select_dtypes(include="number")
    for column in numeric.columns[:25]:
        series = pd.to_numeric(numeric[column], errors="coerce")
        if series.notna().any():
            metrics[f"{prefix}_{column}_mean"] = float(series.mean())
            metrics[f"{prefix}_{column}_last"] = float(series.iloc[-1])
    return metrics


def _safe_metric_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(name))[:250]

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metric_rows(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    model_name: str,
    thresholds: Iterable[float] = (0.5,),
) -> list[dict[str, object]]:
    y, probability = sanitize_binary_prediction_inputs(y_true, y_probability)
    rows: list[dict[str, object]] = []
    if y.nunique() == 2:
        rows.append({"model": model_name, "metric_name": "auc_roc", "metric_value": roc_auc_score(y, probability)})
        rows.append({"model": model_name, "metric_name": "pr_auc", "metric_value": average_precision_score(y, probability)})

    rows.append({"model": model_name, "metric_name": "brier_score", "metric_value": brier_score_loss(y, probability)})
    rows.append({"model": model_name, "metric_name": "calibration_error_5bin", "metric_value": calibration_error(y, probability)})

    for threshold in normalized_thresholds(thresholds):
        label = threshold_label(threshold)
        prediction = (probability >= threshold).astype(int)
        counts = confusion_counts(y, prediction)
        rows.extend(
            [
                {"model": model_name, "metric_name": f"accuracy_at_{label}", "metric_value": accuracy_score(y, prediction)},
                {"model": model_name, "metric_name": f"precision_at_{label}", "metric_value": precision_score(y, prediction, zero_division=0)},
                {"model": model_name, "metric_name": f"recall_at_{label}", "metric_value": recall_score(y, prediction, zero_division=0)},
                {"model": model_name, "metric_name": f"f1_at_{label}", "metric_value": f1_score(y, prediction, zero_division=0)},
                {"model": model_name, "metric_name": f"false_positive_rate_at_{label}", "metric_value": _false_positive_rate(counts)},
            ]
        )
    return rows


def confusion_diagnostic_rows(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    model_name: str,
    target: str,
    threshold: float,
    model_version: str,
    feature_version: str,
) -> list[dict[str, object]]:
    y, probability = sanitize_binary_prediction_inputs(y_true, y_probability)
    counts = confusion_counts(y, (probability >= threshold).astype(int))
    return [
        diagnostic_row(
            artifact_type="confusion_matrix",
            model_name=model_name,
            target=target,
            bucket=metric_name,
            observed_rate=np.nan,
            predicted_probability=np.nan,
            row_count=value,
            model_version=model_version,
            feature_version=feature_version,
            threshold=threshold,
        )
        for metric_name, value in counts.items()
    ]


def calibration_diagnostic_rows(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    model_name: str,
    target: str,
    model_version: str,
    feature_version: str,
    bins: int = 5,
) -> list[dict[str, object]]:
    y, probability = sanitize_binary_prediction_inputs(y_true, y_probability)
    frame = pd.DataFrame({"y": y, "p": probability})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows: list[dict[str, object]] = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append(
            diagnostic_row(
                artifact_type="calibration_curve",
                model_name=model_name,
                target=target,
                bucket=str(interval),
                observed_rate=float(group["y"].mean()) if len(group) else 0.0,
                predicted_probability=float(group["p"].mean()) if len(group) else 0.0,
                row_count=int(len(group)),
                model_version=model_version,
                feature_version=feature_version,
            )
        )
    return rows


def diagnostic_row(
    artifact_type: str,
    model_name: str,
    target: str,
    bucket: str,
    observed_rate: float,
    predicted_probability: float,
    row_count: int,
    model_version: str,
    feature_version: str,
    threshold: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "artifact_type": artifact_type,
        "model": model_name,
        "target": target,
        "bucket": bucket,
        "observed_rate": observed_rate,
        "predicted_probability": predicted_probability,
        "row_count": row_count,
        "model_version": model_version,
        "feature_version": feature_version,
    }
    if threshold is not None:
        row["threshold"] = threshold
    return row


def confusion_counts(y_true: pd.Series, y_predicted: pd.Series | np.ndarray) -> dict[str, int]:
    y = pd.Series(y_true).fillna(0).astype(int)
    prediction = pd.Series(np.asarray(y_predicted), index=y.index).fillna(0).astype(int)
    return {
        "true_positive": int(((y == 1) & (prediction == 1)).sum()),
        "false_positive": int(((y == 0) & (prediction == 1)).sum()),
        "true_negative": int(((y == 0) & (prediction == 0)).sum()),
        "false_negative": int(((y == 1) & (prediction == 0)).sum()),
    }


def calibration_error(y_true: pd.Series, y_probability: pd.Series | np.ndarray, bins: int = 5) -> float:
    y, probability = sanitize_binary_prediction_inputs(y_true, y_probability)
    frame = pd.DataFrame({"y": y, "p": probability})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    total = len(frame)
    if total == 0:
        return 0.0

    error = 0.0
    for _, group in frame.groupby("bin", observed=True):
        error += len(group) / total * abs(float(group["y"].mean()) - float(group["p"].mean()))
    return float(error)


def sanitize_binary_prediction_inputs(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    y = pd.Series(y_true).fillna(0).astype(int)
    probability = pd.Series(np.asarray(y_probability), index=y.index)
    probability = pd.to_numeric(probability, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return y, probability.clip(0.0, 1.0)


def normalized_thresholds(thresholds: Iterable[float]) -> list[float]:
    clean_values = []
    for value in thresholds:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= threshold <= 1.0:
            clean_values.append(threshold)
    clean = sorted(set(clean_values))
    return clean or [0.5]


def threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100.0))}pct"


def _false_positive_rate(counts: dict[str, int]) -> float:
    false_positive = counts["false_positive"]
    true_negative = counts["true_negative"]
    return false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0

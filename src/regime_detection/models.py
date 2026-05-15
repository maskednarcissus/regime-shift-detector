from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import PREDICTION_FEATURES


def train_prediction_models(training_set: pd.DataFrame, model_version: str, feature_version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = training_set.copy()
    for column in PREDICTION_FEATURES:
        if column not in df.columns:
            df[column] = 0.0
    df[PREDICTION_FEATURES] = df[PREDICTION_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    outputs = df[["date", "strategy_name"]].copy()
    metrics: list[dict[str, object]] = []

    for horizon in (30, 60, 90):
        target = f"regime_change_next_{horizon}d"
        prob_col = f"prob_regime_change_{horizon}d"
        outputs[prob_col] = _fit_predict_probability(df, target, model="gbm")
        metrics.extend(_classification_metrics(df[target], outputs[prob_col], f"gbm_{horizon}d"))

    outputs["prob_regime_change_60d_logistic"] = _fit_predict_probability(df, "regime_change_next_60d", model="logistic")
    outputs["prob_regime_change_60d_gbm"] = outputs["prob_regime_change_60d"]
    logistic_hazard = _fit_predict_probability(df, "regime_break_today", model="logistic")
    outputs["hazard_of_regime_change_logistic"] = logistic_hazard
    survival_outputs, survival_metrics = _fit_predict_cox_survival(df, logistic_hazard)
    for column, values in survival_outputs.items():
        outputs[column] = values
    metrics.extend(_classification_metrics(df["regime_break_today"], outputs["hazard_of_regime_change"], "hazard_break_today"))
    metrics.extend(survival_metrics)
    outputs["predicted_alert_level"] = pd.cut(
        outputs["prob_regime_change_60d"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["green", "yellow", "orange", "red"],
    ).astype(str)
    outputs["model_version"] = model_version
    outputs["feature_version"] = feature_version
    return outputs, pd.DataFrame(metrics)


def build_prediction_feature_importance(
    training_set: pd.DataFrame,
    model_version: str,
    feature_version: str,
) -> pd.DataFrame:
    df = training_set.copy()
    for column in PREDICTION_FEATURES:
        if column not in df.columns:
            df[column] = 0.0
    df[PREDICTION_FEATURES] = df[PREDICTION_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_mask = df.get("train_test_split", "train").eq("train")
    if train_mask.sum() < 20:
        train_mask = pd.Series(True, index=df.index)

    rows: list[dict[str, object]] = []
    targets = ["regime_break_today", "regime_change_next_30d", "regime_change_next_60d", "regime_change_next_90d"]
    for target in targets:
        if target not in df.columns:
            continue
        y = df.loc[train_mask, target].fillna(0).astype(int)
        x = df.loc[train_mask, PREDICTION_FEATURES]
        rows.extend(_univariate_importance_rows(x, y, target, model_version, feature_version))
        rows.extend(_logistic_importance_rows(x, y, target, model_version, feature_version))
    return pd.DataFrame(rows)


def build_prediction_diagnostics(
    training_set: pd.DataFrame,
    model_outputs: pd.DataFrame,
    model_version: str,
    feature_version: str,
) -> pd.DataFrame:
    df = training_set.merge(model_outputs, on=["date", "strategy_name"], how="left")
    targets = {
        "gbm_30d": ("regime_change_next_30d", "prob_regime_change_30d"),
        "gbm_60d": ("regime_change_next_60d", "prob_regime_change_60d"),
        "gbm_90d": ("regime_change_next_90d", "prob_regime_change_90d"),
        "hazard_break_today": ("regime_break_today", "hazard_of_regime_change"),
        "cox_survival_30d": ("regime_change_next_30d", "prob_regime_change_30d_survival"),
        "cox_survival_60d": ("regime_change_next_60d", "prob_regime_change_60d_survival"),
        "cox_survival_90d": ("regime_change_next_90d", "prob_regime_change_90d_survival"),
    }
    rows: list[dict[str, object]] = []
    for model_name, (target, probability) in targets.items():
        if target not in df.columns or probability not in df.columns:
            continue
        y_true = df[target].fillna(0).astype(int)
        y_prob = pd.to_numeric(df[probability], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        rows.extend(_confusion_diagnostic_rows(model_name, target, y_true, y_prob, model_version, feature_version))
        rows.extend(_calibration_diagnostic_rows(model_name, target, y_true, y_prob, model_version, feature_version))
    return pd.DataFrame(rows)


def _fit_predict_probability(df: pd.DataFrame, target: str, model: str) -> np.ndarray:
    y = df[target].fillna(0).astype(int)
    if y.nunique() < 2:
        return np.full(len(df), float(y.iloc[0]) if len(y) else 0.0)

    train_mask = df.get("train_test_split", "train").eq("train")
    if train_mask.sum() < 20 or y[train_mask].nunique() < 2:
        train_mask = pd.Series(True, index=df.index)

    if model == "logistic":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        estimator = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=7)

    estimator.fit(df.loc[train_mask, PREDICTION_FEATURES], y.loc[train_mask])
    return estimator.predict_proba(df[PREDICTION_FEATURES])[:, 1]


def _confusion_diagnostic_rows(
    model_name: str,
    target: str,
    y_true: pd.Series,
    y_prob: pd.Series,
    model_version: str,
    feature_version: str,
) -> list[dict[str, object]]:
    y_pred = (y_prob >= 0.5).astype(int)
    values = {
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    return [
        _diagnostic_row(
            artifact_type="confusion_matrix",
            model_name=model_name,
            target=target,
            bucket=metric_name,
            observed_rate=np.nan,
            predicted_probability=np.nan,
            row_count=value,
            model_version=model_version,
            feature_version=feature_version,
        )
        for metric_name, value in values.items()
    ]


def _calibration_diagnostic_rows(
    model_name: str,
    target: str,
    y_true: pd.Series,
    y_prob: pd.Series,
    model_version: str,
    feature_version: str,
    bins: int = 5,
) -> list[dict[str, object]]:
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows: list[dict[str, object]] = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append(
            _diagnostic_row(
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


def _diagnostic_row(
    artifact_type: str,
    model_name: str,
    target: str,
    bucket: str,
    observed_rate: float,
    predicted_probability: float,
    row_count: int,
    model_version: str,
    feature_version: str,
) -> dict[str, object]:
    return {
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


def _univariate_importance_rows(
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    model_version: str,
    feature_version: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_scores: dict[str, float] = {}
    for feature_name in PREDICTION_FEATURES:
        feature = pd.to_numeric(x[feature_name], errors="coerce").fillna(0.0)
        if feature.std() == 0.0 or y.nunique() < 2:
            auc = 0.5
            signed_correlation = 0.0
        else:
            auc = float(roc_auc_score(y, feature))
            signed_correlation = float(feature.corr(y))
            if not np.isfinite(signed_correlation):
                signed_correlation = 0.0
        raw_scores[feature_name] = abs(auc - 0.5) * 2.0
        rows.append(
            _importance_row(
                target=target,
                model_name="univariate_screen",
                feature_name=feature_name,
                importance_type="auc_lift",
                importance_value=raw_scores[feature_name],
                normalized_importance=0.0,
                model_version=model_version,
                feature_version=feature_version,
                extra={"auc": auc, "signed_correlation": signed_correlation},
            )
        )

    total = sum(raw_scores.values())
    if total > 0.0:
        for row in rows:
            row["normalized_importance"] = float(raw_scores[str(row["feature_name"])] / total)
    return rows


def _logistic_importance_rows(
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    model_version: str,
    feature_version: str,
) -> list[dict[str, object]]:
    if y.nunique() < 2 or len(y) < 20:
        return []
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    try:
        estimator.fit(x, y)
    except Exception:
        return []

    coefficients = estimator.named_steps["logisticregression"].coef_[0]
    abs_sum = float(np.abs(coefficients).sum())
    rows = []
    for feature_name, coefficient in zip(PREDICTION_FEATURES, coefficients):
        importance = float(abs(coefficient))
        rows.append(
            _importance_row(
                target=target,
                model_name="logistic_regression",
                feature_name=feature_name,
                importance_type="abs_standardized_coefficient",
                importance_value=importance,
                normalized_importance=float(importance / abs_sum) if abs_sum else 0.0,
                model_version=model_version,
                feature_version=feature_version,
                extra={"coefficient": float(coefficient)},
            )
        )
    return rows


def _importance_row(
    target: str,
    model_name: str,
    feature_name: str,
    importance_type: str,
    importance_value: float,
    normalized_importance: float,
    model_version: str,
    feature_version: str,
    extra: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model_name,
        "target": target,
        "feature_name": feature_name,
        "importance_type": importance_type,
        "importance_value": float(importance_value) if np.isfinite(importance_value) else 0.0,
        "normalized_importance": float(normalized_importance) if np.isfinite(normalized_importance) else 0.0,
        "model_version": model_version,
        "feature_version": feature_version,
    }
    row.update(extra)
    return row


def _fit_predict_cox_survival(df: pd.DataFrame, fallback_hazard: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    duration, event = _time_to_next_regime_break(df)
    train_mask = df.get("train_test_split", "train").eq("train").to_numpy()
    if train_mask.sum() < 30 or event[train_mask].sum() < 2:
        return _logistic_survival_fallback(fallback_hazard), [
            {"model": "cox_survival", "metric_name": "fit_method_code", "metric_value": 0},
            {"model": "cox_survival", "metric_name": "training_event_rows", "metric_value": int(event[train_mask].sum())},
        ]

    x = df[PREDICTION_FEATURES].to_numpy(dtype=float)
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    x_scaled = (x - mean) / std

    beta = _fit_cox_coefficients(x_scaled[train_mask], duration[train_mask], event[train_mask])
    if beta is None:
        return _logistic_survival_fallback(fallback_hazard), [
            {"model": "cox_survival", "metric_name": "fit_method_code", "metric_value": 0},
            {"model": "cox_survival", "metric_name": "training_event_rows", "metric_value": int(event[train_mask].sum())},
        ]

    train_risk = np.exp(np.clip(x_scaled[train_mask] @ beta, -20.0, 20.0))
    baseline_durations, baseline_cumulative_hazard = _breslow_baseline_hazard(
        duration[train_mask],
        event[train_mask],
        train_risk,
    )
    risk = np.exp(np.clip(x_scaled @ beta, -20.0, 20.0))
    outputs: dict[str, object] = {
        "survival_model_method": "cox_proportional_hazards",
        "cox_relative_risk": risk,
    }
    for horizon in (30, 60, 90):
        baseline_hazard = _baseline_cumulative_hazard_at(baseline_durations, baseline_cumulative_hazard, horizon)
        survival = np.exp(-baseline_hazard * risk)
        outputs[f"prob_regime_survival_{horizon}d"] = np.clip(survival, 0.0, 1.0)
        outputs[f"prob_regime_change_{horizon}d_survival"] = np.clip(1.0 - survival, 0.0, 1.0)

    daily_baseline = _baseline_cumulative_hazard_at(baseline_durations, baseline_cumulative_hazard, 30) / 30.0
    outputs["cox_hazard_of_regime_change"] = np.clip(1.0 - np.exp(-daily_baseline * risk), 0.0, 1.0)
    outputs["hazard_of_regime_change"] = outputs["cox_hazard_of_regime_change"]

    metrics: list[dict[str, object]] = [
        {"model": "cox_survival", "metric_name": "fit_method_code", "metric_value": 1},
        {"model": "cox_survival", "metric_name": "training_event_rows", "metric_value": int(event[train_mask].sum())},
        {"model": "cox_survival", "metric_name": "concordance_index", "metric_value": _concordance_index(duration, event, risk)},
        {"model": "cox_survival", "metric_name": "coefficient_l1", "metric_value": float(np.abs(beta).sum())},
    ]
    for horizon in (30, 60, 90):
        metrics.extend(
            _classification_metrics(
                df[f"regime_change_next_{horizon}d"],
                outputs[f"prob_regime_change_{horizon}d_survival"],
                f"cox_survival_{horizon}d",
            )
        )
    return outputs, metrics


def _logistic_survival_fallback(fallback_hazard: np.ndarray) -> dict[str, object]:
    hazard = np.clip(np.asarray(fallback_hazard, dtype=float), 0.0, 1.0)
    outputs: dict[str, object] = {
        "hazard_of_regime_change": hazard,
        "cox_hazard_of_regime_change": np.full(len(hazard), np.nan),
        "cox_relative_risk": np.full(len(hazard), np.nan),
        "survival_model_method": "logistic_fallback",
    }
    for horizon in (30, 60, 90):
        survival = np.power(1.0 - hazard, horizon)
        outputs[f"prob_regime_survival_{horizon}d"] = np.clip(survival, 0.0, 1.0)
        outputs[f"prob_regime_change_{horizon}d_survival"] = np.clip(1.0 - survival, 0.0, 1.0)
    return outputs


def _time_to_next_regime_break(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    duration = np.ones(len(df), dtype=float)
    event = np.zeros(len(df), dtype=int)
    position_by_index = {index_value: position for position, index_value in enumerate(df.index)}
    for _, group in df.sort_values(["strategy_name", "date"]).groupby("strategy_name", sort=False):
        positions = np.asarray([position_by_index[index_value] for index_value in group.index], dtype=int)
        breaks = group["regime_break_today"].fillna(0).astype(int).to_numpy()
        next_event_position: int | None = None
        for local_idx in range(len(group) - 1, -1, -1):
            if breaks[local_idx] == 1:
                next_event_position = local_idx
            output_idx = int(positions[local_idx])
            if next_event_position is None:
                duration[output_idx] = float(len(group) - local_idx)
                event[output_idx] = 0
            else:
                duration[output_idx] = float(max(next_event_position - local_idx, 1))
                event[output_idx] = 1
    return duration, event


def _fit_cox_coefficients(x: np.ndarray, duration: np.ndarray, event: np.ndarray) -> np.ndarray | None:
    l2_penalty = 1e-3
    initial = np.zeros(x.shape[1], dtype=float)
    try:
        from scipy.optimize import minimize

        result = minimize(
            lambda beta: _cox_loss_gradient(beta, x, duration, event, l2_penalty),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 250, "ftol": 1e-8},
        )
        if result.success and np.isfinite(result.x).all():
            return result.x
    except Exception:
        pass

    beta = initial.copy()
    step = 0.05
    previous_loss = np.inf
    for _ in range(500):
        loss, gradient = _cox_loss_gradient(beta, x, duration, event, l2_penalty)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            return None
        candidate = beta - step * gradient
        candidate_loss, _ = _cox_loss_gradient(candidate, x, duration, event, l2_penalty)
        if candidate_loss <= loss:
            beta = candidate
            if abs(previous_loss - candidate_loss) < 1e-7:
                break
            previous_loss = candidate_loss
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    return beta if np.isfinite(beta).all() else None


def _cox_loss_gradient(
    beta: np.ndarray,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    order = np.argsort(-duration)
    sorted_duration = duration[order]
    sorted_event = event[order].astype(bool)
    sorted_x = x[order]
    eta = np.clip(sorted_x @ beta, -40.0, 40.0)
    exp_eta = np.exp(eta)
    cumulative_risk = np.cumsum(exp_eta)
    cumulative_x_risk = np.cumsum(exp_eta[:, None] * sorted_x, axis=0)
    risk_end = _risk_set_end_positions(sorted_duration)
    event_positions = np.flatnonzero(sorted_event)
    if len(event_positions) == 0:
        return 0.0, np.zeros_like(beta)

    denominators = cumulative_risk[risk_end[event_positions]].clip(min=1e-12)
    expected_x = cumulative_x_risk[risk_end[event_positions]] / denominators[:, None]
    log_likelihood = float((eta[event_positions] - np.log(denominators)).sum())
    gradient = (sorted_x[event_positions] - expected_x).sum(axis=0)
    event_count = float(len(event_positions))
    loss = -log_likelihood / event_count + 0.5 * l2_penalty * float(beta @ beta)
    loss_gradient = -gradient / event_count + l2_penalty * beta
    return loss, loss_gradient


def _risk_set_end_positions(sorted_duration_desc: np.ndarray) -> np.ndarray:
    ends = np.empty(len(sorted_duration_desc), dtype=int)
    start = 0
    while start < len(sorted_duration_desc):
        end = start + 1
        while end < len(sorted_duration_desc) and sorted_duration_desc[end] == sorted_duration_desc[start]:
            end += 1
        ends[start:end] = end - 1
        start = end
    return ends


def _breslow_baseline_hazard(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event_durations = np.sort(np.unique(duration[event.astype(bool)]))
    cumulative = 0.0
    cumulative_hazards: list[float] = []
    for value in event_durations:
        events_at_duration = (duration == value) & event.astype(bool)
        risk_at_duration = duration >= value
        denominator = float(risk[risk_at_duration].sum())
        if denominator <= 0.0:
            cumulative_hazards.append(cumulative)
            continue
        cumulative += float(events_at_duration.sum()) / denominator
        cumulative_hazards.append(cumulative)
    return event_durations, np.asarray(cumulative_hazards, dtype=float)


def _baseline_cumulative_hazard_at(durations: np.ndarray, cumulative_hazard: np.ndarray, horizon: int) -> float:
    if len(durations) == 0:
        return 0.0
    position = np.searchsorted(durations, horizon, side="right") - 1
    if position < 0:
        return 0.0
    return float(cumulative_hazard[position])


def _concordance_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    permissible = 0
    concordant = 0.0
    event_positions = np.flatnonzero(event.astype(bool))
    for idx in event_positions:
        comparable = duration > duration[idx]
        if not comparable.any():
            continue
        risk_delta = risk[idx] - risk[comparable]
        permissible += int(comparable.sum())
        concordant += float((risk_delta > 0.0).sum())
        concordant += 0.5 * float((risk_delta == 0.0).sum())
    if permissible == 0:
        return 0.0
    return float(concordant / permissible)


def _classification_metrics(y_true: pd.Series, y_prob: pd.Series, name: str) -> list[dict[str, object]]:
    y = y_true.fillna(0).astype(int)
    y_hat = (pd.Series(y_prob, index=y.index) >= 0.5).astype(int)
    rows: list[dict[str, object]] = []
    if y.nunique() == 2:
        rows.append({"model": name, "metric_name": "auc_roc", "metric_value": roc_auc_score(y, y_prob)})
        rows.append({"model": name, "metric_name": "pr_auc", "metric_value": average_precision_score(y, y_prob)})
        rows.append({"model": name, "metric_name": "precision_at_50pct", "metric_value": precision_score(y, y_hat, zero_division=0)})
        rows.append({"model": name, "metric_name": "recall_at_50pct", "metric_value": recall_score(y, y_hat, zero_division=0)})
        rows.append({"model": name, "metric_name": "f1_at_50pct", "metric_value": f1_score(y, y_hat, zero_division=0)})
    rows.append({"model": name, "metric_name": "brier_score", "metric_value": brier_score_loss(y, y_prob)})
    rows.append({"model": name, "metric_name": "calibration_error_5bin", "metric_value": _calibration_error(y, y_prob)})
    return rows


def _calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 5) -> float:
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=True)
    total = len(frame)
    if total == 0:
        return 0.0
    error = 0.0
    for _, group in grouped:
        error += len(group) / total * abs(float(group["y"].mean()) - float(group["p"].mean()))
    return float(error)

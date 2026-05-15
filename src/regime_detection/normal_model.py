from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import NORMAL_MODEL_FEATURES


def run_rolling_normal_model(features: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, group in features.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        df = group.copy()
        df["expected_return"] = df["actual_return"].shift(1).rolling(window, min_periods=20).mean().fillna(0.0)
        first_equity = float(df["actual_equity"].iloc[0])
        df["normal_equity"] = first_equity * np.exp(df["expected_return"].cumsum())
        df["gap"] = df["actual_equity"] - df["normal_equity"]
        df["gap_pct"] = df["gap"] / df["normal_equity"].replace(0.0, np.nan)
        df["residual_return"] = df["actual_return"] - df["expected_return"]
        outputs.append(df)
    return pd.concat(outputs, ignore_index=True)


def run_macro_factor_normal_model(
    features: pd.DataFrame,
    split_date: str,
    alpha: float = 1.0,
    fallback_window: int = 252,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, group in features.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        df = group.copy()
        expected = _macro_factor_expected_return(df, split_date=split_date, alpha=alpha)
        fallback = df["actual_return"].shift(1).rolling(fallback_window, min_periods=20).mean().fillna(0.0)
        df["expected_return"] = expected.fillna(fallback).fillna(0.0)
        df["normal_model_name"] = "macro_factor_ridge"
        first_equity = float(df["actual_equity"].iloc[0])
        df["normal_equity"] = first_equity * np.exp(df["expected_return"].cumsum())
        df["gap"] = df["actual_equity"] - df["normal_equity"]
        df["gap_pct"] = df["gap"] / df["normal_equity"].replace(0.0, np.nan)
        df["residual_return"] = df["actual_return"] - df["expected_return"]
        outputs.append(df)
    return pd.concat(outputs, ignore_index=True)


def build_normal_model_diagnostics(
    features: pd.DataFrame,
    split_date: str,
    alpha: float = 1.0,
    rolling_window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    importance_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for strategy_name, group in features.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        df = group.copy()
        y = pd.to_numeric(df["actual_return"], errors="coerce").fillna(0.0)
        dates = pd.to_datetime(df["date"], errors="coerce")
        rolling_expected = y.shift(1).rolling(rolling_window, min_periods=20).mean().fillna(0.0)
        validation_rows.extend(_validation_rows(strategy_name, "rolling_expected_return", y, rolling_expected, dates, split_date))

        fit = _fit_macro_factor_estimator(df, split_date=split_date, alpha=alpha)
        if fit is None:
            continue

        estimator, x, y_fit, _, train_mask = fit
        ridge_expected = pd.Series(estimator.predict(x), index=df.index)
        validation_rows.extend(_validation_rows(strategy_name, "macro_factor_ridge", y_fit, ridge_expected, dates, split_date))

        ridge = estimator.named_steps["ridge"]
        coef_abs_sum = float(np.abs(ridge.coef_).sum())
        for feature_name, coefficient in zip(NORMAL_MODEL_FEATURES, ridge.coef_):
            importance_rows.append(
                {
                    "strategy_name": strategy_name,
                    "normal_model_name": "macro_factor_ridge",
                    "feature_name": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                    "normalized_abs_importance": float(abs(coefficient) / coef_abs_sum) if coef_abs_sum else 0.0,
                    "alpha": alpha,
                    "train_rows": int(train_mask.sum()),
                    "split_date": split_date,
                }
            )

    return pd.DataFrame(importance_rows), pd.DataFrame(validation_rows)


def _macro_factor_expected_return(df: pd.DataFrame, split_date: str, alpha: float) -> pd.Series:
    fit = _fit_macro_factor_estimator(df, split_date=split_date, alpha=alpha)
    if fit is None:
        return pd.Series(np.nan, index=df.index)

    estimator, x, _, _, _ = fit
    return pd.Series(estimator.predict(x), index=df.index)


def _fit_macro_factor_estimator(
    df: pd.DataFrame,
    split_date: str,
    alpha: float,
) -> tuple[object, pd.DataFrame, pd.Series, pd.Series, pd.Series] | None:
    model_frame = df.copy()
    for column in NORMAL_MODEL_FEATURES:
        if column not in model_frame.columns:
            model_frame[column] = 0.0

    # Shift by one row so expected_return[t] only uses factors known before t.
    x = model_frame[NORMAL_MODEL_FEATURES].shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(model_frame["actual_return"], errors="coerce").fillna(0.0)
    dates = pd.to_datetime(model_frame["date"], errors="coerce")
    train_mask = dates < pd.Timestamp(split_date)

    if train_mask.sum() < 30 or y.loc[train_mask].std() == 0.0:
        train_mask = pd.Series(True, index=model_frame.index)
    if train_mask.sum() < 30 or y.loc[train_mask].std() == 0.0:
        return None

    estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    estimator.fit(x.loc[train_mask], y.loc[train_mask])
    return estimator, x, y, dates, train_mask


def _validation_rows(
    strategy_name: str,
    model_name: str,
    actual: pd.Series,
    expected: pd.Series,
    dates: pd.Series,
    split_date: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = {
        "all": pd.Series(True, index=actual.index),
        "train": dates < pd.Timestamp(split_date),
        "test": dates >= pd.Timestamp(split_date),
    }
    residual = actual - expected
    for split_name, mask in masks.items():
        if mask.sum() == 0:
            continue
        actual_split = actual.loc[mask]
        expected_split = expected.loc[mask]
        residual_split = residual.loc[mask]
        rows.extend(
            [
                _validation_metric(strategy_name, model_name, split_name, "rows", int(mask.sum())),
                _validation_metric(strategy_name, model_name, split_name, "rmse", float(np.sqrt(np.mean(residual_split**2)))),
                _validation_metric(strategy_name, model_name, split_name, "mae", float(np.mean(np.abs(residual_split)))),
                _validation_metric(strategy_name, model_name, split_name, "mean_residual", float(residual_split.mean())),
                _validation_metric(
                    strategy_name,
                    model_name,
                    split_name,
                    "directional_accuracy",
                    float((np.sign(actual_split) == np.sign(expected_split)).mean()),
                ),
            ]
        )
        if actual_split.std() != 0.0 and expected_split.std() != 0.0:
            rows.append(
                _validation_metric(
                    strategy_name,
                    model_name,
                    split_name,
                    "correlation",
                    float(actual_split.corr(expected_split)),
                )
            )
    return rows


def _validation_metric(strategy_name: str, model_name: str, split_name: str, metric_name: str, metric_value: float | int) -> dict[str, object]:
    return {
        "strategy_name": strategy_name,
        "normal_model_name": model_name,
        "split": split_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }

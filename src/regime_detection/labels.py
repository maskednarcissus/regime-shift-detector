from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MANUAL_REGIME_RANGES


def create_prediction_labels(signals: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, group in signals.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        df = group.copy()
        df["regime_break_today"] = (
            (df["regime_alert_level"] == "red") | (df["change_point_signal"].fillna(0) == 1)
        ).astype(int)
        for horizon in (30, 60, 90):
            df[f"regime_change_next_{horizon}d"] = _future_window_max(df["regime_break_today"], horizon)
        df["manual_regime_label"] = _manual_labels(df["date"])
        outputs.append(df)
    return pd.concat(outputs, ignore_index=True)


def add_train_test_split(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    out = df.copy()
    out["train_test_split"] = np.where(pd.to_datetime(out["date"]) < pd.Timestamp(split_date), "train", "test")
    return out


def _future_window_max(values: pd.Series, horizon: int) -> pd.Series:
    arr = values.to_numpy()
    result = np.zeros(len(arr), dtype=int)
    for idx in range(len(arr)):
        result[idx] = int(arr[idx + 1 : idx + 1 + horizon].max(initial=0))
    return pd.Series(result, index=values.index)


def _manual_labels(dates: pd.Series) -> pd.Series:
    labels = pd.Series("normal_or_transition", index=dates.index)
    parsed = pd.to_datetime(dates)
    for start, end, label in MANUAL_REGIME_RANGES:
        mask = (parsed >= pd.Timestamp(start)) & (parsed <= pd.Timestamp(end))
        labels.loc[mask] = label
    return labels


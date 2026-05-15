from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .constants import (
    COMMODITY_COLUMN_ALIASES,
    FX_COLUMN_ALIASES,
    MACRO_COLUMN_ALIASES,
    MARKET_RISK_COLUMN_ALIASES,
    NEUTRAL_FEATURE_DEFAULTS,
    NEWS_KEYWORDS,
    NEWS_SCORE_ALIASES,
    RATES_COLUMN_ALIASES,
)


def build_silver_strategy_returns(strategy: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "actual_equity", "strategy_name"}
    missing = required - set(strategy.columns)
    if missing:
        raise ValueError(f"strategy equity is missing columns: {sorted(missing)}")

    df = strategy.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["strategy_name", "date"]).drop_duplicates(["strategy_name", "date"])
    df["actual_equity"] = pd.to_numeric(df["actual_equity"], errors="coerce")
    df["actual_return"] = (
        df.groupby("strategy_name")["actual_equity"]
        .transform(lambda s: np.log(s / s.shift(1)))
        .fillna(0.0)
    )
    return df[["date", "strategy_name", "actual_equity", "actual_return"]]


def build_silver_brazil_macro(bronze_macro: pd.DataFrame) -> pd.DataFrame:
    if bronze_macro.empty:
        return pd.DataFrame()

    wide = _macro_to_wide(bronze_macro)
    if wide.empty:
        return pd.DataFrame()

    out = pd.DataFrame({"date": pd.to_datetime(wide["date"])})
    for target, aliases in MACRO_COLUMN_ALIASES.items():
        out[target] = _first_available_numeric(wide, aliases)
    out["inflation_pressure"] = out[["ipca", "core_inflation", "inflation_expectations"]].mean(axis=1)
    return out.sort_values("date").drop_duplicates("date")


def build_silver_fx_rates(bronze_fx: pd.DataFrame) -> pd.DataFrame:
    df = _build_wide_feature_table(bronze_fx, FX_COLUMN_ALIASES)
    if df.empty:
        return df
    if "brl_return" not in df and "brl_usd" in df:
        df["brl_return"] = _log_return(df["brl_usd"])
    if "brl_volatility_20d" not in df and "brl_return" in df:
        df["brl_volatility_20d"] = df["brl_return"].rolling(20, min_periods=5).std()
    if "brl_volatility_60d" not in df and "brl_return" in df:
        df["brl_volatility_60d"] = df["brl_return"].rolling(60, min_periods=10).std()
    if "dollar_index_return" not in df and "dollar_index" in df:
        df["dollar_index_return"] = _log_return(df["dollar_index"])
    return _finalize_feature_table(df, FX_COLUMN_ALIASES)


def build_silver_interest_rates(bronze_rates: pd.DataFrame) -> pd.DataFrame:
    df = _build_wide_feature_table(bronze_rates, RATES_COLUMN_ALIASES)
    if df.empty:
        return df
    if "selic_change" not in df and "selic" in df:
        df["selic_change"] = df["selic"].diff()
    if "yield_curve_slope" not in df and {"di_5y", "di_1y"}.issubset(df.columns):
        df["yield_curve_slope"] = df["di_5y"] - df["di_1y"]
    if "rate_differential" not in df and {"selic", "us_10y"}.issubset(df.columns):
        df["rate_differential"] = df["selic"] - df["us_10y"]
    return _finalize_feature_table(df, RATES_COLUMN_ALIASES)


def build_silver_commodities(bronze_commodities: pd.DataFrame) -> pd.DataFrame:
    df = _build_wide_feature_table(bronze_commodities, COMMODITY_COLUMN_ALIASES)
    if df.empty:
        return df
    for price_col, return_col in (
        ("oil_price", "oil_return"),
        ("iron_ore_price", "iron_ore_return"),
        ("soybean_price", "soybean_return"),
        ("commodity_basket", "commodity_return"),
    ):
        if return_col not in df and price_col in df:
            df[return_col] = _log_return(df[price_col])
    if "commodity_basket" not in df:
        price_cols = [column for column in ("oil_price", "iron_ore_price", "soybean_price") if column in df]
        if price_cols:
            normalized = df[price_cols].div(df[price_cols].iloc[0].replace(0.0, np.nan))
            df["commodity_basket"] = normalized.mean(axis=1)
            if "commodity_return" not in df:
                df["commodity_return"] = _log_return(df["commodity_basket"])
    if "commodity_z" not in df and "commodity_basket" in df:
        df["commodity_z"] = _rolling_z(df["commodity_basket"], 60)
    return _finalize_feature_table(df, COMMODITY_COLUMN_ALIASES)


def build_silver_market_risk(bronze_market_risk: pd.DataFrame) -> pd.DataFrame:
    df = _build_wide_feature_table(bronze_market_risk, MARKET_RISK_COLUMN_ALIASES)
    if df.empty:
        return df
    if "ibovespa_return" not in df and "ibovespa" in df:
        df["ibovespa_return"] = _log_return(df["ibovespa"])
    if "cds_change" not in df and "brazil_cds" in df:
        df["cds_change"] = df["brazil_cds"].diff()
    if "cds_z" not in df and "brazil_cds" in df:
        df["cds_z"] = _rolling_z(df["brazil_cds"], 60)
    if "global_risk_z" not in df and "vix" in df:
        df["global_risk_z"] = _rolling_z(df["vix"], 60)
    return _finalize_feature_table(df, MARKET_RISK_COLUMN_ALIASES)


def build_silver_news_signals(bronze_news: pd.DataFrame) -> pd.DataFrame:
    if bronze_news.empty or "date" not in bronze_news.columns:
        return pd.DataFrame()

    df = _standardize_date_frame(bronze_news)
    if df.empty:
        return df

    out = pd.DataFrame({"date": df["date"]})
    for target, aliases in NEWS_SCORE_ALIASES.items():
        out[target] = _first_available_numeric(df, aliases)

    text_columns = [column for column in df.columns if _normalize_name(column) in {"headline", "title", "body", "text", "summary"}]
    if text_columns:
        text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        for target, keywords in NEWS_KEYWORDS.items():
            if target not in out or out[target].fillna(0.0).abs().sum() == 0.0:
                out[target] = text.apply(lambda value: sum(keyword in value for keyword in keywords)).astype(float)

    score_columns = [column for column in out.columns if column != "date"]
    return (
        out[["date", *score_columns]]
        .groupby("date", as_index=False)
        .sum(numeric_only=True)
        .sort_values("date")
    )


def build_gold_regime_features(
    strategy_returns: pd.DataFrame,
    brazil_macro: pd.DataFrame | None = None,
    feature_tables: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    df = strategy_returns.copy().sort_values(["strategy_name", "date"])
    if brazil_macro is not None and not brazil_macro.empty:
        df = _merge_feature_table(df, brazil_macro)
    for feature_table in feature_tables or []:
        if feature_table is not None and not feature_table.empty:
            df = _merge_feature_table(df, feature_table)

    for column, default in NEUTRAL_FEATURE_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    grouped = df.groupby("strategy_name", group_keys=False)
    df["rolling_vol_20d"] = grouped["actual_return"].apply(lambda s: s.rolling(20, min_periods=5).std()).fillna(0.0)
    df["rolling_vol_60d"] = grouped["actual_return"].apply(lambda s: s.rolling(60, min_periods=10).std()).fillna(0.0)
    df["rolling_sharpe_20d"] = _rolling_sharpe(grouped, 20)
    df["rolling_sharpe_60d"] = _rolling_sharpe(grouped, 60)
    df["rolling_drawdown"] = grouped["actual_equity"].apply(_drawdown).fillna(0.0)
    df["max_drawdown_60d"] = (
        grouped["rolling_drawdown"].apply(lambda s: s.rolling(60, min_periods=10).min()).fillna(0.0)
    )
    return df


def _rolling_sharpe(grouped: pd.core.groupby.DataFrameGroupBy, window: int) -> pd.Series:
    mean = grouped["actual_return"].apply(lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
    vol = grouped["actual_return"].apply(lambda s: s.rolling(window, min_periods=max(5, window // 4)).std())
    return (mean / vol.replace(0.0, np.nan) * np.sqrt(252)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _build_wide_feature_table(raw: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    if raw.empty or "date" not in raw.columns:
        return pd.DataFrame()
    df = _standardize_date_frame(raw)
    if df.empty:
        return df
    out = pd.DataFrame({"date": df["date"]})
    for target, column_aliases in aliases.items():
        series = _first_available_numeric(df, column_aliases)
        if series.notna().any() and series.fillna(0.0).abs().sum() != 0.0:
            out[target] = series
    return _dedupe_daily(out)


def _finalize_feature_table(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    columns = ["date", *[column for column in aliases if column in df.columns]]
    out = df[columns].copy()
    numeric_columns = [column for column in out.columns if column != "date"]
    out[numeric_columns] = out[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _dedupe_daily(out)


def _standardize_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _dedupe_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("date").groupby("date", as_index=False).last()


def _log_return(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric / numeric.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.rolling(window, min_periods=max(10, window // 4)).mean()
    vol = numeric.rolling(window, min_periods=max(10, window // 4)).std()
    return ((numeric - mean) / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _macro_to_wide(bronze_macro: pd.DataFrame) -> pd.DataFrame:
    df = bronze_macro.copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if {"column_name", "value"}.issubset(df.columns):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return (
            df.pivot_table(index="date", columns="column_name", values="value", aggfunc="last")
            .reset_index()
            .rename_axis(columns=None)
        )

    return df


def _first_available_numeric(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    normalized = {_normalize_name(column): column for column in df.columns}
    for alias in aliases:
        column = normalized.get(_normalize_name(alias))
        if column is not None:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(0.0, index=df.index)


def _merge_feature_table(strategy_returns: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    features = feature_table.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features.dropna(subset=["date"]).sort_values("date")
    feature_columns = [column for column in features.columns if column != "date"]
    if not feature_columns:
        return strategy_returns

    merged_groups = []
    for _, group in strategy_returns.groupby("strategy_name", sort=False):
        left = group.copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        left = left.sort_values("date")
        overlapping = [column for column in feature_columns if column in left.columns]
        if overlapping:
            left = left.drop(columns=overlapping)
        merged = pd.merge_asof(left, features[["date", *feature_columns]], on="date", direction="backward")
        merged_groups.append(merged)
    return pd.concat(merged_groups, ignore_index=True)


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")

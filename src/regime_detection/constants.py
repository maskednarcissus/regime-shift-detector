from __future__ import annotations

NEUTRAL_FEATURE_DEFAULTS: dict[str, float] = {
    "brl_return": 0.0,
    "brl_volatility_20d": 0.0,
    "brl_volatility_60d": 0.0,
    "cds_change": 0.0,
    "cds_z": 0.0,
    "selic_change": 0.0,
    "yield_curve_slope": 0.0,
    "commodity_return": 0.0,
    "commodity_z": 0.0,
    "inflation_pressure": 0.0,
    "political_risk_score": 0.0,
    "pandemic_score": 0.0,
    "supply_chain_score": 0.0,
    "inflation_news_score": 0.0,
    "fiscal_risk_score": 0.0,
    "ibovespa_return": 0.0,
    "global_risk_z": 0.0,
    "dollar_index_return": 0.0,
    "rate_differential": 0.0,
    "gdp_growth": 0.0,
    "industrial_production": 0.0,
    "retail_sales": 0.0,
    "services_activity": 0.0,
    "unemployment": 0.0,
    "ipca": 0.0,
    "core_inflation": 0.0,
    "inflation_expectations": 0.0,
    "fiscal_deficit": 0.0,
    "public_debt": 0.0,
    "current_account": 0.0,
    "consumer_confidence": 0.0,
    "business_confidence": 0.0,
}

MACRO_COLUMN_ALIASES: dict[str, list[str]] = {
    "gdp_growth": ["gdp_growth", "brazilian_gdp", "real_gdp", "gdp_yoy"],
    "industrial_production": ["industrial_production", "industrial_production_yoy", "ip_yoy"],
    "retail_sales": ["retail_sales", "retail_sales_yoy"],
    "services_activity": ["services_activity", "services", "services_yoy"],
    "unemployment": ["unemployment", "unemployment_rate"],
    "ipca": ["ipca", "ipca_inflation", "ipca_yoy"],
    "core_inflation": ["core_inflation", "core_ipca", "core_ipca_yoy"],
    "inflation_expectations": ["inflation_expectations", "inflation_expectation", "ipca_expectations"],
    "fiscal_deficit": ["fiscal_deficit", "primary_deficit", "nominal_deficit"],
    "public_debt": ["public_debt", "gross_debt", "debt_to_gdp"],
    "current_account": ["current_account", "current_account_balance", "current_account_balance_12m"],
    "consumer_confidence": ["consumer_confidence", "consumer_confidence_index"],
    "business_confidence": ["business_confidence", "business_confidence_index"],
}

FX_COLUMN_ALIASES: dict[str, list[str]] = {
    "brl_usd": ["brl_usd", "usd_brl", "usdbRL", "brl", "fx_rate"],
    "brl_return": ["brl_return", "usd_brl_return", "fx_return"],
    "brl_volatility_20d": ["brl_volatility_20d", "fx_volatility_20d", "brl_vol_20d"],
    "brl_volatility_60d": ["brl_volatility_60d", "fx_volatility_60d", "brl_vol_60d"],
    "dollar_index": ["dollar_index", "dxy", "dollar_index_level"],
    "dollar_index_return": ["dollar_index_return", "dxy_return"],
}

RATES_COLUMN_ALIASES: dict[str, list[str]] = {
    "selic": ["selic", "selic_rate", "policy_rate"],
    "selic_change": ["selic_change", "selic_delta", "policy_rate_change"],
    "di_1y": ["di_1y", "di1y", "di_12m"],
    "di_2y": ["di_2y", "di2y", "di_24m"],
    "di_5y": ["di_5y", "di5y", "di_60m"],
    "yield_curve_slope": ["yield_curve_slope", "di_slope", "curve_slope"],
    "us_10y": ["us_10y", "ust_10y", "us_treasury_10y"],
    "rate_differential": ["rate_differential", "br_us_rate_differential"],
}

COMMODITY_COLUMN_ALIASES: dict[str, list[str]] = {
    "oil_price": ["oil_price", "brent", "wti"],
    "oil_return": ["oil_return", "brent_return", "wti_return"],
    "iron_ore_price": ["iron_ore_price", "iron_ore"],
    "iron_ore_return": ["iron_ore_return"],
    "soybean_price": ["soybean_price", "soybeans", "soybean"],
    "soybean_return": ["soybean_return", "soybeans_return"],
    "commodity_basket": ["commodity_basket", "commodity_index", "broad_commodity_index"],
    "commodity_return": ["commodity_return", "commodity_basket_return", "commodity_index_return"],
    "commodity_z": ["commodity_z", "commodity_basket_z", "commodity_index_z"],
}

MARKET_RISK_COLUMN_ALIASES: dict[str, list[str]] = {
    "ibovespa": ["ibovespa", "bovespa", "ibov"],
    "ibovespa_return": ["ibovespa_return", "ibov_return"],
    "brazil_cds": ["brazil_cds", "cds", "brasil_cds"],
    "cds_change": ["cds_change", "brazil_cds_change"],
    "cds_z": ["cds_z", "brazil_cds_z"],
    "vix": ["vix", "global_volatility"],
    "global_risk_z": ["global_risk_z", "vix_z"],
}

NEWS_SCORE_ALIASES: dict[str, list[str]] = {
    "political_risk_score": ["political_risk_score", "political_score"],
    "pandemic_score": ["pandemic_score", "covid_score"],
    "supply_chain_score": ["supply_chain_score", "supply_score"],
    "inflation_news_score": ["inflation_news_score", "inflation_score"],
    "fiscal_risk_score": ["fiscal_risk_score", "fiscal_score"],
    "lava_jato_score": ["lava_jato_score", "lava_jato"],
    "impeachment_score": ["impeachment_score"],
}

NEWS_KEYWORDS: dict[str, list[str]] = {
    "political_risk_score": ["political", "election", "congress", "government", "crisis"],
    "pandemic_score": ["covid", "pandemic", "coronavirus", "lockdown"],
    "supply_chain_score": ["supply chain", "shortage", "logistics", "shipping"],
    "inflation_news_score": ["inflation", "ipca", "prices", "cost pressure"],
    "fiscal_risk_score": ["fiscal", "deficit", "debt", "spending cap"],
    "lava_jato_score": ["lava jato", "car wash", "corruption probe"],
    "impeachment_score": ["impeachment"],
}

PREDICTION_FEATURES: list[str] = [
    "gap_z",
    "gap_pct",
    "residual_return",
    "rolling_sharpe_20d",
    "rolling_sharpe_60d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "rolling_drawdown",
    "change_point_score",
    "change_point_signal",
    "change_point_age_days",
    "brl_volatility_60d",
    "cds_z",
    "selic_change",
    "commodity_z",
    "inflation_pressure",
    "political_risk_score",
    "pandemic_score",
    "supply_chain_score",
]

NORMAL_MODEL_FEATURES: list[str] = [
    "brl_return",
    "brl_volatility_60d",
    "cds_change",
    "cds_z",
    "selic_change",
    "yield_curve_slope",
    "commodity_return",
    "commodity_z",
    "inflation_pressure",
    "political_risk_score",
    "pandemic_score",
    "supply_chain_score",
    "ibovespa_return",
    "global_risk_z",
    "dollar_index_return",
    "rate_differential",
]

REGIME_STATE_FEATURES: list[str] = [
    "actual_return",
    "residual_return",
    "gap_z",
    "rolling_vol_60d",
    "rolling_sharpe_60d",
    "rolling_drawdown",
    "brl_return",
    "cds_z",
    "commodity_z",
    "inflation_pressure",
    "political_risk_score",
    "pandemic_score",
    "supply_chain_score",
]

RISK_MULTIPLIERS: dict[str, float] = {
    "green": 1.0,
    "yellow": 0.7,
    "orange": 0.4,
    "red": 0.1,
}

MANUAL_REGIME_RANGES: list[tuple[str, str, str]] = [
    ("2014-09-01", "2016-12-31", "domestic_crisis"),
    ("2020-02-15", "2020-06-30", "pandemic_shock"),
    ("2021-01-01", "2022-03-31", "inflationary_supply_shock"),
]

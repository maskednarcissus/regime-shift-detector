# Regime Detection MVP

Compact Databricks/Python scaffold for the system in `IMPLEMENTATION.md`.

## Tonight Run Order

1. Put strategy equity CSV at `data/raw/strategy_equity.csv`.
   Required columns: `date`, `actual_equity`, `strategy_name`.
2. Optional CSVs can be added later under `data/raw/` for macro, FX, rates, commodities, market risk, and news.
3. Run notebooks in order from `notebooks/01_ingest_data.py` through `notebooks/10_operational_alerts.py`.
4. For a local smoke run, install dependencies and run:

```powershell
pip install -r requirements.txt
pip install -e .
python -m regime_detection.pipeline --config config/pipeline.yml
```

To try the macro-factor normal model instead of the rolling baseline, set:

```yaml
models:
  normal_model_type: macro_factor_ridge
  normal_ridge_alpha: 1.0
```

Advanced detection knobs:

```yaml
models:
  change_point_method: pelt
  change_point_window: 30
  change_point_z_threshold: 3.0
  change_point_lookback: 180
  change_point_min_size: 20
  change_point_penalty:
  change_point_confirmation_days: 5
  change_point_cooldown_days: 20
  regime_state_count: 4
  regime_state_model: hmm
  hmm_max_iter: 25
  hmm_transition_prior: 2.0
```

Set `change_point_method: rolling` to use the previous rolling mean/volatility shift score, or
`change_point_method: bayesian` to use the Bayesian mean-shift alternative. The default `pelt` path runs an
online trailing-window mean-shift detector, so each date is scored using only information available through that
date. Set `regime_state_model: gaussian_mixture` to use the previous hidden-state fallback; the default `hmm`
path estimates Gaussian emissions and a transition matrix directly.

Production data-quality controls:

```yaml
quality:
  enabled: true
  fail_on_error: false
  max_required_null_fraction: 0.0
  max_duplicate_key_fraction: 0.0
```

Every persisted pipeline table writes checks to `data_quality_results`. Set `fail_on_error: true` when you want the
workflow to stop on missing required columns, duplicate keys, bad dates, empty outputs, or non-finite numeric values.

Phase 4 platform controls:

```yaml
mlflow:
  enabled: true
  experiment_name: /Shared/regime_detection

alerts:
  enabled: false
  webhook_urls: []
  min_hazard_probability: 0.5
  record_heartbeat: true

dashboard:
  definition_path: dashboards/regime_detection_dashboard.sql
```

MLflow logging is best-effort by default and records stage metrics plus CSV artifacts for model diagnostics,
feature importance, backtests, and dashboard status. Operational alerts write `regime_operational_alerts` every
run and can send webhook/email notifications when credentials are configured. The Databricks workflow in
`workflows/regime_detection_job.yml` is scheduled and includes an optional SQL dashboard refresh task.

## Structure

```text
config/pipeline.yml          Small set of paths, table names, and model params
notebooks/                   Databricks notebook entrypoints
src/regime_detection/        Reusable MVP pipeline code
workflows/                   Databricks workflow/job definition stub
dashboards/                  Databricks SQL dashboard query definitions
data/raw/                    Local CSV input drop zone
data/processed/              Local CSV outputs for demos
data_extraction.py           Optional FactSet Formula API economics extract (see below)
mnemonics.csv                Metric specs for data_extraction.py (FactSet mnemonics)
```

## FactSet macro extract (optional)

`data_extraction.py` pulls **FactSet Economics**-style series via the **Formula API** Python SDK (`fds.sdk.Formula`, `TimeSeriesApi` / `/time-series`, default FQL `FDS_ECON_DATA`). It is **not** part of `python -m regime_detection.pipeline`; use it when you need a **monthly Brazil macro panel** as CSV/XLSX before wiring data into Databricks bronze/silver.

1. Fill `mnemonics.csv` (columns: `metric_name`, `mnemonic`, `frequency`, `unit`, `transform`, `description`).
2. Set credentials: `FACTSET_APP_CONFIG_JSON` (OAuth) **or** `FACTSET_USERNAME` + `FACTSET_API_KEY`.
3. `pip install -r requirements.txt` then `python data_extraction.py`.

Outputs (gitignored by default): `brazil_macro_monthly_wide.csv`, `brazil_macro_monthly_long.csv`, `brazil_macro_master.xlsx`. See `IMPLEMENTATION.md` §4.2 companion note and [ROADMAP.md](ROADMAP.md) for how this relates to `silver_brazil_macro` / `gold_regime_features`.

## Optional market CSV inputs

The pipeline also ingests optional daily CSVs from `config/pipeline.yml`:

- `data/raw/factset_fx.csv`: date plus `brl_usd` or `usd_brl`, optional `dollar_index` / `dxy`.
- `data/raw/factset_rates.csv`: date plus `selic`, `di_1y`, `di_5y`, optional `us_10y`.
- `data/raw/factset_commodities.csv`: date plus any of `oil_price`, `iron_ore_price`, `soybean_price`, `commodity_basket`.
- `data/raw/factset_market_risk.csv`: date plus any of `ibovespa`, `brazil_cds`, `vix`.
- `data/raw/factset_news.csv`: date plus score columns or text fields like `headline`, `title`, `body`, `summary`.

Returns, volatilities, z-scores, curve slope, and simple keyword news scores are derived when raw levels/text are provided.

## MVP Scope

Implemented first:

- bronze strategy ingest and optional FactSet CSV bronze ingest
- `silver_strategy_returns`
- bridge from `brazil_macro_monthly_wide.csv` / long CSV to `silver_brazil_macro`
- optional silver tables for FX, rates, commodities, market risk, and news
- `gold_regime_features` with real macro/market forward-fill when CSVs exist
- rolling normal-condition equity curve
- optional Ridge macro-factor normal-condition model
- Ridge coefficient table `normal_model_feature_importance`
- rolling vs Ridge validation table `normal_model_validation`
- gap, residual, gap z-score, CUSUM, alert level
- online PELT change-point score/signal with method, count, detected date, and age metadata
- Bayesian and rolling change-point fallback methods
- Gaussian HMM hidden-state regime labels/probabilities with transition dynamics
- 30/60/90 day labels
- logistic, gradient boosting, Cox survival, and fallback discrete hazard outputs with AUC/PR-AUC/Brier/calibration/concordance metrics
- prediction feature-importance artifact `regime_model_feature_importance`
- prediction diagnostics artifact `regime_prediction_diagnostics`
- historical shock, prediction lead-time, and overlay backtest metrics in `regime_backtest_results`
- daily overlay/equity frame in `regime_backtest_timeseries`
- dashboard current status table
- append-only data quality results in `data_quality_results`
- MLflow stage logging, SQL dashboard definitions, scheduled workflow config, and operational alerts

Deferred by design:

- Kalman filter, advanced NLP, SHAP, model registry.

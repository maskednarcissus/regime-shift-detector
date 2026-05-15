-- Databricks SQL dashboard definition for the regime detection pipeline.
-- Replace table names if your deployment maps logical tables to catalog-qualified names.

-- Current status by strategy
SELECT
  date,
  strategy_name,
  current_alert_level,
  gap_pct,
  gap_z,
  rolling_sharpe_60d,
  rolling_drawdown,
  change_point_score,
  change_point_signal,
  change_point_method,
  hazard_of_regime_change,
  prob_regime_change_30d,
  prob_regime_change_60d,
  prob_regime_change_90d,
  prob_regime_survival_30d,
  prob_regime_survival_60d,
  prob_regime_survival_90d,
  recommended_risk_multiplier,
  dominant_stress_driver,
  model_version
FROM regime_dashboard_current_status
ORDER BY
  CASE current_alert_level
    WHEN 'red' THEN 1
    WHEN 'orange' THEN 2
    WHEN 'yellow' THEN 3
    ELSE 4
  END,
  strategy_name;

-- Daily equity and overlay timeseries
SELECT
  date,
  strategy_name,
  actual_equity,
  normal_equity,
  regime_adjusted_equity,
  regime_alert_level,
  risk_multiplier,
  prob_regime_change_60d,
  hazard_of_regime_change
FROM regime_backtest_timeseries
ORDER BY strategy_name, date;

-- Backtest metric summary
SELECT
  strategy_name,
  test_name,
  metric_name,
  metric_value,
  model_version,
  feature_version
FROM regime_backtest_results
ORDER BY strategy_name, test_name, metric_name;

-- Prediction model metrics
SELECT
  model,
  metric_name,
  metric_value
FROM regime_prediction_metrics
ORDER BY model, metric_name;

-- Top prediction feature drivers
SELECT
  target,
  model,
  feature_name,
  importance_type,
  importance_value,
  normalized_importance,
  model_version,
  feature_version
FROM regime_model_feature_importance
WHERE normalized_importance IS NOT NULL
ORDER BY target, model, normalized_importance DESC;

-- Prediction diagnostics: confusion matrices and calibration curves
SELECT
  artifact_type,
  model,
  target,
  bucket,
  observed_rate,
  predicted_probability,
  row_count,
  model_version,
  feature_version
FROM regime_prediction_diagnostics
ORDER BY artifact_type, model, target, bucket;

-- Data quality failures
SELECT
  checked_at,
  stage,
  logical_table,
  check_name,
  severity,
  observed_value,
  threshold,
  details
FROM data_quality_results
WHERE status = 'fail'
ORDER BY checked_at DESC;

-- Operational alerts
SELECT
  created_at,
  alert_type,
  severity,
  entity,
  message
FROM regime_operational_alerts
ORDER BY created_at DESC;

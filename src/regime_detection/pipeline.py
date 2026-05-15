from __future__ import annotations

import argparse

from .stages import (
    build_dashboard_outputs,
    build_gold_features,
    build_silver_tables,
    create_prediction_training_set,
    emit_operational_alerts_stage,
    ingest_data,
    run_backtest_stage,
    run_detection_stage,
    run_normal_condition_model,
    train_prediction_stage,
)


def run_pipeline(config_path: str = "config/pipeline.yml") -> None:
    ingest_data(config_path)
    build_silver_tables(config_path)
    build_gold_features(config_path)
    run_normal_condition_model(config_path)
    run_detection_stage(config_path)
    create_prediction_training_set(config_path)
    train_prediction_stage(config_path)
    run_backtest_stage(config_path)
    build_dashboard_outputs(config_path)
    emit_operational_alerts_stage(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

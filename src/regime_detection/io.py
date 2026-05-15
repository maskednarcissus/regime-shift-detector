from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path, required: bool = True) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def save_table(df: pd.DataFrame, table_name: str, config: dict[str, Any]) -> None:
    """Write a Databricks table when Spark exists, otherwise write local CSV."""
    spark_session = _active_spark()
    if spark_session is not None:
        spark_session.createDataFrame(df).write.mode("overwrite").saveAsTable(table_name)
        return

    processed_dir = Path(config["data"]["processed_dir"])
    write_csv(df, processed_dir / f"{table_name}.csv")


def append_table(df: pd.DataFrame, table_name: str, config: dict[str, Any]) -> None:
    """Append to a Databricks table when Spark exists, otherwise append local CSV."""
    spark_session = _active_spark()
    if spark_session is not None:
        mode = "append" if spark_session.catalog.tableExists(table_name) else "overwrite"
        spark_session.createDataFrame(df).write.mode(mode).saveAsTable(table_name)
        return

    processed_dir = Path(config["data"]["processed_dir"])
    file_path = processed_dir / f"{table_name}.csv"
    if file_path.exists():
        existing = pd.read_csv(file_path)
        df = pd.concat([existing, df], ignore_index=True)
    write_csv(df, file_path)


def load_processed_table(table_name: str, config: dict[str, Any]) -> pd.DataFrame:
    spark_session = _active_spark()
    if spark_session is not None:
        return spark_session.table(table_name).toPandas()

    processed_dir = Path(config["data"]["processed_dir"])
    return read_csv(processed_dir / f"{table_name}.csv")


def _active_spark() -> Any | None:
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except Exception:
        return None

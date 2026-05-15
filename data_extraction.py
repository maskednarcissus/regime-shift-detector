import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2013-04-01"
END_DATE = "2024-11-30"

INPUT_MNEMONICS_CSV = "mnemonics.csv"

OUTPUT_XLSX = "brazil_macro_master.xlsx"
OUTPUT_WIDE_CSV = "brazil_macro_monthly_wide.csv"
OUTPUT_LONG_CSV = "brazil_macro_monthly_long.csv"

MONTHLY_INDEX = pd.date_range(
    start=pd.Timestamp(START_DATE).to_period("M").to_timestamp("M"),
    end=pd.Timestamp(END_DATE).to_period("M").to_timestamp("M"),
    freq="M",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class MetricSpec:
    metric_name: str
    mnemonic: str
    frequency: str
    unit: str
    transform: str
    description: str


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_column_name(name: str) -> str:
    """
    Convert a human-readable metric name into a model-friendly column name.

    Example:
        "IPCA YoY" -> "ipca_yoy"
        "Current Account Balance 12M" -> "current_account_balance_12m"
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def load_metric_specs(path: str) -> List[MetricSpec]:
    """
    Load the mnemonic mapping file.
    """
    df = pd.read_csv(path)

    required_columns = {
        "metric_name",
        "mnemonic",
        "frequency",
        "unit",
        "transform",
        "description",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    specs = []
    for _, row in df.iterrows():
        specs.append(
            MetricSpec(
                metric_name=str(row["metric_name"]).strip(),
                mnemonic=str(row["mnemonic"]).strip(),
                frequency=str(row["frequency"]).strip().upper(),
                unit=str(row["unit"]).strip(),
                transform=str(row["transform"]).strip().lower(),
                description=str(row["description"]).strip(),
            )
        )

    return specs


# ============================================================
# FACTSET DATA FETCHING (Formula API — Time-Series)
# ============================================================

def _fql_quote_literal(value: str) -> str:
    """Escape a string for use inside single-quoted FQL literals."""
    return value.replace("'", "''")


def _factset_econ_frequency_token(freq: str) -> str:
    """
    Map mnemonics.csv frequency codes to the token expected inside FDS_ECON_DATA.

    FactSet economics frequencies are typically single-letter codes (e.g. D, W, M, Q, S, Y).
    Pass through uppercased tokens we do not recognize.
    """
    f = (freq or "M").strip().upper()
    aliases = {
        "MONTHLY": "M",
        "MONTH": "M",
        "M": "M",
        "DAILY": "D",
        "DAY": "D",
        "D": "D",
        "WEEKLY": "W",
        "W": "W",
        "QUARTERLY": "Q",
        "Q": "Q",
        "SEMIANNUAL": "S",
        "SEMI-ANNUAL": "S",
        "S": "S",
        "ANNUAL": "Y",
        "YEARLY": "Y",
        "Y": "Y",
    }
    return aliases.get(f, f)


def _build_fds_econ_data_formula(
    mnemonic: str,
    start_date: str,
    end_date: str,
    frequency: str,
) -> str:
    """
    Build an FQL expression for FactSet Economics (FDS_ECON_DATA) for the
    Formula API `/time-series` endpoint (FactSet Economics Formula API content).

    Override the full template with env FACTSET_ECON_FQL_TEMPLATE using placeholders:
    {mnemonic}, {start_date}, {end_date}, {frequency}
    """
    m = _fql_quote_literal(mnemonic.strip())
    s = _fql_quote_literal(start_date.strip())
    e = _fql_quote_literal(end_date.strip())
    freq_token = _factset_econ_frequency_token(frequency)

    template = os.environ.get(
        "FACTSET_ECON_FQL_TEMPLATE",
        "FDS_ECON_DATA('{mnemonic}','{start_date}','{end_date}',{frequency},STEP,AVERAGE,1)",
    )
    return template.format(mnemonic=m, start_date=s, end_date=e, frequency=freq_token)


def _formula_api_configuration():
    """
    Build fds.sdk.Formula.Configuration from environment variables.

    OAuth 2.0 (preferred by FactSet enterprise SDK docs):
        FACTSET_APP_CONFIG_JSON — path to app-config.json for ConfidentialClient

    API key (FactSetApiKey):
        FACTSET_USERNAME — FactSet serial username, e.g. COMPANY-123456
        FACTSET_API_KEY — API key string (some deployments use FACTSET_PASSWORD instead)
    """
    try:
        import fds.sdk.Formula
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install FactSet Formula SDK: pip install 'fds.sdk.Formula>=2.0' 'fds.sdk.utils'"
        ) from exc

    app_cfg = os.environ.get("FACTSET_APP_CONFIG_JSON", "").strip()
    if app_cfg:
        from fds.sdk.utils.authentication import ConfidentialClient

        return fds.sdk.Formula.Configuration(
            fds_oauth_client=ConfidentialClient(app_cfg),
        )

    username = (
        os.environ.get("FACTSET_USERNAME", "").strip()
        or os.environ.get("FACTSET_SERIAL", "").strip()
    )
    password = (
        os.environ.get("FACTSET_API_KEY", "").strip()
        or os.environ.get("FACTSET_PASSWORD", "").strip()
    )
    if username and password:
        return fds.sdk.Formula.Configuration(username=username, password=password)

    raise RuntimeError(
        "FactSet credentials not configured. Set FACTSET_APP_CONFIG_JSON (OAuth) "
        "or FACTSET_USERNAME and FACTSET_API_KEY. See data_extraction.py docstring."
    )


def _scalar_cell_to_float(cell: Any) -> float:
    """Normalize a Formula API scalar / JSON cell to float."""
    if cell is None:
        return float("nan")
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, str):
        try:
            return float(cell)
        except ValueError:
            return float("nan")
    if isinstance(cell, dict):
        for key in ("value", "d", "double", "float", "data"):
            if key in cell and cell[key] is not None:
                return _scalar_cell_to_float(cell[key])
        if len(cell) == 1:
            return _scalar_cell_to_float(next(iter(cell.values())))
    return float("nan")


def _deep_find_dates_values(obj: Any) -> Optional[tuple[list[Any], list[Any]]]:
    """Locate a TIMESERIES-style dict with dates + values anywhere in nested JSON."""
    if isinstance(obj, dict):
        dates = obj.get("dates") if "dates" in obj else obj.get("Dates")
        values = obj.get("values") if "values" in obj else obj.get("Values")
        if isinstance(dates, list) and isinstance(values, list) and (dates or values):
            return dates, values
        for v in obj.values():
            found = _deep_find_dates_values(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_dates_values(item)
            if found is not None:
                return found
    return None


def _time_series_response_to_frame(api_response: Any) -> pd.DataFrame:
    """Turn TimeSeriesResponse into columns date, value."""
    rows: list[dict[str, Any]] = []

    data = getattr(api_response, "data", None) or []
    for item in data:
        err = getattr(item, "error", None)
        if err not in (None, 0):
            msg = getattr(item, "error_message", None) or getattr(item, "errorMessage", None)
            formula = getattr(item, "formula", None) or ""
            raise RuntimeError(f"FactSet Formula error on {formula!r}: code={err} message={msg}")

        as_dict = item.to_dict() if hasattr(item, "to_dict") else item
        pair = _deep_find_dates_values(as_dict)
        if pair is None:
            continue
        dates, values = pair
        n = min(len(dates), len(values))
        for i in range(n):
            rows.append(
                {
                    "date": pd.to_datetime(dates[i], errors="coerce"),
                    "value": _scalar_cell_to_float(values[i]),
                }
            )

    if not rows:
        raise RuntimeError("FactSet Formula API returned no time-series rows (check mnemonic and entitlements).")

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["date"])
    return out.sort_values("date").reset_index(drop=True)


def fetch_factset_econ_series(
    mnemonic: str,
    start_date: str,
    end_date: str,
    frequency: str,
) -> pd.DataFrame:
    """
    Fetch one economic time series via FactSet **Formula API** (`/time-series`),
    using the Python SDK package **fds.sdk.Formula** (same endpoint family as the
    **FactSet Economics Formula API** for macro series).

    Expected return format::

        date        value
        2013-04-30  6.49
        2013-05-31  6.50
        ...

    **Authentication (environment)**

    - OAuth: set ``FACTSET_APP_CONFIG_JSON`` to the path of your FactSet
      ``app-config.json`` (uses ``fds.sdk.utils.authentication.ConfidentialClient``).
    - API key: set ``FACTSET_USERNAME`` (e.g. ``COMPANY-123456``) and
      ``FACTSET_API_KEY`` (or ``FACTSET_PASSWORD``).

    **Optional tuning**

    - ``FACTSET_ECON_FQL_TEMPLATE`` — Python ``str.format`` template with placeholders
      ``{mnemonic}``, ``{start_date}``, ``{end_date}``, ``{frequency}`` (each mnemonic
      segment is single-quote escaped for FQL). Default uses ``FDS_ECON_DATA(...)``.
    - ``FACTSET_FORMULA_CALENDAR`` — calendar code for ``TimeSeriesRequestData``
      (default ``SEVENDAY``; try ``FIVEDAY`` or FactSet docs for your series).
    - ``FACTSET_FORMULA_TS_IDS`` — comma-separated ``ids`` for the time-series request
      if your firm requires a non-empty ``ids`` list alongside economics FQL.
    - ``FACTSET_FORMULA_FLATTEN`` — ``Y`` or ``N`` for the API ``flatten`` flag.
    - ``FACTSET_INTER_REQUEST_SLEEP_SECONDS`` — throttle between mnemonic calls.

    Requires: ``pip install 'fds.sdk.Formula>=2.0' 'fds.sdk.utils'``
    """
    import fds.sdk.Formula
    from fds.sdk.Formula.api import time_series_api
    from fds.sdk.Formula.models import TimeSeriesRequest, TimeSeriesRequestData

    sleep_s = float(os.environ.get("FACTSET_INTER_REQUEST_SLEEP_SECONDS", "0") or "0")
    if sleep_s > 0:
        time.sleep(sleep_s)

    formula = _build_fds_econ_data_formula(mnemonic, start_date, end_date, frequency)

    ids_env = os.environ.get("FACTSET_FORMULA_TS_IDS", "").strip()
    ids: Optional[List[str]] = [s.strip() for s in ids_env.split(",") if s.strip()] if ids_env else None

    calendar = os.environ.get("FACTSET_FORMULA_CALENDAR", "SEVENDAY").strip() or "SEVENDAY"
    flatten = os.environ.get("FACTSET_FORMULA_FLATTEN", "N").strip().upper() or "N"
    if flatten not in ("Y", "N"):
        flatten = "N"

    kwargs: Dict[str, Any] = {
        "formulas": [formula],
        "calendar": calendar,
        "flatten": flatten,
        "dates": "Y",
        "batch": "N",
    }
    if ids:
        kwargs["ids"] = ids

    request_body = TimeSeriesRequest(
        data=TimeSeriesRequestData(
            **kwargs,
        )
    )

    configuration = _formula_api_configuration()
    with fds.sdk.Formula.ApiClient(configuration) as api_client:
        api_instance = time_series_api.TimeSeriesApi(api_client)
        wrapper = api_instance.get_time_series_data_for_list(request_body)

    status = wrapper.get_status_code()
    if status == 202:
        raise RuntimeError(
            "FactSet Formula API returned 202 (async batch). "
            "Set batch to N (default here) or implement batch polling per FactSet docs."
        )
    if status != 200:
        raise RuntimeError(f"FactSet Formula API unexpected HTTP status {status}")

    api_response = wrapper.get_response_200()
    return _time_series_response_to_frame(api_response)


# ============================================================
# TRANSFORMATION LOGIC
# ============================================================

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date column is datetime and sorted.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["date"])
    out = out.sort_values("date")
    return out


def transform_series_to_monthly(
    raw_df: pd.DataFrame,
    spec: MetricSpec,
) -> pd.Series:
    """
    Convert raw FactSet series into a monthly series aligned to month-end dates.

    Supported transforms:
        raw
        monthly_average
        month_end
        rolling_12m_sum
        rolling_12m_average
        quarter_end_only
        quarterly_forward_fill
    """
    df = normalize_dates(raw_df)

    if "value" not in df.columns:
        raise ValueError(f"Raw data for {spec.metric_name} does not contain a 'value' column.")

    df = df[["date", "value"]].dropna()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if df.empty:
        return pd.Series(index=MONTHLY_INDEX, dtype=float, name=clean_column_name(spec.metric_name))

    df = df.set_index("date").sort_index()

    transform = spec.transform

    if transform == "raw":
        monthly = df["value"].resample("M").last()

    elif transform == "monthly_average":
        monthly = df["value"].resample("M").mean()

    elif transform == "month_end":
        monthly = df["value"].resample("M").last()

    elif transform == "rolling_12m_sum":
        monthly_raw = df["value"].resample("M").sum()
        monthly = monthly_raw.rolling(window=12, min_periods=12).sum()

    elif transform == "rolling_12m_average":
        monthly_raw = df["value"].resample("M").mean()
        monthly = monthly_raw.rolling(window=12, min_periods=12).mean()

    elif transform == "quarter_end_only":
        quarterly = df["value"].resample("Q").last()
        monthly = quarterly.reindex(MONTHLY_INDEX)

    elif transform == "quarterly_forward_fill":
        quarterly = df["value"].resample("Q").last()
        monthly = quarterly.reindex(MONTHLY_INDEX).ffill()

    else:
        raise ValueError(
            f"Unsupported transform '{spec.transform}' for metric '{spec.metric_name}'."
        )

    monthly = monthly.reindex(MONTHLY_INDEX)
    monthly.name = clean_column_name(spec.metric_name)

    return monthly


# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(specs: List[MetricSpec]) -> Dict[str, pd.DataFrame]:
    """
    Fetch, transform, and combine all series into wide and long datasets.
    """
    wide_df = pd.DataFrame(index=MONTHLY_INDEX)
    raw_audit_rows = []
    errors = []

    unique_fetches = {}

    for spec in specs:
        key = (spec.mnemonic, spec.frequency)

        if key not in unique_fetches:
            logging.info(f"Fetching {spec.mnemonic} [{spec.frequency}]")

            try:
                raw_df = fetch_factset_econ_series(
                    mnemonic=spec.mnemonic,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    frequency=spec.frequency,
                )

                raw_df = normalize_dates(raw_df)
                unique_fetches[key] = raw_df

            except Exception as exc:
                logging.error(f"Failed to fetch {spec.mnemonic}: {exc}")
                unique_fetches[key] = None
                errors.append({
                    "metric_name": spec.metric_name,
                    "mnemonic": spec.mnemonic,
                    "error": str(exc),
                })

        raw_df = unique_fetches[key]

        if raw_df is None:
            wide_df[clean_column_name(spec.metric_name)] = np.nan
            continue

        try:
            transformed = transform_series_to_monthly(raw_df, spec)
            wide_df[transformed.name] = transformed

            raw_audit_rows.append({
                "metric_name": spec.metric_name,
                "mnemonic": spec.mnemonic,
                "raw_start": raw_df["date"].min(),
                "raw_end": raw_df["date"].max(),
                "raw_observations": len(raw_df),
                "non_null_monthly_observations": int(transformed.notna().sum()),
            })

        except Exception as exc:
            logging.error(f"Failed to transform {spec.metric_name}: {exc}")
            wide_df[clean_column_name(spec.metric_name)] = np.nan
            errors.append({
                "metric_name": spec.metric_name,
                "mnemonic": spec.mnemonic,
                "error": str(exc),
            })

    wide_df = wide_df.reset_index().rename(columns={"index": "date"})

    long_df = wide_to_long(wide_df, specs)

    mnemonic_map_df = pd.DataFrame([spec.__dict__ for spec in specs])
    mnemonic_map_df["column_name"] = mnemonic_map_df["metric_name"].apply(clean_column_name)

    audit_df = pd.DataFrame(raw_audit_rows)
    errors_df = pd.DataFrame(errors)

    data_dictionary_df = build_data_dictionary(specs)

    return {
        "wide": wide_df,
        "long": long_df,
        "mnemonic_map": mnemonic_map_df,
        "audit": audit_df,
        "errors": errors_df,
        "data_dictionary": data_dictionary_df,
    }


def wide_to_long(wide_df: pd.DataFrame, specs: List[MetricSpec]) -> pd.DataFrame:
    """
    Convert the wide modeling table into a long table.
    """
    map_rows = []

    for spec in specs:
        map_rows.append({
            "metric_name": spec.metric_name,
            "column_name": clean_column_name(spec.metric_name),
            "mnemonic": spec.mnemonic,
            "frequency": spec.frequency,
            "unit": spec.unit,
            "transform": spec.transform,
            "description": spec.description,
        })

    map_df = pd.DataFrame(map_rows)

    value_vars = [c for c in wide_df.columns if c != "date"]

    long_df = wide_df.melt(
        id_vars=["date"],
        value_vars=value_vars,
        var_name="column_name",
        value_name="value",
    )

    long_df = long_df.merge(map_df, on="column_name", how="left")

    long_df = long_df[
        [
            "date",
            "column_name",
            "metric_name",
            "mnemonic",
            "value",
            "frequency",
            "unit",
            "transform",
            "description",
        ]
    ]

    return long_df


def build_data_dictionary(specs: List[MetricSpec]) -> pd.DataFrame:
    """
    Build a clean data dictionary for the workbook.
    """
    rows = []

    for spec in specs:
        rows.append({
            "column_name": clean_column_name(spec.metric_name),
            "metric_name": spec.metric_name,
            "mnemonic": spec.mnemonic,
            "unit": spec.unit,
            "source": "FactSet",
            "raw_frequency": spec.frequency,
            "final_frequency": "M",
            "date_alignment": "Month-end date index",
            "transformation": spec.transform,
            "description": spec.description,
        })

    return pd.DataFrame(rows)


# ============================================================
# EXPORT LOGIC
# ============================================================

def export_outputs(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Export the dataset to CSV and XLSX.
    """
    wide_df = datasets["wide"]
    long_df = datasets["long"]

    wide_df.to_csv(OUTPUT_WIDE_CSV, index=False)
    long_df.to_csv(OUTPUT_LONG_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        datasets["wide"].to_excel(writer, sheet_name="Data_Wide", index=False)
        datasets["long"].to_excel(writer, sheet_name="Data_Long", index=False)
        datasets["mnemonic_map"].to_excel(writer, sheet_name="Mnemonic_Map", index=False)
        datasets["data_dictionary"].to_excel(writer, sheet_name="Data_Dictionary", index=False)
        datasets["audit"].to_excel(writer, sheet_name="Audit", index=False)
        datasets["errors"].to_excel(writer, sheet_name="Errors", index=False)

        workbook = writer.book

        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "border": 1,
        })

        for sheet_name, df in datasets.items():
            excel_sheet_name = {
                "wide": "Data_Wide",
                "long": "Data_Long",
                "mnemonic_map": "Mnemonic_Map",
                "data_dictionary": "Data_Dictionary",
                "audit": "Audit",
                "errors": "Errors",
            }.get(sheet_name)

            if excel_sheet_name is None:
                continue

            worksheet = writer.sheets[excel_sheet_name]

            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(0, col_idx, col_name, header_format)

                if col_name == "date" or "date" in col_name.lower() or col_name.endswith("_start") or col_name.endswith("_end"):
                    worksheet.set_column(col_idx, col_idx, 14, date_format)
                else:
                    max_len = max(
                        [len(str(col_name))]
                        + [len(str(x)) for x in df[col_name].head(200).fillna("").values]
                    )
                    worksheet.set_column(col_idx, col_idx, min(max_len + 2, 45))

            worksheet.freeze_panes(1, 1)
            worksheet.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))

    logging.info(f"Exported {OUTPUT_XLSX}")
    logging.info(f"Exported {OUTPUT_WIDE_CSV}")
    logging.info(f"Exported {OUTPUT_LONG_CSV}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    specs = load_metric_specs(INPUT_MNEMONICS_CSV)

    logging.info(f"Loaded {len(specs)} metric specifications.")
    logging.info(f"Date range: {START_DATE} to {END_DATE}")
    logging.info(f"Monthly rows expected: {len(MONTHLY_INDEX)}")

    datasets = build_dataset(specs)
    export_outputs(datasets)

    if not datasets["errors"].empty:
        logging.warning("Some series failed. Check the Errors sheet in the workbook.")


if __name__ == "__main__":
    main()
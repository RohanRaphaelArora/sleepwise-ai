"""Loading and validating the sleep dataset, and writing the schema report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config
from .preprocessing import clean_dataframe, get_feature_roles


class DatasetError(Exception):
    """Raised with a human-readable message when the dataset is missing or unusable."""


def load_raw_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV exactly as stored (no cleaning). Raises ``DatasetError`` on problems."""
    csv_path = config.get_data_path(path)
    if not csv_path.exists():
        raise DatasetError(
            f"Dataset not found at: {csv_path}\n"
            f"1. Download the Kaggle 'Sleep Health and Lifestyle Dataset' CSV.\n"
            f"2. Save it as {config.DEFAULT_DATA_PATH} (or pass --data-path, or set the "
            f"{config.DATA_PATH_ENV_VAR} environment variable)."
        )
    try:
        # keep_default_na=False so the category "None" in 'Sleep Disorder' is kept as text.
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=config.NA_VALUES)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"The file {csv_path} is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"The file {csv_path} could not be parsed as a CSV: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DatasetError(f"The file {csv_path} has a header but no data rows.")
    return df


def validate_dataset(df: pd.DataFrame, target: str = config.TARGET_COLUMN) -> None:
    """Check that the frame can be used for training; raise ``DatasetError`` if not."""
    if target not in df.columns:
        raise DatasetError(
            f"Required target column '{target}' was not found.\n"
            f"Columns present: {list(df.columns)}\n"
            "This project expects the Kaggle 'Sleep Health and Lifestyle' dataset, or a "
            "CSV with a numeric 'Quality of Sleep' column."
        )
    numeric_target = pd.to_numeric(df[target], errors="coerce")
    if numeric_target.notna().sum() == 0:
        raise DatasetError(f"Target column '{target}' contains no numeric values.")
    if numeric_target.nunique() < 2:
        raise DatasetError(f"Target column '{target}' is constant; nothing can be learned.")
    clean = clean_dataframe(df)
    roles = get_feature_roles(clean, target)
    if not roles.features:
        raise DatasetError("No usable feature columns remain after exclusions.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_schema_report(df: pd.DataFrame, path: str | Path | None = None) -> dict:
    """Describe the raw dataset: size, columns, missing values, duplicates, feature roles."""
    target = config.TARGET_COLUMN
    clean = clean_dataframe(df)
    roles = get_feature_roles(clean, target)

    columns = []
    for col in df.columns:
        series = df[col]
        columns.append(
            {
                "name": col,
                "dtype": "numeric" if is_numeric_dtype(series) else "text",
                "missing": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "example": None if series.dropna().empty else str(series.dropna().iloc[0]),
            }
        )

    id_cols = [c for c in df.columns if c in config.EXCLUDED_FEATURES and c == config.ID_COLUMN]
    content_cols = [c for c in df.columns if c not in id_cols]
    model_cols = [c for c in [*roles.features, target] if c in clean.columns]

    report: dict = {
        "file": str(config.get_data_path(path)),
        "sha256": None,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "column_details": columns,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows_exact": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(df[content_cols].duplicated().sum()),
        "distinct_rows_ignoring_id": int(len(df[content_cols].drop_duplicates())),
        "distinct_model_feature_combinations": int(
            len(clean[roles.features].drop_duplicates()) if roles.features else 0
        ),
        "distinct_feature_and_target_combinations": int(
            len(clean[model_cols].drop_duplicates()) if model_cols else 0
        ),
        "target": {"name": target, "present": target in df.columns},
        "feature_roles": roles.to_dict(),
        "notes": [
            "'None' in 'Sleep Disorder' is a real category (no disorder), not a missing value.",
            "'Blood Pressure' text is split into numeric 'Systolic BP' and 'Diastolic BP'.",
            "'Normal Weight' is merged into 'Normal' in 'BMI Category' (same group, two spellings).",
        ],
    }
    if target in df.columns:
        t = pd.to_numeric(df[target], errors="coerce")
        report["target"].update(
            {
                "min": float(t.min()),
                "max": float(t.max()),
                "mean": float(t.mean()),
                "value_counts": {str(k): int(v) for k, v in t.value_counts().sort_index().items()},
            }
        )
    csv_path = config.get_data_path(path)
    if csv_path.exists():
        report["sha256"] = file_sha256(csv_path)
    return report


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    return path


def print_summary(report: dict) -> None:
    """Console summary shown at the start of training."""
    print(f"Dataset: {report['file']}")
    print(f"  Rows x columns          : {report['rows']} x {report['columns']}")
    print(f"  Missing values (total)  : {report['missing_values_total']}")
    print(f"  Exact duplicate rows    : {report['duplicate_rows_exact']}")
    print(f"  Duplicates ignoring ID  : {report['duplicate_rows_ignoring_id']}")
    print(f"  Detected columns        : {report['column_names']}")
    roles = report["feature_roles"]
    print(f"  Numeric features        : {roles['numeric']}")
    print(f"  Categorical features    : {roles['categorical']}")
    for col, why in roles["excluded"].items():
        print(f"  Excluded '{col}': {why}")

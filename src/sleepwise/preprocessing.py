"""Cleaning, feature selection and the scikit-learn preprocessing transformer.

Order of operations
-------------------
1. ``clean_dataframe``  - strip text, harmonise category spellings, split
   ``Blood Pressure`` ("120/80") into numeric systolic and diastolic columns.
2. ``get_feature_roles`` - decide which columns are numeric inputs, categorical
   inputs, or excluded (ID, target, Sleep Disorder, empty columns).
3. ``build_preprocessor`` - a ``ColumnTransformer`` that is placed *inside* the
   model pipeline, so imputers/scalers/encoders are fitted on training rows only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def parse_blood_pressure(df: pd.DataFrame, column: str = config.BP_COLUMN) -> pd.DataFrame:
    """Split values such as ``"126/83"`` into ``Systolic BP`` and ``Diastolic BP``.

    * Values that do not match ``<number>/<number>`` become NaN (never guessed) and a
      warning reports how many there were; the imputer later handles them.
    * The original text column is dropped from the returned frame.
    * If the column does not exist, the frame is returned unchanged (copy).
    """
    out = df.copy()
    if column not in out.columns:
        return out

    text = out[column].astype("string").str.strip()
    parts = text.str.extract(r"^(\d{2,3})\s*/\s*(\d{2,3})$")
    out[config.SYSTOLIC_COLUMN] = pd.to_numeric(parts[0], errors="coerce").astype("float64")
    out[config.DIASTOLIC_COLUMN] = pd.to_numeric(parts[1], errors="coerce").astype("float64")

    unparseable = int((text.notna() & parts[0].isna()).sum())
    if unparseable:
        warnings.warn(
            f"{unparseable} value(s) in '{column}' were not in the form 'systolic/diastolic' "
            "and were set to missing.",
            stacklevel=2,
        )
    return out.drop(columns=[column])


def normalise_categories(
    df: pd.DataFrame, aliases: dict[str, dict[str, str]] | None = None
) -> pd.DataFrame:
    """Strip whitespace in text columns and merge duplicate spellings of one category."""
    aliases = config.CATEGORY_ALIASES if aliases is None else aliases
    out = df.copy()
    for col in out.columns:
        if not is_numeric_dtype(out[col]):
            out[col] = out[col].astype("string").str.strip().astype(object)
            # astype(object) on a string column can leave pd.NA; use NaN for sklearn.
            out[col] = out[col].where(out[col].notna(), np.nan)
    for col, mapping in aliases.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)
    return out


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic cleaning steps. Never invents or imputes values."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = normalise_categories(out)
    out = parse_blood_pressure(out)
    return out


# --------------------------------------------------------------------------- #
# Feature roles
# --------------------------------------------------------------------------- #
@dataclass
class FeatureRoles:
    """Which columns the model uses, and why the others are not used."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    derived: dict[str, list[str]] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def to_dict(self) -> dict:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "excluded": self.excluded,
            "derived": self.derived,
        }


def get_feature_roles(clean_df: pd.DataFrame, target: str = config.TARGET_COLUMN) -> FeatureRoles:
    """Classify the columns of a *cleaned* frame (see ``clean_dataframe``)."""
    roles = FeatureRoles()
    if config.SYSTOLIC_COLUMN in clean_df.columns:
        roles.derived[config.BP_COLUMN] = [config.SYSTOLIC_COLUMN, config.DIASTOLIC_COLUMN]

    for col in clean_df.columns:
        if col == target:
            roles.excluded[col] = "Prediction target; using it as an input would be target leakage."
        elif col in config.EXCLUDED_FEATURES:
            roles.excluded[col] = config.EXCLUDED_FEATURES[col]
        elif col.strip().lower() in config.ID_LIKE_NAMES:
            roles.excluded[col] = "Looks like a row identifier; carries no information about sleep quality."
        elif clean_df[col].notna().sum() == 0:
            roles.excluded[col] = "Column has no observed values."
        elif is_numeric_dtype(clean_df[col]) and not is_bool_dtype(clean_df[col]):
            roles.numeric.append(col)
        else:
            roles.categorical.append(col)
    return roles


# --------------------------------------------------------------------------- #
# Prepared dataset
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    roles: FeatureRoles
    clean_df: pd.DataFrame
    rows_dropped_missing_target: int


def prepare_dataset(raw_df: pd.DataFrame, target: str = config.TARGET_COLUMN) -> PreparedData:
    """Clean the raw frame and split it into features ``X`` and target ``y``.

    Rows with a missing target are dropped (a label cannot be imputed honestly) and
    the number is reported. Missing *feature* values are kept and imputed later,
    inside the pipeline, using training rows only.
    """
    clean = clean_dataframe(raw_df)
    if target not in clean.columns:
        raise KeyError(f"Target column '{target}' is not present.")
    clean[target] = pd.to_numeric(clean[target], errors="coerce")

    n_before = len(clean)
    clean = clean.dropna(subset=[target]).reset_index(drop=True)
    dropped = n_before - len(clean)

    roles = get_feature_roles(clean, target)
    X = clean[roles.features].copy()
    for col in roles.categorical:
        X[col] = X[col].astype(object)
    for col in roles.numeric:
        X[col] = X[col].astype("float64")
    y = clean[target].astype("float64")
    return PreparedData(X=X, y=y, roles=roles, clean_df=clean, rows_dropped_missing_target=dropped)


def feature_group_ids(X: pd.DataFrame) -> np.ndarray:
    """Integer id per distinct combination of feature values.

    The Kaggle file contains many rows whose features are identical. If such rows
    land on both sides of a random train/test split, the model is "tested" on rows
    it has effectively memorised. Grouping identical rows keeps them together.
    """
    keys = X.astype(str).agg("|".join, axis=1)
    return pd.factorize(keys)[0]


# --------------------------------------------------------------------------- #
# scikit-learn transformer
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot for categories.

    ``handle_unknown="ignore"`` means a category never seen in training is encoded as
    all zeros instead of raising an error at prediction time.
    """
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were found after exclusions.")
    return ColumnTransformer(transformers, remainder="drop")

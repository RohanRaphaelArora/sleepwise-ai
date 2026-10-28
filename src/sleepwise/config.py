"""Central configuration for SleepWise.

Everything that other modules need to agree on (paths, column names, split
settings, plausible input ranges, colours) lives here, so there is exactly one
place to change it.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives in <root>/src/sleepwise/, so the project root is two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

DEFAULT_CSV_NAME = "Sleep_health_and_lifestyle_dataset.csv"
DEFAULT_DATA_PATH: Path = RAW_DIR / DEFAULT_CSV_NAME
# Set this environment variable to use a CSV stored somewhere else.
DATA_PATH_ENV_VAR = "SLEEPWISE_DATA_PATH"

# File names of generated artefacts (the directory is passed in where needed).
MODEL_FILE = "sleepwise_model.joblib"
METADATA_FILE = "model_metadata.json"
DURATION_MODEL_FILE = "sleepwise_duration_only.joblib"
DURATION_METADATA_FILE = "duration_only_metadata.json"
METRICS_FILE = "model_metrics.json"
SCHEMA_FILE = "dataset_schema.json"
EDA_SUMMARY_FILE = "eda_summary.json"
IMPORTANCE_FILE = "feature_importance.csv"
ARCHITECTURE_FILE = "network_architecture.json"
CLEANED_CSV_FILE = "sleep_cleaned.csv"


def get_data_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve which CSV to use: explicit argument > environment variable > default."""
    if path:
        return Path(path).expanduser()
    env_value = os.environ.get(DATA_PATH_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_DATA_PATH


# --------------------------------------------------------------------------- #
# Column names (Kaggle "Sleep Health and Lifestyle" naming)
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Quality of Sleep"
ID_COLUMN = "Person ID"
BP_COLUMN = "Blood Pressure"
SYSTOLIC_COLUMN = "Systolic BP"
DIASTOLIC_COLUMN = "Diastolic BP"
SLEEP_DISORDER_COLUMN = "Sleep Disorder"
SLEEP_DURATION_COLUMN = "Sleep Duration"
STRESS_COLUMN = "Stress Level"
ACTIVITY_COLUMN = "Physical Activity Level"
BMI_COLUMN = "BMI Category"
GENDER_COLUMN = "Gender"
OCCUPATION_COLUMN = "Occupation"

# Columns that are never used as model inputs, with the reason (shown in reports).
EXCLUDED_FEATURES: dict[str, str] = {
    ID_COLUMN: "Row identifier; carries no information about sleep quality.",
    SLEEP_DISORDER_COLUMN: (
        "Excluded from the initial model by design: it is a recorded outcome that sits "
        "alongside sleep quality and could leak information about the target."
    ),
}
# Lower-cased names that are treated as identifiers even if spelled differently.
ID_LIKE_NAMES = {"person id", "id", "user id", "userid", "index", "row id"}

# Value harmonisation applied before modelling: {column: {raw value: canonical value}}.
# The dataset spells the same BMI group two ways ("Normal" and "Normal Weight").
CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    BMI_COLUMN: {"Normal Weight": "Normal"},
}

# Strings treated as missing when reading the CSV. NOTE: "None" is deliberately NOT
# in this list: in "Sleep Disorder", "None" is a real category (no disorder).
NA_VALUES = ["", " ", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null"]

# --------------------------------------------------------------------------- #
# Modelling settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_DISTINCT_ROWS = 20  # refuse to train on fewer distinct feature combinations
SELECTION_METRIC = "rmse"  # lowest mean cross-validated RMSE on the TRAINING split
# The course project is about neural networks, so the NEURAL NETWORK is the deployed model
# by design (decided before looking at any test result). All other models are comparators and
# the best model by cross-validation is reported next to it. Set to "auto" to deploy whichever
# candidate has the lowest cross-validated RMSE instead.
DEPLOYED_MODEL = "Neural Network (MLP)"

# The Kaggle description presents sleep quality on a 1-10 scale.
TARGET_SCALE_MIN = 1.0
TARGET_SCALE_MAX = 10.0

# --------------------------------------------------------------------------- #
# Input validation for the prediction form (hard limits, not the training range)
# --------------------------------------------------------------------------- #
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "Age": (6, 100),  # minors are allowed so the app can say the model was trained on adults only
    SLEEP_DURATION_COLUMN: (0.0, 16.0),  # interface limits only, NOT clinical thresholds
    ACTIVITY_COLUMN: (0, 300),
    STRESS_COLUMN: (1, 10),
    "Heart Rate": (35, 200),
    "Daily Steps": (0, 50_000),
    SYSTOLIC_COLUMN: (70, 250),
    DIASTOLIC_COLUMN: (40, 150),
}
UNITS: dict[str, str] = {
    "Age": "years",
    SLEEP_DURATION_COLUMN: "hours per night",
    "Heart Rate": "beats per minute",
    "Daily Steps": "steps per day",
    SYSTOLIC_COLUMN: "mmHg",
    DIASTOLIC_COLUMN: "mmHg",
    STRESS_COLUMN: "scale 1-10",
    ACTIVITY_COLUMN: "dataset activity level",
}

# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #
NAVY = "#0B2545"      # titles / lines in figures (light theme)
TEAL = "#13A89E"
TEAL_DARK = "#0E7C74"
SLATE = "#5B6B7F"
LIGHT_BG = "#F7FAFC"

# Figures are rendered twice: a light set (reports/figures, used in documents) and a dark set
# (reports/figures/dark, used by the dark dashboard). Plot code reads these names at call time.
FIG_BG = "#FFFFFF"
FIG_SURFACE = "#F2F7F9"
FIG_LINE = "#B8C4CE"
FIG_RC: dict[str, str] = {}
_LIGHT = dict(NAVY="#0B2545", TEAL="#13A89E", TEAL_DARK="#0E7C74", SLATE="#5B6B7F",
              FIG_BG="#FFFFFF", FIG_SURFACE="#F2F7F9", FIG_LINE="#B8C4CE", FIG_RC={})
_DARK = dict(NAVY="#E8EEEA", TEAL="#3CCBC0", TEAL_DARK="#3CCBC0", SLATE="#9AA7A1",
             FIG_BG="#161D1B", FIG_SURFACE="#1D2523", FIG_LINE="#3A4743",
             FIG_RC={"figure.facecolor": "#161D1B", "axes.facecolor": "#161D1B", "savefig.facecolor": "#161D1B",
                     "text.color": "#E8EEEA", "axes.labelcolor": "#C9D3CE", "axes.edgecolor": "#3A4743",
                     "xtick.color": "#C9D3CE", "ytick.color": "#C9D3CE", "grid.color": "#2A332F",
                     "legend.labelcolor": "#C9D3CE"})


def set_figure_theme(name: str) -> None:
    """Switch the colours used by all plotting code. name is "light" or "dark"."""
    values = _DARK if name == "dark" else _LIGHT
    globals().update(values)

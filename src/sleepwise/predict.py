"""Loading the saved pipeline, validating form input and producing a prediction.

The Streamlit app is a thin layer over this module so the logic can be unit-tested.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from . import config


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model or its metadata has not been created yet."""


class InputValidationError(ValueError):
    """Raised when form input is missing, out of range or not an allowed category."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class PredictionResult:
    raw_prediction: float
    displayed_prediction: float
    was_clipped: bool
    warnings: list[str] = field(default_factory=list)
    input_frame: pd.DataFrame | None = None


VARIANTS = ("full", "duration_only")


def artifact_paths(models_dir: str | Path = config.MODELS_DIR, variant: str = "full") -> tuple[Path, Path]:
    """Model and metadata paths. variant: "full" (all features) or "duration_only" (Experiment A)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    models_dir = Path(models_dir)
    if variant == "duration_only":
        return models_dir / config.DURATION_MODEL_FILE, models_dir / config.DURATION_METADATA_FILE
    return models_dir / config.MODEL_FILE, models_dir / config.METADATA_FILE


def model_artifacts_available(models_dir: str | Path = config.MODELS_DIR, variant: str = "full") -> bool:
    model_path, meta_path = artifact_paths(models_dir, variant)
    return model_path.exists() and meta_path.exists()


def load_metadata(models_dir: str | Path = config.MODELS_DIR, variant: str = "full") -> dict:
    _, meta_path = artifact_paths(models_dir, variant)
    if not meta_path.exists():
        raise ModelNotFoundError(
            f"Model metadata not found at {meta_path}. Run `python train.py` first."
        )
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_model(models_dir: str | Path = config.MODELS_DIR, variant: str = "full"):
    """Load the saved scikit-learn pipeline; warn if scikit-learn versions differ."""
    model_path, _ = artifact_paths(models_dir, variant)
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at {model_path}. Run `python train.py` first."
        )
    pipeline = joblib.load(model_path)
    try:
        trained_with = load_metadata(models_dir, variant).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} "
                "is installed. Re-run `python train.py` if you see errors.",
                stacklevel=2,
            )
    except ModelNotFoundError:
        pass
    return pipeline


def feature_names(metadata: dict) -> list[str]:
    return list(metadata["feature_order"])


def validate_inputs(inputs: dict, metadata: dict) -> tuple[dict, list[str]]:
    """Check user input against the model's feature specification.

    Returns ``(clean_inputs, warnings)``. Hard problems (missing value, not a number,
    outside plausible limits, unknown category) raise ``InputValidationError`` listing
    *all* problems. Values outside the range seen in training are allowed but produce
    a warning, because the model is then extrapolating.
    """
    problems: list[str] = []
    notes: list[str] = []
    clean: dict = {}

    for spec in metadata["features"]:
        name = spec["name"]
        if name not in inputs or inputs[name] is None or (isinstance(inputs[name], str) and not inputs[name].strip()):
            problems.append(f"'{name}' is required.")
            continue
        value = inputs[name]

        if spec["kind"] == "numeric":
            if isinstance(value, bool):
                problems.append(f"'{name}' must be a number.")
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"'{name}' must be a number (got {value!r}).")
                continue
            if math.isnan(number) or math.isinf(number):
                problems.append(f"'{name}' must be a finite number.")
                continue
            lo, hi = spec["hard_min"], spec["hard_max"]
            if not lo <= number <= hi:
                problems.append(f"'{name}' must be between {lo:g} and {hi:g} (got {number:g}).")
                continue
            if not spec["observed_min"] <= number <= spec["observed_max"]:
                notes.append(
                    f"'{name}' = {number:g} is outside the range seen in training "
                    f"({spec['observed_min']:g}-{spec['observed_max']:g}); the estimate is an extrapolation."
                )
            clean[name] = number
        else:
            text = str(value).strip()
            if text not in spec["categories"]:
                problems.append(f"'{name}' must be one of {spec['categories']} (got {text!r}).")
                continue
            clean[name] = text

    names = {spec["name"] for spec in metadata["features"]}
    if config.SYSTOLIC_COLUMN in names and config.DIASTOLIC_COLUMN in names:
        sys_v, dia_v = clean.get(config.SYSTOLIC_COLUMN), clean.get(config.DIASTOLIC_COLUMN)
        if sys_v is not None and dia_v is not None and sys_v <= dia_v:
            problems.append("Systolic blood pressure must be higher than diastolic blood pressure.")

    if problems:
        raise InputValidationError(problems)
    return clean, notes


def build_input_frame(clean_inputs: dict, metadata: dict) -> pd.DataFrame:
    """One-row DataFrame with exactly the training feature names, in training order."""
    order = feature_names(metadata)
    frame = pd.DataFrame([{name: clean_inputs[name] for name in order}], columns=order)
    for spec in metadata["features"]:
        if spec["kind"] == "numeric":
            frame[spec["name"]] = frame[spec["name"]].astype("float64")
        else:
            frame[spec["name"]] = frame[spec["name"]].astype(object)
    return frame


def predict_quality(
    inputs: dict,
    pipeline=None,
    metadata: dict | None = None,
    models_dir: str | Path = config.MODELS_DIR,
    variant: str = "full",
) -> PredictionResult:
    """Validate ``inputs`` and return the model's estimate of the sleep-quality score.

    The displayed value is clipped to the nominal 1-10 scale. Clipping is a presentation
    step only; it is not part of, and says nothing about, model accuracy.
    """
    metadata = metadata if metadata is not None else load_metadata(models_dir, variant)
    pipeline = pipeline if pipeline is not None else load_model(models_dir, variant)

    clean, notes = validate_inputs(inputs, metadata)
    frame = build_input_frame(clean, metadata)

    expected = list(getattr(pipeline, "feature_names_in_", frame.columns))
    if expected != list(frame.columns):
        raise ValueError(
            "Feature mismatch between the saved model and its metadata. "
            f"Model expects {expected}, form provides {list(frame.columns)}. Re-run `python train.py`."
        )

    raw = float(pipeline.predict(frame)[0])
    lo = metadata["target"]["scale_min"]
    hi = metadata["target"]["scale_max"]
    shown = min(max(raw, lo), hi)
    return PredictionResult(
        raw_prediction=raw,
        displayed_prediction=shown,
        was_clipped=shown != raw,
        warnings=notes,
        input_frame=frame,
    )

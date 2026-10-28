import json

import joblib
import pytest

from helpers import write_synthetic_csv
from sleepwise import config
from sleepwise import predict as pred
from sleepwise.train_model import run_training


def _train(tmp_path):
    csv = write_synthetic_csv(tmp_path / "synthetic.csv", n=200)
    run_training(
        data_path=csv,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        processed_dir=tmp_path / "processed",
        make_figures=False,
        run_eda=False,
    )
    return tmp_path / "models"


def _valid_inputs(metadata):
    values = {}
    for spec in metadata["features"]:
        values[spec["name"]] = spec["median"] if spec["kind"] == "numeric" else spec["default"]
    values["Systolic BP"], values["Diastolic BP"] = 125, 80
    return values


def test_training_creates_all_artifacts_and_metrics(tmp_path):
    models_dir = _train(tmp_path)
    assert pred.model_artifacts_available(models_dir)
    metrics = json.loads((tmp_path / "reports" / config.METRICS_FILE).read_text())
    assert set(metrics["test_set"]["selected_model"]) == {"mae", "rmse", "r2"}
    assert metrics["model_selection"]["selected_model"] == config.DEPLOYED_MODEL
    assert metrics["model_selection"]["best_by_cv"] in metrics["model_selection"]["cv_results"]
    assert set(metrics["test_set"]["all_models"]) >= {"Linear Regression", "Neural Network (MLP)"}
    assert "baseline_training_mean" in metrics["test_set"]


def test_training_and_prediction_use_identical_feature_columns(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    pipeline = pred.load_model(models_dir)
    frame = pred.build_input_frame(_valid_inputs(metadata), metadata)
    assert list(pipeline.feature_names_in_) == metadata["feature_order"] == list(frame.columns)
    for banned in ("Person ID", "Quality of Sleep", "Sleep Disorder", "Blood Pressure"):
        assert banned not in frame.columns


def test_prediction_is_finite_and_on_the_target_scale(tmp_path):
    models_dir = _train(tmp_path)
    result = pred.predict_quality(_valid_inputs(pred.load_metadata(models_dir)), models_dir=models_dir)
    assert result.displayed_prediction == result.displayed_prediction  # not NaN
    assert config.TARGET_SCALE_MIN <= result.displayed_prediction <= config.TARGET_SCALE_MAX


def test_prediction_responds_to_inputs(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    pipeline = pred.load_model(models_dir)
    calm = {**_valid_inputs(metadata), "Stress Level": 2, "Sleep Duration": 8.0}
    stressed = {**_valid_inputs(metadata), "Stress Level": 9, "Sleep Duration": 5.5}
    a = pred.predict_quality(calm, pipeline, metadata).raw_prediction
    b = pred.predict_quality(stressed, pipeline, metadata).raw_prediction
    assert a != b  # output is computed, not hard-coded


def test_missing_model_artifacts_raise_helpful_error(tmp_path):
    assert not pred.model_artifacts_available(tmp_path)
    with pytest.raises(pred.ModelNotFoundError, match="python train.py"):
        pred.load_model(tmp_path)
    with pytest.raises(pred.ModelNotFoundError, match="python train.py"):
        pred.load_metadata(tmp_path)


def test_invalid_inputs_are_all_reported(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    bad = {**_valid_inputs(metadata), "Age": 5, "Occupation": "Astronaut", "Systolic BP": 70, "Diastolic BP": 90}
    del bad["Heart Rate"]
    with pytest.raises(pred.InputValidationError) as err:
        pred.validate_inputs(bad, metadata)
    text = " ".join(err.value.problems)
    assert "Age" in text and "Occupation" in text and "Heart Rate" in text and "Systolic" in text


def test_non_numeric_and_nan_inputs_are_rejected(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    for bad_value in ("abc", float("nan"), float("inf"), True):
        with pytest.raises(pred.InputValidationError):
            pred.validate_inputs({**_valid_inputs(metadata), "Age": bad_value}, metadata)


def test_out_of_training_range_gives_warning_not_error(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    spec = next(s for s in metadata["features"] if s["name"] == "Daily Steps")
    inputs = {**_valid_inputs(metadata), "Daily Steps": min(spec["hard_max"], spec["observed_max"] + 5000)}
    _, notes = pred.validate_inputs(inputs, metadata)
    assert any("Daily Steps" in n for n in notes)


def test_saved_pipeline_matches_metadata_model_name(tmp_path):
    models_dir = _train(tmp_path)
    pipeline = joblib.load(models_dir / config.MODEL_FILE)
    metadata = pred.load_metadata(models_dir)
    expected = {"Linear Regression": "LinearRegression", "Decision Tree": "DecisionTreeRegressor",
                "Random Forest": "RandomForestRegressor", "Neural Network (MLP)": "TransformedTargetRegressor",
                "Neural Network (NumPy, from scratch)": "NumpyMLPRegressor"}[metadata["model_name"]]
    assert type(pipeline.named_steps["model"]).__name__ == expected


def test_deployed_model_is_a_neural_network_with_recorded_architecture(tmp_path):
    models_dir = _train(tmp_path)
    metadata = pred.load_metadata(models_dir)
    assert metadata["model_name"] == "Neural Network (MLP)"
    net = metadata["network"]
    assert net["layer_sizes"][-1] == 1 and net["layer_sizes"][0] > 1
    assert net["activation"] == "relu" and net["trainable_parameters"] > 0


def test_duration_only_model_works_with_only_sleep_duration(tmp_path):
    models_dir = _train(tmp_path)
    meta = pred.load_metadata(models_dir, "duration_only")
    assert meta["feature_order"] == ["Sleep Duration"]
    result = pred.predict_quality({"Sleep Duration": 7.5}, models_dir=models_dir, variant="duration_only")
    assert config.TARGET_SCALE_MIN <= result.displayed_prediction <= config.TARGET_SCALE_MAX
    assert list(result.input_frame.columns) == ["Sleep Duration"]


def test_duration_only_prediction_responds_to_duration(tmp_path):
    models_dir = _train(tmp_path)
    short = pred.predict_quality({"Sleep Duration": 5.2}, models_dir=models_dir, variant="duration_only").raw_prediction
    long = pred.predict_quality({"Sleep Duration": 8.8}, models_dir=models_dir, variant="duration_only").raw_prediction
    assert short != long


def test_sleep_duration_outside_interface_range_is_rejected(tmp_path):
    models_dir = _train(tmp_path)
    for bad in (-1, 17, "abc"):
        with pytest.raises(pred.InputValidationError):
            pred.predict_quality({"Sleep Duration": bad}, models_dir=models_dir, variant="duration_only")


def test_missing_duration_only_artifacts_raise_helpful_error(tmp_path):
    assert not pred.model_artifacts_available(tmp_path, "duration_only")
    with pytest.raises(pred.ModelNotFoundError, match="python train.py"):
        pred.load_model(tmp_path, "duration_only")


def test_experiments_are_recorded_in_the_metrics_file(tmp_path):
    _train(tmp_path)
    metrics = json.loads((tmp_path / "reports" / config.METRICS_FILE).read_text())
    exp = metrics["experiments"]
    assert exp["A_duration_only"]["features"] == ["Sleep Duration"]
    assert set(exp["A_duration_only"]["test_set"]["selected_model"]) == {"mae", "rmse", "r2"}
    assert exp["B_multivariable"]["model"] == config.DEPLOYED_MODEL

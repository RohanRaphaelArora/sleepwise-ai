import pandas as pd
import pytest

from helpers import make_synthetic_df, write_synthetic_csv
from sleepwise import config
from sleepwise.data_loader import (
    DatasetError,
    build_schema_report,
    load_raw_csv,
    validate_dataset,
)


def test_loads_csv_and_reports_shape(tmp_path):
    path = write_synthetic_csv(tmp_path / "data.csv", n=50)
    df = load_raw_csv(path)
    assert df.shape == (50, 13)
    report = build_schema_report(df, path)
    assert report["rows"] == 50 and report["columns"] == 13
    assert report["missing_values_total"] == 0
    assert report["sha256"]


def test_sleep_disorder_none_is_a_category_not_missing(tmp_path):
    path = write_synthetic_csv(tmp_path / "data.csv", n=60)
    df = load_raw_csv(path)
    assert "None" in set(df["Sleep Disorder"])
    assert df["Sleep Disorder"].isna().sum() == 0


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetError, match="Dataset not found"):
        load_raw_csv(tmp_path / "does_not_exist.csv")


def test_empty_file_gives_helpful_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetError):
        load_raw_csv(path)


def test_missing_target_column_is_rejected():
    df = make_synthetic_df(30).drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(DatasetError, match="Quality of Sleep"):
        validate_dataset(df)


def test_non_numeric_target_is_rejected():
    df = make_synthetic_df(30)
    df[config.TARGET_COLUMN] = "high"
    with pytest.raises(DatasetError, match="no numeric values"):
        validate_dataset(df)


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    path = write_synthetic_csv(tmp_path / "elsewhere.csv", n=40)
    monkeypatch.setenv(config.DATA_PATH_ENV_VAR, str(path))
    assert load_raw_csv().shape[0] == 40


def test_schema_report_excludes_id_target_and_sleep_disorder():
    report = build_schema_report(make_synthetic_df(40))
    excluded = report["feature_roles"]["excluded"]
    assert {"Person ID", "Quality of Sleep", "Sleep Disorder"} <= set(excluded)
    used = report["feature_roles"]["numeric"] + report["feature_roles"]["categorical"]
    assert not {"Person ID", "Quality of Sleep", "Sleep Disorder"} & set(used)


def test_duplicates_are_reported_ignoring_id():
    df = make_synthetic_df(20)
    df = pd.concat([df, df.assign(**{"Person ID": df["Person ID"] + 1000})], ignore_index=True)
    report = build_schema_report(df)
    assert report["duplicate_rows_exact"] == 0
    assert report["duplicate_rows_ignoring_id"] == 20

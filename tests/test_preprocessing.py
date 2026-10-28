import numpy as np
import pandas as pd
import pytest

from helpers import make_synthetic_df
from sleepwise import config
from sleepwise.preprocessing import (
    build_preprocessor,
    clean_dataframe,
    feature_group_ids,
    get_feature_roles,
    parse_blood_pressure,
    prepare_dataset,
)


def test_blood_pressure_is_split_into_numbers():
    df = pd.DataFrame({"Blood Pressure": ["126/83", "140/90", " 120 / 80 "]})
    out = parse_blood_pressure(df)
    assert out["Systolic BP"].tolist() == [126, 140, 120]
    assert out["Diastolic BP"].tolist() == [83, 90, 80]
    assert "Blood Pressure" not in out.columns


def test_invalid_blood_pressure_becomes_missing_with_warning():
    df = pd.DataFrame({"Blood Pressure": ["120/80", "abc", "130"]})
    with pytest.warns(UserWarning, match="not in the form"):
        out = parse_blood_pressure(df)
    assert out["Systolic BP"].isna().tolist() == [False, True, True]


def test_frame_without_blood_pressure_is_unchanged():
    df = pd.DataFrame({"Age": [30, 40]})
    assert parse_blood_pressure(df).equals(df)


def test_bmi_spellings_are_merged():
    out = clean_dataframe(make_synthetic_df(80))
    assert "Normal Weight" not in set(out["BMI Category"])
    assert "Normal" in set(out["BMI Category"])


def test_feature_roles_split_numeric_and_categorical():
    roles = get_feature_roles(clean_dataframe(make_synthetic_df(60)))
    assert {"Age", "Systolic BP", "Diastolic BP", "Stress Level"} <= set(roles.numeric)
    assert set(roles.categorical) == {"Gender", "Occupation", "BMI Category"}
    for banned in ("Person ID", "Quality of Sleep", "Sleep Disorder"):
        assert banned not in roles.features
        assert banned in roles.excluded


def test_prepare_dataset_never_contains_target_or_id():
    prepared = prepare_dataset(make_synthetic_df(60))
    assert config.TARGET_COLUMN not in prepared.X.columns
    assert config.ID_COLUMN not in prepared.X.columns
    assert len(prepared.X) == len(prepared.y) == 60


def test_rows_with_missing_target_are_dropped_and_counted():
    df = make_synthetic_df(40)
    df.loc[:4, config.TARGET_COLUMN] = np.nan
    prepared = prepare_dataset(df)
    assert prepared.rows_dropped_missing_target == 5
    assert len(prepared.X) == 35


def test_preprocessor_imputes_and_ignores_unseen_categories():
    prepared = prepare_dataset(make_synthetic_df(80))
    X = prepared.X
    pre = build_preprocessor(prepared.roles.numeric, prepared.roles.categorical).fit(X)
    new = X.head(3).copy()
    new.loc[new.index[0], "Age"] = np.nan          # missing number -> median imputed
    new.loc[new.index[1], "Occupation"] = "Astronaut"  # unseen category -> all zeros
    out = pre.transform(new)
    assert not np.isnan(out).any()
    names = list(pre.get_feature_names_out())
    occ_cols = [i for i, n in enumerate(names) if n.startswith("cat__Occupation_")]
    assert out[1, occ_cols].sum() == 0


def test_preprocessor_is_fitted_on_training_data_only():
    X = prepare_dataset(make_synthetic_df(80)).X
    roles = get_feature_roles(clean_dataframe(make_synthetic_df(80)))
    pre = build_preprocessor(roles.numeric, roles.categorical).fit(X.iloc[:40])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], X.iloc[:40][roles.numeric[0]].median(), atol=5)  # sane scale
    # fitting on the first 40 rows must not "see" the other 40
    assert scaler.n_samples_seen_ == 40


def test_identical_rows_share_a_group_id():
    X = prepare_dataset(make_synthetic_df(30)).X
    X = pd.concat([X, X.iloc[:5]], ignore_index=True)
    ids = feature_group_ids(X)
    assert ids[0] == ids[30] and len(set(ids)) == 30

"""Training, model selection and final evaluation.

Protocol (so the test set stays an honest, one-time estimate)
-------------------------------------------------------------
1. Clean data and build features (see ``preprocessing``).
2. Split 80/20 with ``random_state=42``. Rows with IDENTICAL feature values are kept
   on the same side of the split (``GroupShuffleSplit``), because the Kaggle file
   contains many repeated rows and a plain random split would leak them into the test set.
3. Compare candidate models with grouped K-fold cross-validation on the TRAINING
   split only. The model with the lowest mean CV RMSE is selected.
4. Fit the selected pipeline on the training split, evaluate ONCE on the test split.
5. Save the pipeline, metadata (used by the app to build the input form) and reports.
"""

from __future__ import annotations

import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from . import config, diagrams, eda, evaluate, explainability
from .numpy_mlp import NumpyMLPRegressor
from .data_loader import DatasetError, build_schema_report, load_raw_csv, print_summary, save_json, validate_dataset
from .preprocessing import FeatureRoles, build_preprocessor, feature_group_ids, prepare_dataset

LOGGER = logging.getLogger(__name__)

BASELINE_NAME = "Baseline (training mean)"
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def candidate_models() -> dict[str, object]:
    """Fresh, un-fitted estimators. Light regularisation only; no test-set tuning."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=config.RANDOM_STATE, n_jobs=1
        ),
        # Feed-forward neural network: 2 hidden ReLU layers, Adam, L2 penalty, early stopping.
        # The target is standardised for training and un-standardised for prediction.
        "Neural Network (MLP)": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=config.RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        ),
        # Same idea written from scratch with NumPy (manual back-propagation + Adam).
        "Neural Network (NumPy, from scratch)": NumpyMLPRegressor(
            hidden_layer_sizes=(32, 16), learning_rate=0.01, epochs=300, batch_size=32,
            l2=1e-3, random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(model, roles: FeatureRoles) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(roles.numeric, roles.categorical)),
            ("model", model),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series, groups: np.ndarray):
    """Grouped 80/20 split; identical feature rows never straddle train and test."""
    if len(np.unique(groups)) < config.MIN_DISTINCT_ROWS:
        raise DatasetError(
            f"Only {len(np.unique(groups))} distinct feature combinations; at least "
            f"{config.MIN_DISTINCT_ROWS} are needed for a meaningful train/test split."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def _summarise_cv(result: dict) -> dict:
    out = {}
    for key in ("mae", "rmse", "r2"):
        values = -result[f"test_{key}"] if key != "r2" else result["test_r2"]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["folds"] = int(len(result["test_rmse"]))
    return out


def cross_validate_candidates(X_train, y_train, groups_train, roles: FeatureRoles) -> dict[str, dict]:
    """Grouped K-fold CV on the training split for each candidate and the mean baseline."""
    n_groups = len(np.unique(groups_train))
    n_splits = max(2, min(config.CV_FOLDS, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    models = {BASELINE_NAME: DummyRegressor(strategy="mean"), **candidate_models()}
    results = {}
    for name, model in models.items():
        pipe = build_pipeline(model, roles)
        raw = cross_validate(pipe, X_train, y_train, groups=groups_train, cv=cv, scoring=SCORING, n_jobs=1)
        results[name] = _summarise_cv(raw)
        LOGGER.info("CV %-26s RMSE %.3f +/- %.3f | R2 %.3f", name, results[name]["rmse_mean"],
                    results[name]["rmse_std"], results[name]["r2_mean"])
    return results


def select_best_model(cv_results: dict[str, dict]) -> str:
    """Lowest mean cross-validated RMSE among real candidates (baseline is not eligible)."""
    eligible = {k: v for k, v in cv_results.items() if k != BASELINE_NAME}
    return min(eligible, key=lambda k: eligible[k]["rmse_mean"])


def choose_deployed_model(cv_results: dict[str, dict]) -> tuple[str, str]:
    """Return (name, reason). The configured model is used when it exists; "auto" = best CV."""
    best = select_best_model(cv_results)
    configured = config.DEPLOYED_MODEL
    if configured == "auto" or configured not in cv_results or configured == BASELINE_NAME:
        return best, "lowest mean cross-validated RMSE on the training split"
    if configured == best:
        return configured, "configured as the deployed model and also the lowest cross-validated RMSE"
    return configured, (
        f"configured as the deployed model because this is a neural-network project; "
        f"the lowest cross-validated RMSE was {best} ({cv_results[best]['rmse_mean']:.3f} vs "
        f"{cv_results[configured]['rmse_mean']:.3f}), so the choice is a design decision, not the CV winner"
    )


def network_summary(pipeline: Pipeline) -> dict | None:
    """Architecture + training facts for neural-network models (None for other models)."""
    model = pipeline.named_steps["model"]
    inner = getattr(model, "regressor_", model)  # unwrap TransformedTargetRegressor
    n_in = int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))
    if hasattr(inner, "hidden_layer_sizes") and hasattr(inner, "coefs_"):  # scikit-learn MLP
        hidden = [int(h) for h in inner.hidden_layer_sizes]
        return {
            "implementation": "scikit-learn MLPRegressor",
            "layer_sizes": [n_in, *hidden, 1],
            "activation": inner.activation, "output_activation": "identity (linear)",
            "optimizer": inner.solver, "l2_alpha": inner.alpha,
            "learning_rate_init": inner.learning_rate_init,
            "early_stopping": bool(inner.early_stopping),
            "epochs_run": int(inner.n_iter_),
            "best_validation_r2": None if getattr(inner, "best_validation_score_", None) is None
            else float(inner.best_validation_score_),
            "trainable_parameters": int(sum(c.size for c in inner.coefs_) + sum(b.size for b in inner.intercepts_)),
        }
    if hasattr(inner, "params_"):  # NumPy from-scratch network
        hidden = [int(h) for h in inner.hidden_layer_sizes]
        return {
            "implementation": "NumPy from scratch", "layer_sizes": [n_in, *hidden, 1],
            "activation": "relu", "output_activation": "identity (linear)", "optimizer": "adam (manual)",
            "l2_alpha": inner.l2, "learning_rate_init": inner.learning_rate, "early_stopping": False,
            "epochs_run": int(len(inner.loss_curve_)), "best_validation_r2": None,
            "trainable_parameters": int(sum(p.size for p in inner.params_)),
        }
    return None


def _feature_metadata(X_train: pd.DataFrame, roles: FeatureRoles) -> list[dict]:
    """Per-feature description used by the app (widget types, ranges, categories)."""
    specs: list[dict] = []
    for col in roles.numeric:
        s = X_train[col].dropna()
        hard = config.PLAUSIBLE_RANGES.get(col, (float(s.min()), float(s.max())))
        specs.append(
            {
                "name": col,
                "kind": "numeric",
                "unit": config.UNITS.get(col, ""),
                "observed_min": float(s.min()),
                "observed_max": float(s.max()),
                "median": float(s.median()),
                "is_integer": bool(np.allclose(s, np.round(s))),
                "hard_min": float(hard[0]),
                "hard_max": float(hard[1]),
            }
        )
    for col in roles.categorical:
        s = X_train[col].dropna()
        counts = s.value_counts()
        specs.append(
            {
                "name": col,
                "kind": "categorical",
                "categories": sorted(str(c) for c in counts.index),
                "default": str(counts.index[0]),
                "category_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
    return specs


def _figure_sets(figures_dir: Path):
    """(theme, folder) pairs: light figures for documents, dark figures for the dashboard."""
    return (("light", Path(figures_dir)), ("dark", Path(figures_dir) / "dark"))


def run_duration_only_experiment(X_train, X_test, y_train, y_test, groups_train, models_dir: Path,
                                 figures_dir: Path | None) -> dict | None:
    """Experiment A: use `Sleep Duration` as the ONLY predictor.

    Same rows, same split and same candidates as the main run. The model with the lowest
    cross-validated RMSE (training split only) is fitted, saved and tested once.
    Returns None if the dataset has no sleep-duration column.
    """
    col = config.SLEEP_DURATION_COLUMN
    if col not in X_train.columns:
        return None
    roles = FeatureRoles(numeric=[col], categorical=[])
    Xa_train, Xa_test = X_train[[col]], X_test[[col]]
    cv_results = cross_validate_candidates(Xa_train, y_train, groups_train, roles)
    best = select_best_model(cv_results)
    pipe = build_pipeline(candidate_models()[best], roles).fit(Xa_train, y_train)
    pred = pipe.predict(Xa_test)
    test_metrics = evaluate.regression_metrics(y_test, pred)
    baseline = evaluate.regression_metrics(y_test, np.full(len(y_test), float(y_train.mean())))
    joblib.dump(pipe, models_dir / config.DURATION_MODEL_FILE)
    meta = {
        "model_name": best,
        "variant": "duration_only",
        "target": {"name": config.TARGET_COLUMN, "observed_min": float(y_train.min()), "observed_max": float(y_train.max()),
                   "scale_min": config.TARGET_SCALE_MIN, "scale_max": config.TARGET_SCALE_MAX},
        "features": _feature_metadata(Xa_train, roles),
        "feature_order": [col],
        "network": network_summary(pipe),
        "sklearn_version": sklearn.__version__,
    }
    save_json(meta, models_dir / config.DURATION_METADATA_FILE)
    if figures_dir is not None:
        for theme, out in _figure_sets(figures_dir):
            config.set_figure_theme(theme)
            evaluate.plot_actual_vs_predicted(y_test, pred, out / "actual_vs_predicted_duration_only.png",
                                              f"Sleep duration only - {best} (test set)")
        config.set_figure_theme("light")
    return {
        "features": [col],
        "cv_results": cv_results,
        "selected_model": best,
        "selection_criterion": "lowest mean cross-validated RMSE on the training split",
        "test_set": {"selected_model": test_metrics, "baseline_training_mean": baseline},
        "distinct_input_values_in_training": int(Xa_train[col].nunique()),
    }


def run_training(
    data_path: str | Path | None = None,
    models_dir: str | Path = config.MODELS_DIR,
    reports_dir: str | Path = config.REPORTS_DIR,
    processed_dir: str | Path = config.PROCESSED_DIR,
    make_figures: bool = True,
    run_eda: bool = True,
) -> dict:
    """Run the whole workflow and write all artefacts. Returns the metrics dictionary."""
    models_dir, reports_dir, processed_dir = Path(models_dir), Path(reports_dir), Path(processed_dir)
    figures_dir = reports_dir / "figures"

    # 1. Load + validate ---------------------------------------------------- #
    raw = load_raw_csv(data_path)
    validate_dataset(raw)
    schema = build_schema_report(raw, data_path)
    print_summary(schema)
    save_json(schema, reports_dir / config.SCHEMA_FILE)

    prepared = prepare_dataset(raw)
    X, y, roles = prepared.X, prepared.y, prepared.roles
    if prepared.rows_dropped_missing_target:
        LOGGER.warning("Dropped %d rows with a missing target.", prepared.rows_dropped_missing_target)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared.clean_df.to_csv(processed_dir / config.CLEANED_CSV_FILE, index=False)

    # 2. EDA ---------------------------------------------------------------- #
    if run_eda:
        save_json(eda.build_eda_summary(raw, prepared.clean_df), reports_dir / config.EDA_SUMMARY_FILE)
        if make_figures:
            for theme, out in _figure_sets(figures_dir):
                config.set_figure_theme(theme)
                eda.generate_eda_figures(prepared.clean_df, out)
            config.set_figure_theme("light")

    # 3. Split -------------------------------------------------------------- #
    groups = feature_group_ids(X)
    train_idx, test_idx = split_data(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

    # 4. Cross-validated comparison (training split only) -------------------- #
    cv_results = cross_validate_candidates(X_train, y_train, groups_train, roles)
    best_by_cv = select_best_model(cv_results)
    deployed_name, deployed_reason = choose_deployed_model(cv_results)
    LOGGER.info("Best by CV: %s | deployed model: %s", best_by_cv, deployed_name)

    # 5. Fit every candidate on the training split; evaluate on the test split once.
    #    Only the deployed model is saved. Test metrics of the others are reported for
    #    transparency and do NOT influence which model is deployed.
    fitted = {name: build_pipeline(model, roles).fit(X_train, y_train) for name, model in candidate_models().items()}
    baseline_pred = np.full(len(y_test), float(y_train.mean()))
    baseline_metrics = evaluate.regression_metrics(y_test, baseline_pred)
    test_all = {name: evaluate.regression_metrics(y_test, pipe.predict(X_test)) for name, pipe in fitted.items()}

    final_pipe = fitted[deployed_name]
    test_pred = final_pipe.predict(X_test)
    train_pred = final_pipe.predict(X_train)
    test_metrics = test_all[deployed_name]
    train_metrics = evaluate.regression_metrics(y_train, train_pred)

    # Diagnostic ONLY (does not influence selection): what a naive random split would report.
    n_tr, n_te = train_test_split(np.arange(len(X)), test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    naive_pipe = clone(final_pipe).fit(X.iloc[n_tr], y.iloc[n_tr])
    naive_metrics = evaluate.regression_metrics(y.iloc[n_te], naive_pipe.predict(X.iloc[n_te]))
    twin_share = float(np.mean(np.isin(groups[n_te], groups[n_tr])))

    # 5b. Experiment A: sleep duration only -------------------------------- #
    models_dir.mkdir(parents=True, exist_ok=True)
    experiment_a = run_duration_only_experiment(X_train, X_test, y_train, y_test, groups_train, models_dir,
                                                figures_dir if make_figures else None)

    # 6. Plots + explainability --------------------------------------------- #
    importance = explainability.compute_permutation_importance(final_pipe, X_test, y_test)
    native = explainability.native_importance(final_pipe, roles.categorical)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(reports_dir / config.IMPORTANCE_FILE, index=False)
    arch = network_summary(final_pipe)
    if arch:
        save_json(arch, reports_dir / config.ARCHITECTURE_FILE)
    if make_figures:
        for theme, out in _figure_sets(figures_dir):
            config.set_figure_theme(theme)
            evaluate.plot_actual_vs_predicted(y_test, test_pred, out / "actual_vs_predicted.png",
                                              f"Actual vs predicted sleep quality - {deployed_name} (test set)")
            evaluate.plot_residuals(y_test, test_pred, out / "residuals.png",
                                    f"Residual analysis - {deployed_name} (test set)")
            evaluate.plot_model_comparison(cv_results, out / "model_comparison.png")
            explainability.plot_feature_importance(
                importance, out / "feature_importance.png",
                f"Permutation feature importance - {deployed_name} (test set)")
            diagrams.plot_workflow(out / "system_architecture.png")
            if experiment_a:
                evaluate.plot_experiment_comparison(
                    [("Mean baseline", baseline_metrics), ("A: duration only", experiment_a["test_set"]["selected_model"]),
                     ("B: all features", test_metrics)], out / "experiment_comparison.png")
            if arch:
                labels = ["Input\n(preprocessed)"] + [f"Hidden {i + 1}\n({arch['activation'].upper()})"
                                                      for i in range(len(arch["layer_sizes"]) - 2)] + ["Output\n(linear)"]
                diagrams.plot_network_architecture(
                    arch["layer_sizes"], labels, out / "network_architecture.png",
                    f"{arch['implementation']}: {' - '.join(str(n) for n in arch['layer_sizes'])} "
                    f"({arch['trainable_parameters']:,} trainable parameters)")
                evaluate.plot_training_curve(final_pipe, out / "training_curve.png")
        config.set_figure_theme("light")

    # 7. Persist model + metadata ------------------------------------------- #
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, models_dir / config.MODEL_FILE)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "model_name": deployed_name,
        "trained_at_utc": trained_at,
        "target": {
            "name": config.TARGET_COLUMN,
            "observed_min": float(y_train.min()),
            "observed_max": float(y_train.max()),
            "scale_min": config.TARGET_SCALE_MIN,
            "scale_max": config.TARGET_SCALE_MAX,
        },
        "features": _feature_metadata(X_train, roles),
        "feature_order": list(X.columns),
        "network": arch,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
    }
    save_json(metadata, models_dir / config.METADATA_FILE)

    metrics = {
        "generated_at_utc": trained_at,
        "dataset": {
            "file": Path(schema["file"]).name,
            "sha256": schema["sha256"],
            "rows_in_file": schema["rows"],
            "rows_used": int(len(X)),
            "rows_dropped_missing_target": prepared.rows_dropped_missing_target,
            "columns_in_file": schema["columns"],
            "duplicate_rows_ignoring_id": schema["duplicate_rows_ignoring_id"],
            "distinct_feature_combinations": int(len(np.unique(groups))),
        },
        "features": roles.to_dict(),
        "split": {
            "strategy": "GroupShuffleSplit (identical feature rows kept together)",
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_distinct_feature_combinations": int(len(np.unique(groups[train_idx]))),
            "test_distinct_feature_combinations": int(len(np.unique(groups[test_idx]))),
        },
        "model_selection": {
            "criterion": "lowest mean cross-validated RMSE on the training split",
            "cv": f"GroupKFold, {next(iter(cv_results.values()))['folds']} folds, training split only",
            "cv_results": cv_results,
            "best_by_cv": best_by_cv,
            "selected_model": deployed_name,
            "deployed_reason": deployed_reason,
        },
        "test_set": {
            "note": "Evaluated once, after the deployed model was fixed, on rows never used for fitting or selection.",
            "selected_model": test_metrics,
            "baseline_training_mean": baseline_metrics,
            "all_models": test_all,
            "all_models_note": ("Shown for transparency only. These numbers did not influence which model is "
                                "deployed. With 22 distinct test combinations, small differences are not reliable."),
        },
        "training_set_fit": train_metrics,
        "experiments": {
            "A_duration_only": experiment_a,
            "B_multivariable": {
                "features": roles.features,
                "model": deployed_name,
                "cv": cv_results[deployed_name],
                "test_set": {"selected_model": test_metrics, "baseline_training_mean": baseline_metrics},
            },
            "note": ("Both experiments use the same rows, the same grouped train/test split and the same "
                     "cross-validation. Experiment A keeps only Sleep Duration as input."),
        },
        "network": arch,
        "diagnostics": {
            "naive_random_split_80_20": {
                "note": ("Same model type, plain random split WITHOUT grouping. Shown only to illustrate "
                         "optimistic bias from repeated rows; not used for selection or reporting."),
                "metrics": naive_metrics,
                "share_of_test_rows_with_identical_feature_twin_in_train": twin_share,
            },
            "native_feature_importance": None if native is None else native.to_dict(orient="records"),
        },
        "target_scale": {
            "observed_in_training": [float(y_train.min()), float(y_train.max())],
            "nominal_scale": [config.TARGET_SCALE_MIN, config.TARGET_SCALE_MAX],
            "clipping": ("Predictions shown in the app are clipped to the nominal 1-10 scale. "
                         "Metrics above use the raw, unclipped predictions."),
        },
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "caveats": [
            "Small dataset: estimates are uncertain; see cross-validation standard deviations.",
            "Many rows repeat identical feature values, so effective sample size is the number of distinct feature combinations.",
            "Estimates the recorded sleep-quality score from other recorded characteristics; not a forecast or diagnosis.",
        ],
    }
    save_json(metrics, reports_dir / config.METRICS_FILE)
    LOGGER.info("Deployed %s | test RMSE %.3f | MAE %.3f | R2 %.3f (baseline RMSE %.3f)", deployed_name,
                test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"], baseline_metrics["rmse"])
    return metrics

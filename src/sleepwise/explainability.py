"""Model-reliance explanations (not causal explanations).

Two views are provided:

* **Permutation importance** (primary): shuffle one *original* input column on the
  held-out rows and measure how much the error (RMSE) rises. Works for any model.
* **Native importance** (Random Forest / Decision Tree only): impurity-based
  importance, summed back to the original input columns.

IMPORTANT: both describe what the *trained model relies on*. They do not show that
changing a feature would change a person's sleep quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance

from . import config

EXPLAINABILITY_NOTE = (
    "Feature importance shows how much the trained model relies on each input to "
    "reproduce the recorded scores in this dataset. It does not show that changing an "
    "input would change anyone's sleep quality, and it is not medical advice."
)


def compute_permutation_importance(
    pipeline, X: pd.DataFrame, y, n_repeats: int = 30, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Increase in RMSE when each original input column is shuffled (higher = more relied on)."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def native_importance(pipeline, categorical: list[str]) -> pd.DataFrame | None:
    """Impurity importance aggregated per original column, or None if the model has none."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    totals: dict[str, float] = {}
    for name, value in zip(names, model.feature_importances_):
        if name.startswith("num__"):
            original = name[len("num__"):]
        else:
            body = name[len("cat__"):]
            matches = [c for c in categorical if body.startswith(f"{c}_")]
            original = max(matches, key=len) if matches else body
        totals[original] = totals.get(original, 0.0) + float(value)
    df = pd.DataFrame({"feature": list(totals), "importance": list(totals.values())})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Ranked horizontal bar chart of permutation importance with std whiskers."""
    data = importance.sort_values("importance_mean", ascending=True)
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7.5, max(3.8, 0.42 * len(data) + 1.4)))
        ax = fig.subplots()
        ax.barh(
            data["feature"],
            data["importance_mean"],
            xerr=data["importance_std"],
            color=config.TEAL,
            edgecolor=config.FIG_BG,
            capsize=3,
        )
        ax.axvline(0, color=config.NAVY, linewidth=0.8)
        ax.set_title(title, color=config.NAVY, fontsize=12, fontweight="bold")
        ax.set_xlabel("Increase in RMSE when the input is shuffled (model reliance, not causation)")
        ax.set_ylabel("Input feature")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=config.FIG_BG)
    return path

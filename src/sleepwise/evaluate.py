"""Regression metrics and evaluation plots.

Figures are built with ``matplotlib.figure.Figure`` objects (not ``pyplot``) so they
can be created on any machine/thread without a GUI backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R-squared. RMSE is computed explicitly (portable across versions)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=config.FIG_BG)
    return path


def plot_actual_vs_predicted(y_true, y_pred, path: str | Path, title: str) -> Path:
    """Scatter of recorded vs predicted scores (small horizontal jitter, targets are integers)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(config.RANDOM_STATE)
    jitter = rng.uniform(-0.12, 0.12, size=len(y_true))

    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(6.4, 5.6))
        ax = fig.subplots()
        ax.scatter(y_true + jitter, y_pred, s=34, alpha=0.65, color=config.TEAL, edgecolor=config.FIG_BG)
        lo = min(y_true.min(), y_pred.min()) - 0.5
        hi = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color=config.NAVY, linewidth=1.2, label="Perfect prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, color=config.NAVY, fontsize=12, fontweight="bold")
        ax.set_xlabel("Recorded sleep quality (held-out test rows, jittered for visibility)")
        ax.set_ylabel("Predicted sleep quality")
        ax.legend(loc="upper left", frameon=False)
    return _save(fig, path)


def plot_residuals(y_true, y_pred, path: str | Path, title: str) -> Path:
    """Residuals (recorded - predicted) against predictions, plus their histogram."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(10, 4.6))
        ax1, ax2 = fig.subplots(1, 2)
        ax1.scatter(y_pred, residuals, s=34, alpha=0.65, color=config.TEAL, edgecolor=config.FIG_BG)
        ax1.axhline(0, color=config.NAVY, linestyle="--", linewidth=1.2)
        ax1.set_title("Residuals vs predicted value", color=config.NAVY, fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted sleep quality")
        ax1.set_ylabel("Residual (recorded - predicted)")

        ax2.hist(residuals, bins=15, color=config.TEAL, edgecolor=config.FIG_BG)
        ax2.axvline(0, color=config.NAVY, linestyle="--", linewidth=1.2)
        ax2.set_title("Distribution of residuals", color=config.NAVY, fontsize=11, fontweight="bold")
        ax2.set_xlabel("Residual (recorded - predicted)")
        ax2.set_ylabel("Number of test rows")
        fig.suptitle(title, color=config.NAVY, fontsize=12, fontweight="bold", y=1.02)
    return _save(fig, path)


def plot_model_comparison(cv_results: dict[str, dict], path: str | Path) -> Path:
    """Bar chart of cross-validated RMSE (training data only) with a mean-baseline bar."""
    names = list(cv_results)
    means = [cv_results[n]["rmse_mean"] for n in names]
    stds = [cv_results[n]["rmse_std"] for n in names]
    colors = [config.SLATE if n.lower().startswith("baseline") else config.TEAL for n in names]

    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7.5, 4.6))
        ax = fig.subplots()
        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor=config.FIG_BG)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(
            "Cross-validated RMSE on the training split (lower is better)",
            color=config.NAVY,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE (sleep-quality points); bars = mean, whiskers = std across folds")
        ax.tick_params(axis="x", rotation=15)
    return _save(fig, path)


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Small helper used by the app: dict of metrics -> tidy DataFrame."""
    return pd.DataFrame(results).T.rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"})


def plot_training_curve(pipeline, path: str | Path) -> Path | None:
    """Loss (and validation R-squared when available) against training epoch for a neural network."""
    model = pipeline.named_steps["model"]
    inner = getattr(model, "regressor_", model)
    loss = getattr(inner, "loss_curve_", None)
    if loss is None:
        return None
    val = getattr(inner, "validation_scores_", None)
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7.5, 4.6))
        ax = fig.subplots()
        ax.plot(range(1, len(loss) + 1), loss, color=config.TEAL, linewidth=2, label="Training loss")
        ax.set_xlabel("Epoch (full passes over the training rows)")
        ax.set_ylabel("Training loss (standardised target)")
        ax.set_yscale("log")
        if val is not None:
            ax2 = ax.twinx()
            ax2.plot(range(1, len(val) + 1), val, color=config.NAVY, linewidth=1.6, linestyle="--",
                     label="Validation R² (early stopping)")
            ax2.set_ylabel("Validation R²")
            ax2.grid(False)
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=False)
        else:
            ax.legend(frameon=False)
        ax.set_title("Neural-network training curve", color=config.NAVY, fontsize=12, fontweight="bold")
    return _save(fig, path)


def plot_experiment_comparison(rows: list[tuple[str, dict]], path: str | Path) -> Path:
    """Compare held-out RMSE and R-squared for baseline, Experiment A and Experiment B."""
    names = [r[0] for r in rows]
    rmse = [r[1]["rmse"] for r in rows]
    r2 = [r[1]["r2"] for r in rows]
    colors = [config.SLATE, config.NAVY, config.TEAL][: len(rows)]
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(10, 4.6))
        ax1, ax2 = fig.subplots(1, 2)
        for ax, vals, label, fmt in ((ax1, rmse, "RMSE on held-out rows (lower is better)", "{:.3f}"),
                                      (ax2, r2, "R² on held-out rows (higher is better)", "{:.3f}")):
            bars = ax.bar(names, vals, color=colors, edgecolor=config.FIG_BG)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)
            ax.set_ylabel(label)
            ax.set_xlabel("Model setup")
            ax.tick_params(axis="x", rotation=12)
        fig.suptitle("Experiment A (sleep duration only) versus Experiment B (all features)",
                     color=config.NAVY, fontsize=12, fontweight="bold", y=1.02)
    return _save(fig, path)

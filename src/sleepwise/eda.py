"""Exploratory data analysis: summary statistics and figures.

Every figure has a title and labelled axes. Associations shown here are descriptive;
they do not establish causation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype

from . import config

TARGET = config.TARGET_COLUMN

# figure key -> (file name, caption shown in the app)
FIGURE_CAPTIONS: dict[str, tuple[str, str]] = {
    "target_distribution": ("01_target_distribution.png", "How often each sleep-quality score appears in the dataset."),
    "sleep_duration_distribution": ("02_sleep_duration_distribution.png", "Distribution of recorded sleep duration."),
    "stress_distribution": ("03_stress_distribution.png", "How many people report each stress level."),
    "quality_vs_stress": ("04_quality_vs_stress.png", "Sleep quality at each stress level (boxes) with group means (diamonds)."),
    "quality_vs_activity": ("05_quality_vs_activity.png", "Sleep quality against physical-activity level (jittered; line is a linear fit)."),
    "quality_vs_duration": ("06_quality_vs_duration.png", "Sleep quality against sleep duration (jittered; line is a linear fit)."),
    "correlation_heatmap": ("07_correlation_heatmap.png", "Pearson correlations between numerical columns. Correlation is not causation."),
    "quality_by_bmi": ("08_quality_by_bmi.png", "Sleep quality by BMI category."),
    "quality_by_gender": ("09_quality_by_gender.png", "Sleep quality by gender."),
    "quality_by_occupation": ("10_quality_by_occupation.png", "Mean sleep quality by occupation; n = number of rows (small groups are unreliable)."),
    "quality_by_sleep_disorder": ("11_quality_by_sleep_disorder.png", "Sleep quality by recorded sleep disorder (context only; NOT a model input)."),
}


def _title(ax, text: str) -> None:
    ax.set_title(text, color=config.NAVY, fontsize=12, fontweight="bold")


def _save(fig: Figure, out_dir: Path, key: str) -> Path:
    path = Path(out_dir) / FIGURE_CAPTIONS[key][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=config.FIG_BG)
    return path


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def build_eda_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Shape, dtypes, descriptive statistics, missing values, duplicates, correlations."""
    numeric = [c for c in clean_df.columns if is_numeric_dtype(clean_df[c]) and c != config.ID_COLUMN]
    summary: dict = {
        "shape": {"rows": int(raw_df.shape[0]), "columns": int(raw_df.shape[1])},
        "dtypes": {c: str(t) for c, t in raw_df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in raw_df.isna().sum().items()},
        "duplicates": {
            "exact_rows": int(raw_df.duplicated().sum()),
            "ignoring_person_id": int(
                raw_df.drop(columns=[c for c in [config.ID_COLUMN] if c in raw_df.columns]).duplicated().sum()
            ),
        },
        "describe_numeric": clean_df[numeric].describe().round(3).to_dict(),
    }
    if TARGET in clean_df.columns:
        corr = clean_df[numeric].corr()[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
        summary["pearson_correlation_with_target"] = {k: round(float(v), 3) for k, v in corr.items()}
    return summary


# --------------------------------------------------------------------------- #
# Individual figures (each returns the saved path, or None if columns are absent)
# --------------------------------------------------------------------------- #
def _histogram(df, col, key, title, xlabel, out_dir, discrete=False):
    if col not in df.columns:
        return None
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7, 4.4))
        ax = fig.subplots()
        sns.histplot(df[col].dropna(), discrete=discrete, bins=None if discrete else 15,
                     color=config.TEAL, edgecolor=config.FIG_BG, ax=ax)
        _title(ax, title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of people (rows)")
    return _save(fig, out_dir, key)


def _scatter_with_fit(df, x, key, title, xlabel, out_dir):
    if x not in df.columns or TARGET not in df.columns:
        return None
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7, 4.8))
        ax = fig.subplots()
        sns.regplot(
            data=df, x=x, y=TARGET, x_jitter=0.02 * float(df[x].std() or 1), y_jitter=0.12,
            scatter_kws={"alpha": 0.45, "s": 26, "color": config.TEAL},
            line_kws={"color": config.NAVY, "linewidth": 2}, ci=None, ax=ax,
        )
        _title(ax, title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Quality of sleep (score, jittered)")
    return _save(fig, out_dir, key)


def _target_by_category(df, col, key, title, xlabel, out_dir, order=None):
    if col not in df.columns or TARGET not in df.columns:
        return None
    with sns.axes_style("whitegrid", rc=config.FIG_RC):
        fig = Figure(figsize=(7, 4.6))
        ax = fig.subplots()
        sns.boxplot(data=df, x=col, y=TARGET, order=order, color=config.TEAL, ax=ax,
                    linecolor=config.NAVY, fliersize=3)
        _title(ax, title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Quality of sleep (score)")
    return _save(fig, out_dir, key)


def generate_eda_figures(clean_df: pd.DataFrame, out_dir: str | Path = config.FIGURES_DIR) -> dict[str, Path]:
    """Create every EDA figure that the available columns allow. Returns {key: path}."""
    out_dir = Path(out_dir)
    df = clean_df
    figures: dict[str, Path | None] = {}

    figures["target_distribution"] = _histogram(
        df, TARGET, "target_distribution", "Distribution of sleep-quality scores",
        "Quality of sleep (score)", out_dir, discrete=True)
    figures["sleep_duration_distribution"] = _histogram(
        df, config.SLEEP_DURATION_COLUMN, "sleep_duration_distribution", "Distribution of sleep duration",
        "Sleep duration (hours per night)", out_dir)
    figures["stress_distribution"] = _histogram(
        df, config.STRESS_COLUMN, "stress_distribution", "Distribution of stress levels",
        "Stress level (scale 1-10)", out_dir, discrete=True)

    if config.STRESS_COLUMN in df.columns and TARGET in df.columns:
        with sns.axes_style("whitegrid", rc=config.FIG_RC):
            fig = Figure(figsize=(7, 4.8))
            ax = fig.subplots()
            sns.boxplot(data=df, x=config.STRESS_COLUMN, y=TARGET, color=config.TEAL, ax=ax,
                        linecolor=config.NAVY, fliersize=3)
            means = df.groupby(config.STRESS_COLUMN)[TARGET].mean()
            positions = sorted(df[config.STRESS_COLUMN].dropna().unique())
            ax.scatter(range(len(positions)), [means[p] for p in positions], marker="D",
                       color=config.NAVY, zorder=5, label="Group mean")
            ax.legend(frameon=False, loc="upper right")
            _title(ax, "Sleep quality versus stress level")
            ax.set_xlabel("Stress level (scale 1-10)")
            ax.set_ylabel("Quality of sleep (score)")
        figures["quality_vs_stress"] = _save(fig, out_dir, "quality_vs_stress")

    figures["quality_vs_activity"] = _scatter_with_fit(
        df, config.ACTIVITY_COLUMN, "quality_vs_activity",
        "Sleep quality versus physical-activity level", "Physical activity level (dataset units)", out_dir)
    figures["quality_vs_duration"] = _scatter_with_fit(
        df, config.SLEEP_DURATION_COLUMN, "quality_vs_duration",
        "Sleep quality versus sleep duration", "Sleep duration (hours per night)", out_dir)

    numeric = [c for c in df.columns if is_numeric_dtype(df[c]) and c != config.ID_COLUMN]
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        with sns.axes_style("white", rc=config.FIG_RC):
            size = max(7.0, 0.85 * len(numeric))
            fig = Figure(figsize=(size, size * 0.85))
            ax = fig.subplots()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                        linewidths=0.5, annot_kws={"size": 8}, cbar_kws={"label": "Pearson correlation"}, ax=ax)
            _title(ax, "Correlation heatmap of numerical columns")
            ax.set_xlabel("Numerical column")
            ax.set_ylabel("Numerical column")
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        figures["correlation_heatmap"] = _save(fig, out_dir, "correlation_heatmap")

    bmi_order = [c for c in ["Normal", "Overweight", "Obese"] if c in set(df.get(config.BMI_COLUMN, []))]
    figures["quality_by_bmi"] = _target_by_category(
        df, config.BMI_COLUMN, "quality_by_bmi", "Sleep quality by BMI category", "BMI category",
        out_dir, order=bmi_order or None)
    figures["quality_by_gender"] = _target_by_category(
        df, config.GENDER_COLUMN, "quality_by_gender", "Sleep quality by gender", "Gender", out_dir)

    if config.OCCUPATION_COLUMN in df.columns and TARGET in df.columns:
        grouped = df.groupby(config.OCCUPATION_COLUMN)[TARGET].agg(["mean", "count"]).sort_values("mean")
        labels = [f"{idx} (n={int(row['count'])})" for idx, row in grouped.iterrows()]
        with sns.axes_style("whitegrid", rc=config.FIG_RC):
            fig = Figure(figsize=(7.5, max(3.8, 0.42 * len(grouped) + 1.4)))
            ax = fig.subplots()
            ax.barh(labels, grouped["mean"], color=config.TEAL, edgecolor=config.FIG_BG)
            _title(ax, "Mean sleep quality by occupation")
            ax.set_xlabel("Mean quality of sleep (score)")
            ax.set_ylabel("Occupation (n = rows)")
        figures["quality_by_occupation"] = _save(fig, out_dir, "quality_by_occupation")

    disorder_order = [c for c in ["None", "Insomnia", "Sleep Apnea"] if c in set(df.get(config.SLEEP_DISORDER_COLUMN, []))]
    figures["quality_by_sleep_disorder"] = _target_by_category(
        df, config.SLEEP_DISORDER_COLUMN, "quality_by_sleep_disorder",
        "Sleep quality by recorded sleep disorder (context only, not a model input)",
        "Recorded sleep disorder", out_dir, order=disorder_order or None)

    return {k: v for k, v in figures.items() if v is not None}

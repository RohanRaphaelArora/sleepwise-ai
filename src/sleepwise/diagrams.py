"""Architecture diagrams drawn with Matplotlib (no Graphviz or other tools required)."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config

MAX_NODES_SHOWN = 8


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=config.FIG_BG)
    return path


def plot_network_architecture(layer_sizes: list[int], layer_labels: list[str], path: str | Path, title: str) -> Path:
    """Fully-connected network diagram. Layers larger than 8 nodes are drawn with an ellipsis."""
    fig = Figure(figsize=(9.5, 5.4))
    ax = fig.subplots()
    ax.set_axis_off()
    xs = [i * 2.4 for i in range(len(layer_sizes))]
    positions: list[list[tuple[float, float]]] = []
    for x, size in zip(xs, layer_sizes):
        shown = min(size, MAX_NODES_SHOWN)
        ys = [(shown - 1) / 2 - j for j in range(shown)]
        positions.append([(x, y) for y in ys])

    for left, right in zip(positions[:-1], positions[1:]):
        for (x1, y1) in left:
            for (x2, y2) in right:
                ax.plot([x1, x2], [y1, y2], color=config.FIG_LINE, linewidth=0.5, zorder=1)
    for layer, size in zip(positions, layer_sizes):
        for (x, y) in layer:
            ax.scatter([x], [y], s=210, color=config.TEAL, edgecolor=config.NAVY, linewidth=1.0, zorder=3)
        if size > MAX_NODES_SHOWN:
            ax.text(layer[0][0], layer[-1][1] - 0.85, f"{MAX_NODES_SHOWN} of {size} nodes shown", ha="center", fontsize=9, color=config.NAVY)
    top = max(len(p) for p in positions) / 2 + 0.6
    for x, label in zip(xs, layer_labels):
        ax.text(x, top + 0.5, label, ha="center", va="bottom", fontsize=10, color=config.NAVY, fontweight="bold")
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_ylim(-top - 1.3, top + 1.6)
    ax.set_title(title, color=config.NAVY, fontsize=12, fontweight="bold", y=1.0)
    return _save(fig, path)


def plot_workflow(path: str | Path) -> Path:
    """System architecture: data -> cleaning -> split -> models -> evaluation -> saved model -> app."""
    steps = [
        ("Kaggle CSV\n(374 x 13)", "data/raw"),
        ("Validate + clean\nsplit blood pressure", "data_loader\npreprocessing"),
        ("Grouped 80/20\nsplit (seed 42)", "train_model"),
        ("Pipeline: impute,\nscale, one-hot", "ColumnTransformer"),
        ("Compare 5 models\n5-fold grouped CV", "train_model"),
        ("Neural network\nfit + test once", "evaluate"),
        ("Saved model\n+ metadata + reports", "models/ reports/"),
        ("Streamlit app\nPredict + dashboards", "app.py"),
    ]
    fig = Figure(figsize=(12.5, 4.6))
    ax = fig.subplots()
    ax.set_axis_off()
    cols = 4
    box_w, box_h, gap_x, gap_y = 2.5, 1.15, 0.65, 1.15
    coords = []
    for i in range(len(steps)):
        row, col = divmod(i, cols)
        if row % 2 == 1:
            col = cols - 1 - col  # snake layout so arrows flow naturally
        coords.append((col * (box_w + gap_x), -row * (box_h + gap_y)))
    for (x, y), (label, module) in zip(coords, steps):
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=config.FIG_SURFACE, edgecolor=config.NAVY, linewidth=1.3))
        ax.text(x + box_w / 2, y + box_h * 0.60, label, ha="center", va="center", fontsize=9.5, color=config.NAVY, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h * 0.14, module, ha="center", va="center", fontsize=7.5, color=config.SLATE, family="monospace")
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        if abs(y1 - y2) < 1e-9:
            start = (x1 + box_w, y1 + box_h / 2) if x2 > x1 else (x1, y1 + box_h / 2)
            end = (x2, y2 + box_h / 2) if x2 > x1 else (x2 + box_w, y2 + box_h / 2)
        else:
            start, end = (x1 + box_w / 2, y1), (x2 + box_w / 2, y2 + box_h)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=config.TEAL_DARK, linewidth=1.6))
    ax.set_xlim(-0.3, cols * (box_w + gap_x))
    ax.set_ylim(-(box_h + gap_y) - 0.4, box_h + 0.5)
    ax.set_title("SleepWise system architecture and workflow", color=config.NAVY, fontsize=12, fontweight="bold")
    return _save(fig, path)

"""Command-line entry point: python train.py [--data-path PATH] [--skip-eda]

Runs validation, EDA figures, cross-validated model selection, final test
evaluation and explainability, then writes everything to models/ and reports/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed; figures are written to files

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sleepwise.data_loader import DatasetError  # noqa: E402
from sleepwise.train_model import run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SleepWise sleep-quality model.")
    parser.add_argument("--data-path", help="Path to the CSV (default: data/raw/Sleep_health_and_lifestyle_dataset.csv)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)  # silence harmless notices
    try:
        metrics = run_training(args.data_path, make_figures=not args.skip_eda, run_eda=not args.skip_eda)
    except DatasetError as exc:
        print(f"\nDATASET PROBLEM:\n{exc}", file=sys.stderr)
        return 1

    sel = metrics["model_selection"]["selected_model"]
    test = metrics["test_set"]["selected_model"]
    base = metrics["test_set"]["baseline_training_mean"]
    print("\n=== Training finished ===")
    print(f"Deployed model: {sel}  (best by CV RMSE: {metrics['model_selection']['best_by_cv']})")
    print(f"Held-out test  RMSE {test['rmse']:.3f} | MAE {test['mae']:.3f} | R2 {test['r2']:.3f}")
    print(f"Mean baseline  RMSE {base['rmse']:.3f} | MAE {base['mae']:.3f} | R2 {base['r2']:.3f}")
    print("Artifacts: models/sleepwise_model.joblib, models/model_metadata.json, reports/")
    print("Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

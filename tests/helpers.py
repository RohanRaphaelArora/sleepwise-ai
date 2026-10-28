"""Small SYNTHETIC fixtures used only by the unit tests.

They are deliberately tiny and clearly artificial. They are never used for training the
real model and never substitute for the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OCCUPATIONS = ["Nurse", "Doctor", "Engineer", "Teacher"]


def make_synthetic_df(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stress = rng.integers(1, 10, n)
    duration = np.round(rng.uniform(5.0, 9.0, n), 1)
    quality = np.clip(np.round(10 - 0.6 * stress + 0.5 * (duration - 7) + rng.normal(0, 0.3, n)), 1, 10)
    systolic = rng.integers(105, 150, n)
    diastolic = rng.integers(65, 95, n)
    return pd.DataFrame(
        {
            "Person ID": np.arange(1, n + 1),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(22, 65, n),
            "Occupation": rng.choice(OCCUPATIONS, n),
            "Sleep Duration": duration,
            "Quality of Sleep": quality.astype(int),
            "Physical Activity Level": rng.integers(20, 100, n),
            "Stress Level": stress,
            "BMI Category": rng.choice(["Normal", "Normal Weight", "Overweight", "Obese"], n),
            "Blood Pressure": [f"{s}/{d}" for s, d in zip(systolic, diastolic)],
            "Heart Rate": rng.integers(60, 90, n),
            "Daily Steps": rng.integers(2000, 11000, n),
            "Sleep Disorder": rng.choice(["None", "Insomnia", "Sleep Apnea"], n),
        }
    )


def write_synthetic_csv(path, **kwargs):
    make_synthetic_df(**kwargs).to_csv(path, index=False)
    return path

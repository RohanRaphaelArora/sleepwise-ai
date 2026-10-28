# data/

| Folder | Contents |
|---|---|
| `raw/` | The original Kaggle CSV, **unmodified**: `Sleep_health_and_lifestyle_dataset.csv`. |
| `processed/` | `sleep_cleaned.csv`, written by `python train.py` (blood pressure split, BMI spellings merged). Safe to delete; it is regenerated. |

The training code never edits the raw file, never fabricates rows, and never fills in missing values outside the
model pipeline (imputers inside the pipeline are fitted on training rows only).

To use a CSV stored elsewhere:

```powershell
python train.py --data-path "C:\path\to\my_sleep_data.csv"
# or, for every command in this terminal session:
$env:SLEEPWISE_DATA_PATH = "C:\path\to\my_sleep_data.csv"
```

See `../DATA_SOURCES.md` for provenance and limitations.

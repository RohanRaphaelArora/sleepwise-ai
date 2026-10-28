# SleepWise AI: sleep quality estimation with a neural network

Course project for **BDA404-5N Artificial Neural Networks and Deep Learning** (ESE, project implementation and deployment).
It estimates the **recorded sleep-quality score** (`Quality of Sleep`) from lifestyle and health characteristics in the Kaggle *Sleep Health and Lifestyle* dataset. The deployed model is a feed-forward neural network; classical models and a from-scratch NumPy network are compared against it. It is an academic prototype: **not** a medical device, **not** a diagnosis, **not** a forecast of future sleep.

Live app: `<add Streamlit URL>`   Code: `<add GitHub URL>`

## 1. Quick start (Windows PowerShell, VS Code terminal)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python train.py
streamlit run app.py
python -m pytest
```

| Command | What it does | Creates |
|---|---|---|
| `python train.py` | Validates the CSV, makes EDA charts, cross-validates 5 models, evaluates once on held-out rows, saves the neural network | `models/*`, `reports/*`, `data/processed/sleep_cleaned.csv` |
| `python train.py --data-path "C:\x\my.csv"` | Same with another CSV | same |
| `python train.py --skip-eda` | Skip EDA charts | |
| `streamlit run app.py` | Opens the dashboard at http://localhost:8501 | |
| `python -m pytest` | Runs the unit tests | |

The CSV is expected at `data\raw\Sleep_health_and_lifestyle_dataset.csv` (a copy is included). If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
**VS Code:** open the folder, Ctrl+Shift+P → *Python: Select Interpreter* → `.venv`, then use the terminal commands above. For the notebook pick the `.venv` kernel and *Run All*.

Deployment to the web: see **`DEPLOYMENT.md`**. Exam submission steps: **`docs/SUBMISSION_CHECKLIST.md`**.

## 2. What is in the box

```text
sleepwise-ai/
├── app.py  train.py  conftest.py  pytest.ini
├── requirements.txt  requirements-dev.txt  requirements-lock.txt
├── README.md  DEPLOYMENT.md  DATA_SOURCES.md  .gitignore  .github/workflows/tests.yml  .streamlit/config.toml
├── data/{raw,processed}/        dataset and cleaned copy
├── models/                      sleepwise_model.joblib, model_metadata.json
├── notebooks/exploratory_analysis.ipynb
├── reports/                     model_metrics.json, dataset_schema.json, eda_summary.json, feature_importance.csv,
│                                network_architecture.json, figures/ (EDA, evaluation, architecture diagrams)
├── src/sleepwise/               config, data_loader, preprocessing, eda, numpy_mlp, train_model, evaluate,
│                                diagrams, explainability, predict, sleep_education
├── tests/                       4 test files + helpers (synthetic fixtures for unit tests only)
└── docs/                        PROJECT_REPORT, TEST_REPORT, VIVA_PREP, DEMO_SCRIPT, SUBMISSION_CHECKLIST
```

## 3. The application

A dark, narrative dashboard modelled on the layout of the reference site you supplied (eyebrow labels, numbered sections, a hero question, a stats strip). Pages:

| Page | Purpose |
|---|---|
| 01 Home | Question, three-step explainer, workflow diagram, dataset summary, model status, limitations |
| 02 Sleep Analysis | Enter sleep duration (required, 0 to 16 hours, interface range only) and optional lifestyle details, press **Analyze Sleep**; without details the duration-only model is used |
| 03 Behavioural Insights | Observed relationships (A), what the model relies on (B), pointer to general information (C) |
| 04 Potential Effects | Educational guidance with age table, effects, evidence type, source links and disclaimers (not model output) |
| 05 Model Performance | Experiment A versus B, architecture, cross-validation, test results, reliability notes |
| 06 About | Objective, dataset, method, ethics, limitations, future work |

Educational content is in `src/sleepwise/sleep_education.py`; its links and wording were checked on 28 October 2026 against CDC, NHLBI and AASM/SRS pages. Verify them again before citing.

## 4. How it works

* **Target:** `Quality of Sleep` (regression). **Inputs:** Age, Gender, Occupation, Sleep Duration, Physical Activity Level, Stress Level, BMI Category, Heart Rate, Daily Steps, Systolic BP, Diastolic BP.
* **Excluded:** `Person ID` (identifier), `Quality of Sleep` (target), `Sleep Disorder` (a recorded outcome that could leak the target).
* **Preprocessing (inside a scikit-learn `Pipeline`, fitted on training rows only):** median imputation + scaling for numbers; most-frequent imputation + one-hot (unseen categories ignored) for text columns.
* **Neural network (deployed):** 23 → 64 → 32 → 1, ReLU hidden layers, linear output, Adam, L2 penalty, early stopping, standardised target (3,649 trainable parameters). A second network is written from scratch in NumPy (`numpy_mlp.py`) with manual back-propagation, checked against numerical gradients in the tests.
* **Split:** 80/20, `random_state=42`, rows with identical inputs kept together (`GroupShuffleSplit`). The CSV has only 109 distinct input combinations among 374 rows; a plain random split would leak near-copies into the test set.
* **Comparison:** mean baseline, Linear Regression, Decision Tree, Random Forest, scikit-learn network, NumPy network, with 5-fold grouped cross-validation on the training split only.
* **Which model is deployed:** the scikit-learn neural network, chosen *by design* (this is a neural-network project) before any test result was seen. The cross-validation winner is reported next to it. Set `DEPLOYED_MODEL = "auto"` in `config.py` to deploy the CV winner instead.
* **Metrics:** MAE, RMSE, R². **Explainability:** permutation importance (model reliance, not causation).
* **Output scale:** predictions are clipped to 1-10 for display only; metrics use raw output.

## 5. Results from the run included in this folder

Generated by actually running `python train.py`; your run should reproduce them closely (tiny differences are possible with other library versions). `reports/model_metrics.json` is the source of truth.

| Model | CV RMSE (training split) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Mean baseline | 1.268 | 0.919 | 1.047 | -0.071 |
| Linear Regression | 0.338 | 0.187 | 0.392 | 0.850 |
| Decision Tree | 0.295 | 0.111 | 0.311 | 0.905 |
| Random Forest | 0.237 | 0.063 | 0.195 | 0.963 |
| **Neural network (MLP), deployed** | 0.268 | 0.121 | 0.244 | 0.942 |
| Neural network (NumPy, from scratch) | 0.243 | 0.134 | 0.286 | 0.920 |

Test set: 86 rows, 22 distinct input combinations. The test numbers for non-deployed models are shown for transparency and did not influence which model was deployed.
**Experiments:** sleep duration only (Decision Tree) gives test MAE 0.370, RMSE 0.531, R² 0.724; all inputs (deployed neural network) give MAE 0.121, RMSE 0.244, R² 0.942. Duration alone carries real signal but is cruder; both are limited by the small, structured dataset.
**Honest reading:** the neural network is far better than the baseline, but the Random Forest is better on this small tabular dataset. The data is highly regular (and of unverified origin), so high R² values say little about real-world accuracy. Prediction takes about 5.5 ms and a full `python train.py` about 9.7 s on the build machine.

## 6. Module guide

| Module | Responsibility |
|---|---|
| `config.py` | Paths, column names, seeds, input ranges, deployed-model switch |
| `data_loader.py` | Read CSV, helpful errors, validation, schema report (rows, columns, missing, duplicates, SHA-256) |
| `preprocessing.py` | Cleaning, blood-pressure parsing, feature roles, group ids, `ColumnTransformer` |
| `eda.py` | Summary statistics and EDA figures |
| `numpy_mlp.py` | Neural network from scratch (forward pass, back-propagation, Adam) |
| `train_model.py` | Split, cross-validation, fitting, test evaluation, saving artefacts |
| `evaluate.py`, `diagrams.py` | Metrics, plots, training curve, architecture diagrams |
| `explainability.py` | Permutation importance and the "not causal" note |
| `predict.py` | Load model, validate input, build one-row frame, predict, clip for display |
| `sleep_education.py` | Age guidance, duration descriptions and effects with source links (not ML) |
| `app.py` | UI only; reads artefacts; retrains once on start-up if they are missing or incompatible |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `DATASET PROBLEM: Dataset not found` | Put the CSV at `data\raw\Sleep_health_and_lifestyle_dataset.csv` or use `--data-path` |
| `Required target column 'Quality of Sleep' was not found` | Wrong file or renamed column; the message lists the columns found |
| App says the model is not trained | It retrains itself if the CSV exists; otherwise run `python train.py` |
| `ModuleNotFoundError: sleepwise` | Run commands from the project root (the folder containing `app.py`) |
| `streamlit` not recognised | Activate the venv and re-run `pip install -r requirements-dev.txt` |
| scikit-learn version warning | Harmless; the app retrains automatically, or run `python train.py` |

## 8. Verification status (read before presenting)

Verified by execution in the build environment: data loading and validation, cleaning, EDA figures, training and cross-validation of all models, test evaluation, artefact creation, single-row prediction, input validation, self-healing start-up (fresh clone and version mismatch), the numerical gradient check of the NumPy network, and all automated tests.
**Not verified here:** the Streamlit interface in a real browser (page code was executed against a stand-in `streamlit` module) and the tests under real `pytest` (a stand-in runner executed the same test functions). Run `python -m pytest` and click through `docs/TEST_REPORT.md` section B once on your machine. The GitHub Actions workflow and the Streamlit Community Cloud deployment have also not been run.

## 9. Limitations and future work

Limitations: small, highly repetitive dataset of unverified provenance; associations not causation; narrow value ranges; no uncertainty estimate; cross-sectional (no forecasting); gender recorded as binary; some occupations have 1 to 4 rows; the neural network does not beat the Random Forest here.
Future work: prediction intervals (conformal prediction or deep ensembles), external validation data, nested-CV tuning, occupation embeddings, fairness checks, a separate justified Sleep Disorder experiment.

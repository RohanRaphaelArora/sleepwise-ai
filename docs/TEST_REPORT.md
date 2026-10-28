# Test report

## A. Automated unit tests (44 cases)
Run with `python -m pytest`. The *Result* column records the run in the build environment, where real pytest could not be installed offline, so a minimal stand-in runner executed the same test functions. **Run `python -m pytest` on your own machine and record its output here before submitting.**

| ID | File | Test case | Result |
|---|---|---|---|
| T01 | `test_data_loader.py` | Loads csv and reports shape | passed |
| T02 | `test_data_loader.py` | Sleep disorder none is a category not missing | passed |
| T03 | `test_data_loader.py` | Missing file gives helpful error | passed |
| T04 | `test_data_loader.py` | Empty file gives helpful error | passed |
| T05 | `test_data_loader.py` | Missing target column is rejected | passed |
| T06 | `test_data_loader.py` | Non numeric target is rejected | passed |
| T07 | `test_data_loader.py` | Env var overrides default path | passed |
| T08 | `test_data_loader.py` | Schema report excludes id target and sleep disorder | passed |
| T09 | `test_data_loader.py` | Duplicates are reported ignoring id | passed |
| T10 | `test_neural_network.py` | Analytic gradients from back-propagation must equal central finite differences. | passed |
| T11 | `test_neural_network.py` | Numpy network learns a simple function | passed |
| T12 | `test_neural_network.py` | Numpy network is reproducible with a seed | passed |
| T13 | `test_prediction.py` | Training creates all artifacts and metrics | passed |
| T14 | `test_prediction.py` | Training and prediction use identical feature columns | passed |
| T15 | `test_prediction.py` | Prediction is finite and on the target scale | passed |
| T16 | `test_prediction.py` | Prediction responds to inputs | passed |
| T17 | `test_prediction.py` | Missing model artifacts raise helpful error | passed |
| T18 | `test_prediction.py` | Invalid inputs are all reported | passed |
| T19 | `test_prediction.py` | Non numeric and nan inputs are rejected | passed |
| T20 | `test_prediction.py` | Out of training range gives warning not error | passed |
| T21 | `test_prediction.py` | Saved pipeline matches metadata model name | passed |
| T22 | `test_prediction.py` | Deployed model is a neural network with recorded architecture | passed |
| T23 | `test_prediction.py` | Duration only model works with only sleep duration | passed |
| T24 | `test_prediction.py` | Duration only prediction responds to duration | passed |
| T25 | `test_prediction.py` | Sleep duration outside interface range is rejected | passed |
| T26 | `test_prediction.py` | Missing duration only artifacts raise helpful error | passed |
| T27 | `test_prediction.py` | Experiments are recorded in the metrics file | passed |
| T28 | `test_preprocessing.py` | Blood pressure is split into numbers | passed |
| T29 | `test_preprocessing.py` | Invalid blood pressure becomes missing with warning | passed |
| T30 | `test_preprocessing.py` | Frame without blood pressure is unchanged | passed |
| T31 | `test_preprocessing.py` | Bmi spellings are merged | passed |
| T32 | `test_preprocessing.py` | Feature roles split numeric and categorical | passed |
| T33 | `test_preprocessing.py` | Prepare dataset never contains target or id | passed |
| T34 | `test_preprocessing.py` | Rows with missing target are dropped and counted | passed |
| T35 | `test_preprocessing.py` | Preprocessor imputes and ignores unseen categories | passed |
| T36 | `test_preprocessing.py` | Preprocessor is fitted on training data only | passed |
| T37 | `test_preprocessing.py` | Identical rows share a group id | passed |
| T38 | `test_sleep_education.py` | Every source is https and from a reputable domain | passed |
| T39 | `test_sleep_education.py` | Every effect has explanation evidence and a known source | passed |
| T40 | `test_sleep_education.py` | Groups cover insufficient adequate and long sleep | passed |
| T41 | `test_sleep_education.py` | Education text has no em dashes and no probability language | passed |
| T42 | `test_sleep_education.py` | Adult duration categories use cautious wording | passed |
| T43 | `test_sleep_education.py` | Minors get age specific guidance and a model warning | passed |
| T44 | `test_sleep_education.py` | Age not given is not assumed to be adult | passed |

Synthetic fixtures (`tests/helpers.py`) are used only inside tests and never for training the real model.

## B. Manual application tests (run these in the browser)
| ID | Steps | Expected | Tick |
|---|---|---|---|
| M1 | Start with `models/` emptied, open the app | Instruction or automatic training, no crash | [ ] |
| M2 | Home page | Dataset card, model status with metrics, limitations box | [ ] |
| M3 | Sleep Analysis with the default form values and "Analyze Sleep" | A number between 1 and 10, insights table | [ ] |
| M4 | Set systolic 80, diastolic 90 | Error: systolic must be higher than diastolic | [ ] |
| M5 | Set Daily Steps 25000 | Warning: outside the training range | [ ] |
| M6 | Change Stress 4 to 8 only | Estimate changes (about 8.06 to 7.61 in the build run) | [ ] |
| M7 | Behavioural Insights page | Statistics, observed charts, feature importance and the "More charts" expander load | [ ] |
| M8 | Model Performance page | Architecture, training curve, CV table, test table, figures, reliability notes | [ ] |
| M9 | About page | Methodology, ethics, limitations | [ ] |
| M11 | Potential Effects page | Age table, effect cards with working source links, disclaimers | [ ] |
| M12 | Switch off "Add lifestyle details" and analyze | Duration-only estimate; works with only sleep duration | [ ] |
| M10 | Open the deployed link on a phone | Pages readable, form usable | [ ] |

## C. Experimental evaluation
See `docs/PROJECT_REPORT.md` section 4 and `reports/model_metrics.json` (cross-validation, held-out test, baseline, diagnostics).

# Demo script: 10 to 12 minutes (matches the 10-slide structure in the exam brief)

Before you start: `.venv\Scripts\activate`, `streamlit run app.py`, open the deployed link in a second tab as a backup, keep this file and the slides open.

| Time | Slide / screen | Say |
|---|---|---|
| 0:00 | 1 Title | Name, roll number, course, project title |
| 0:30 | 2 Problem and objectives | Estimate recorded sleep quality from 11 characteristics with a neural network; not diagnosis |
| 1:30 | 3 Background | Backprop, Adam, ReLU, regularisation; trees often strong on small tabular data |
| 2:30 | 4 Method and architecture | System diagram; 23 → 64 → 32 → 1 network; grouped split and why |
| 3:45 | 5 Dataset and preprocessing | 374 x 13; 242 repeated rows; 109 distinct combinations; exclusions; provenance unverified |
| 4:45 | 6 Implementation | Pipeline in scikit-learn MLP; NumPy network with hand-written backprop and the gradient check |
| 5:45 | 7 Results | CV table; test MAE 0.121, RMSE 0.244, R² 0.942; Random Forest is better; naive split would mislead |
| 7:00 | 8 Live demo (switch to the app) | see below |
| 9:30 | 9 Limitations and future scope | small data, associations, no uncertainty |
| 10:30 | 10 Conclusion and references | one-sentence takeaway; invite questions |

## Live demo (about 2.5 minutes)
Sleep Analysis page (leave "Add lifestyle details" on for A to D). Values below are the real outputs of the deployed network in the build run; yours may differ in the second decimal after retraining.

| Scenario | Inputs | Estimate |
|---|---|---|
| A: rested, low stress | Age 35, Female, Nurse, Sleep 7.5 h, Activity 60, Stress 4, BMI Normal, HR 70, Steps 7000, BP 120/80 | about 8.06 |
| B: stressed, short sleep | as A but Stress 8, Sleep 5.9, HR 85, BMI Overweight, Occupation Salesperson | about 5.05 |
| C: only stress 4 → 8 | as A, Stress 8 | about 7.61 |
| D: only sleep 7.5 → 6.0 | as A, Sleep 6.0 | about 7.35 |

Say: the output is computed from the trained network, changes with inputs, and shows what this model relies on in this dataset, not what would happen to a real person.
Validation: systolic 80 with diastolic 90 gives an error; Daily Steps 25000 gives an extrapolation warning.
Duration-only demo: switch off "Add lifestyle details" and enter only sleep duration.

| Hours | Duration-only estimate |
|---|---|
| 5.5 | about 4.0 |
| 7.5 | about 7.6 |
| 9.5 | about 9.0 (warning: outside the training range, an extrapolation) |

Say: with duration only the model sees one number, so it is cruder (R² 0.72 against 0.94 with all clues); a decision tree gives stepped values.

Then show **Potential Effects**: the age table, one effect card with its source link, and the disclaimer that this is general guidance and not a model output.
Then open **Model Performance**: Experiment A versus B, architecture, training curve, tables, the "How reliable are these numbers?" section.

## If something breaks
* App says the model is not trained: it retrains itself in about a minute; or run `python train.py`.
* Port busy: `streamlit run app.py --server.port 8502`.
* No internet: use the local app, not the deployed link.

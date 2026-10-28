# Data sources and limitations

## Source
* **Dataset:** "Sleep Health and Lifestyle Dataset", distributed on Kaggle as `Sleep_health_and_lifestyle_dataset.csv`.
* **File used in this project:** 374 rows x 13 columns, SHA-256 `1efe7b6f781ff88078d08d81fe136cff98b1b0c32560f35f8650ca5984b77841`
  (the hash lets anyone confirm they are using the identical file; it is also stored in `reports/dataset_schema.json`).

## Fields YOU must fill in (they cannot be determined from the CSV, so none are invented here)
| Item | Value |
|---|---|
| Kaggle page URL | _TO BE FILLED IN by the student_ |
| Author / uploader | _TO BE FILLED IN by the student_ |
| Licence shown on the Kaggle page | _TO BE FILLED IN by the student_ (check it before redistributing the CSV) |
| Download date | _TO BE FILLED IN by the student_ |
| Collection methodology stated by the author | _TO BE FILLED IN by the student, or "not stated"_ |

## What the file itself shows (verified by running code on it)
* 374 rows, 13 columns: Person ID, Gender, Age, Occupation, Sleep Duration, Quality of Sleep, Physical Activity Level,
  Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps, Sleep Disorder.
* No missing values. `Sleep Disorder` uses the text `None` for "no disorder" (a real category, not missing data).
* 0 exact duplicate rows, but **242 rows repeat another row's values once `Person ID` is ignored**; only **109 distinct
  combinations of the model's input features** exist. Within this file every distinct feature combination has exactly one
  sleep-quality score.
* `BMI Category` spells one group two ways ("Normal" and "Normal Weight"); the project merges them.
* `Quality of Sleep` takes the integer values 4-9 (nominal scale 1-10); `Stress Level` spans 3-8.

## Provenance statement
The CSV contains no documentation of how, where, from whom or when the values were collected, and this project has not
independently verified that. The highly repetitive structure above is consistent with templated or synthetic data, but
that is an observation, not a finding about origin. Therefore:
* Do **not** describe the data as clinically validated or as verified real-world measurements.
* Describe it as "a publicly shared sleep and lifestyle dataset of unverified provenance, used for an academic prototype".

## Limitations
1. Small sample; effective sample size is closer to 109 than 374 because of repeated rows.
2. Cross-sectional: one record per person, no repeated nights, so it cannot support forecasting.
3. Narrow ranges (e.g. ages 27-59, daily steps 3000-10000); predictions outside them are extrapolation.
4. Several occupations have very few rows (some only 1-4), so occupation effects are unreliable.
5. Units/definitions of some columns (e.g. Physical Activity Level) are not documented in the file.
6. Gender is recorded as Male/Female only.

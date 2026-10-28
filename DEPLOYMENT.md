# Deployment guide

The app is deployed from GitHub to **Streamlit Community Cloud** (free). Local use needs no cloud at all.
Menu names on those websites change from time to time; follow the intent if a label differs.

## 0. Check locally first
```powershell
.venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py
python -m pytest
streamlit run app.py
```
Work through the manual checklist in `docs/TEST_REPORT.md` (section B).

## 1. Put the code on GitHub (the exam requires this)
1. Create an empty repository on github.com (for example `sleepwise`). Public is simplest for Streamlit Community Cloud.
2. In the project folder:
```powershell
git init
git add .
git commit -m "SleepWise: sleep quality estimation with a neural network"
git branch -M main
git remote add origin https://github.com/<your-username>/sleepwise.git
git push -u origin main
```
3. The repository intentionally includes the model, reports, figures and the CSV so the app starts instantly. **Check the dataset's licence on its Kaggle page first.** If it does not allow redistribution, delete `data/raw/*.csv` and `data/processed/*.csv` from the repository; the app will still run from the committed model, but it can only retrain if a CSV is present.

## 2. Deploy on Streamlit Community Cloud
1. Sign in at https://share.streamlit.io with your GitHub account and allow access to the repository.
2. **Create app** → choose the repository, branch `main`, main file path `app.py`.
3. Open **Advanced settings** and choose Python 3.12 (3.11 also works).
4. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes.
5. Optionally edit the app URL to something readable.
6. Copy the URL into `docs/PROJECT_REPORT.md`, the README and your slides.

## 3. How the deployment is protected against surprises
* Artefacts are committed, so no training is needed at start-up.
* If the saved model is missing or was built with a different scikit-learn version, `app.py` retrains once from the bundled CSV (about a minute) and shows a spinner.
* `requirements.txt` uses version ranges; `requirements-lock.txt` lists the exact versions the project was built with.
* The app only reads files inside the repository and stores nothing about users.

## 4. Troubleshooting
| Symptom | Fix |
|---|---|
| Build fails on a package | Open the build log; pin the version from `requirements-lock.txt` |
| App shows "Automatic training failed" | The CSV is missing from the repository, or a library error; read the message |
| App is slow after being idle | Free apps sleep; open the link a few minutes before presenting and keep the local app as a backup |
| Out-of-date results after retraining | Commit and push the new `models/` and `reports/` files |

## 5. Alternative: run only locally
Everything works offline with `python train.py` and `streamlit run app.py`. Exam instructions accept "a deployment link or local setup".

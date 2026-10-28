# Submission checklist

**Presentation:** 29 October 2026 (1-hour session; about 10 to 12 minutes each, including Q&A).
**Submission deadline:** 3 November 2026, 12:30 PM, Google Classroom. Late submissions are not accepted without prior approval.

## Before 29 October
- [ ] `python train.py` and `python -m pytest` both succeed on your own machine; paste the pytest summary into `docs/TEST_REPORT.md`
- [ ] Work through the manual checks M1 to M12
- [ ] Fill in the contribution section and check the references in `docs/PROJECT_REPORT.md`
- [ ] Fill in the Kaggle URL, author, licence and download date in `DATA_SOURCES.md`
- [ ] Push to GitHub, deploy, put both links in the report and slides
- [ ] Open the deployed app 10 minutes before presenting; keep the local app running as a backup
- [ ] Rehearse with `docs/DEMO_SCRIPT.md` and `docs/VIVA_PREP.md`; time yourself
- [ ] Bring: report, source code, dataset details, test results (all in this folder)

## For Google Classroom
- [ ] ZIP the whole project folder (source, data, models, reports, documentation, slides) and upload it
- [ ] The ZIP runs: unzip, create the venv, `pip install -r requirements.txt`, `streamlit run app.py`
- [ ] GitHub repository contains the code; share the GitHub link in the Classroom submission
- [ ] Click **Turn in**

## Rubric map
| Component (marks) | Where it is covered |
|---|---|
| Problem, objectives, requirements (10) | Report section 1 |
| Design, architecture, methodology (15) | Report section 2, system and network diagrams |
| Implementation and code quality (25) | `src/sleepwise/`, tests, report section 3 |
| Testing, results, evaluation (15) | Report section 4, `docs/TEST_REPORT.md`, `reports/` |
| Deployment and live demo (20) | `app.py`, `DEPLOYMENT.md`, `docs/DEMO_SCRIPT.md` |
| Presentation, documentation, references (5) | slides, report, references |
| Viva and individual contribution (10) | `docs/VIVA_PREP.md`, report section 7 |

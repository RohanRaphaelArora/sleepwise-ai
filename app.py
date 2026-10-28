"""SleepWise AI - Streamlit dashboard.   Run with:  streamlit run app.py

The app only *reads* artefacts produced by `python train.py`. If they are missing it trains once
from the bundled CSV (or shows instructions) and never hard-codes a prediction or a metric.
"""

from __future__ import annotations

import html as _html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sleepwise import config, predict as pred, sleep_education as edu  # noqa: E402
from sleepwise.eda import FIGURE_CAPTIONS  # noqa: E402
from sleepwise.explainability import EXPLAINABILITY_NOTE  # noqa: E402

st.set_page_config(page_title="SleepWise AI", layout="wide", initial_sidebar_state="collapsed")

FIG_DIR = config.FIGURES_DIR
DARK_DIR = FIG_DIR / "dark"
PAGES = ["01 Home", "02 Sleep Analysis", "03 Behavioural Insights", "04 Potential Effects", "05 Model Performance", "06 About"]
DURATION = config.SLEEP_DURATION_COLUMN

CSS = """
<style>
:root{--bg:#101312;--card:#161D1B;--card2:#1B2321;--line:#2A322F;--text:#E8EEEA;--muted:#9AA7A1;--accent:#3CCBC0;--accent-dim:#1F6F69}
html,body,[data-testid="stAppViewContainer"],.stApp{background:var(--bg)!important;color:var(--text)}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important}
header[data-testid="stHeader"]{background:transparent!important;height:0!important}
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
.block-container{max-width:1120px;padding:1.6rem 1.6rem 4rem!important}
h1,h2,h3{font-family:Georgia,"Times New Roman",serif!important;color:var(--text)!important;letter-spacing:-.01em}
h2{font-size:2rem!important;line-height:1.15!important;margin:.2rem 0 .6rem!important}
h3{font-size:1.25rem!important}
p,li{line-height:1.6}
a{color:var(--accent)!important}
.sw-brand{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:.4rem 1.5rem;border-bottom:1px solid var(--line);padding-bottom:.8rem;margin-bottom:.4rem}
.sw-brand b{font-family:Georgia,serif;font-size:1.25rem}
.sw-brand span{font:600 .72rem ui-monospace,Consolas,monospace;letter-spacing:.14em;color:var(--muted);margin-left:.7rem}
.eyebrow{font:600 .74rem ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:2.2rem 0 .5rem}
.tag{display:inline-block;border:1px solid var(--accent-dim);color:var(--accent);font:600 .72rem ui-monospace,Consolas,monospace;letter-spacing:.12em;padding:.25rem .6rem;border-radius:3px;text-transform:uppercase}
.hero{padding:2.4rem 0 1.2rem}
.hero h1{font-size:clamp(2.3rem,5.4vw,4.1rem)!important;line-height:1.05!important;font-weight:600;margin:1rem 0 1.1rem!important;max-width:15em}
.hero h1 em{color:var(--accent);font-style:italic}
.hero p.lead{font-size:1.15rem;color:#C5D0CA;max-width:40em}
.small{color:var(--muted);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.3rem 1.4rem}
.card .k{font:600 .72rem ui-monospace,Consolas,monospace;letter-spacing:.14em;color:var(--accent);text-transform:uppercase}
.card h3{margin:.5rem 0 .4rem!important}
.card p{color:#C5D0CA;font-size:.98rem;margin:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:1.6rem 0}
.stats div{background:var(--card);padding:1.1rem 1.3rem}
.stats b{display:block;font:600 2.1rem Georgia,serif;color:var(--text)}
.stats span{color:var(--muted);font-size:.88rem}
.callout{display:flex;gap:1rem;align-items:flex-start;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.1rem 1.3rem;margin:1.4rem 0}
.callout .q{flex:0 0 2rem;height:2rem;border:1px solid var(--accent);color:var(--accent);border-radius:4px;display:flex;align-items:center;justify-content:center;font:700 1rem Georgia,serif}
.callout p{margin:0;color:#C5D0CA}
.callout b{color:var(--text)}
.result{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.5rem}
.result .big{font:600 4.2rem/1 Georgia,serif;color:var(--text)}
.result .big small{font-size:1.4rem;color:var(--muted)}
.scale{position:relative;height:6px;background:var(--line);border-radius:3px;margin:1.4rem 0 .3rem}
.scale i{position:absolute;top:-6px;width:4px;height:18px;background:var(--accent);border-radius:2px}
.scale-l{display:flex;justify-content:space-between;color:var(--muted);font:.72rem ui-monospace,Consolas,monospace}
.eff{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1.1rem 1.3rem;margin:0 0 .8rem}
.eff h4{margin:0 0 .35rem;font:600 1.05rem Georgia,serif;color:var(--text)}
.eff p{margin:.2rem 0;color:#C5D0CA;font-size:.96rem}
.eff .m{color:var(--muted);font-size:.84rem}
.pillless{display:inline-block;color:var(--muted);font:.72rem ui-monospace,Consolas,monospace;letter-spacing:.08em;border:1px solid var(--line);padding:.15rem .5rem;border-radius:3px;margin-right:.4rem}
.sw-foot{border-top:1px solid var(--line);margin-top:3rem;padding-top:1.2rem;color:var(--muted);font-size:.88rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
div[data-testid="stRadio"] [role="radiogroup"]{gap:1.6rem;flex-wrap:wrap}
div[data-testid="stRadio"] label{margin:0!important;padding:.45rem 0;border-bottom:2px solid transparent;cursor:pointer}
div[data-testid="stRadio"] label>div:first-child{display:none!important}
div[data-testid="stRadio"] label p{font:600 .74rem ui-monospace,SFMono-Regular,Consolas,monospace!important;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)!important}
div[data-testid="stRadio"] label:has(input:checked){border-bottom-color:var(--accent)}
div[data-testid="stRadio"] label:has(input:checked) p{color:var(--text)!important}
.stButton>button,.stFormSubmitButton>button{border-radius:4px!important;font-weight:600;border:1px solid var(--accent)!important;background:transparent;color:var(--accent)!important}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:var(--accent)!important;color:#06201E!important}
.stButton>button:hover,.stFormSubmitButton>button:hover{filter:brightness(1.12)}
[data-testid="stImage"] img{border:1px solid var(--line);border-radius:4px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:4px}
@media (max-width:760px){.result .big{font-size:3.2rem}.eyebrow{margin-top:1.6rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def h(text: str) -> str:
    return _html.escape(str(text))


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    md(f"<div class='eyebrow'>{h(text)}</div>")


def callout(body_html: str) -> None:
    md(f"<div class='callout'><div class='q'>?</div><p>{body_html}</p></div>")


def go(page: str) -> None:
    st.session_state["nav"] = page


def nav_button(label: str, page: str, key: str, primary: bool = False) -> None:
    st.button(label, key=key, on_click=go, args=(page,), type="primary" if primary else "secondary")


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def _load_model(models_dir: str, variant: str, stamp: float):
    return pred.load_model(models_dir, variant)


@st.cache_data
def _load_json(path: str, stamp: float):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def _load_csv(path: str, stamp: float):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def get_json(path: Path):
    return _load_json(str(path), _mtime(path))


def get_csv(path: Path):
    return _load_csv(str(path), _mtime(path))


@st.cache_resource(show_spinner="Preparing the models (first start only, about a minute)...")
def _ensure_artifacts() -> tuple[str, str]:
    """Make sure trained models matching the installed scikit-learn exist (self-healing start-up)."""
    import sklearn

    ready = all(pred.model_artifacts_available(config.MODELS_DIR, v) for v in pred.VARIANTS)
    stale = False
    if ready:
        meta = json.loads((config.MODELS_DIR / config.METADATA_FILE).read_text(encoding="utf-8"))
        stale = meta.get("sklearn_version") != sklearn.__version__
    if ready and not stale:
        return "ready", ""
    if not config.get_data_path().exists():
        return ("ready", "scikit-learn version differs from training and no CSV is available to retrain.") if ready \
            else ("missing_data", f"No trained model and no dataset at {config.DEFAULT_DATA_PATH}.")
    try:
        from sleepwise.train_model import run_training
        run_training()
        return "trained", "Models were (re)trained on startup."
    except Exception as exc:  # show the reason in the app instead of crashing
        return "error", f"Automatic training failed: {exc}"


BOOT_STATUS, BOOT_MESSAGE = _ensure_artifacts()


def model_ready(variant: str = "full") -> bool:
    return pred.model_artifacts_available(config.MODELS_DIR, variant)


def show_train_instruction() -> None:
    if BOOT_STATUS in ("missing_data", "error"):
        st.error(BOOT_MESSAGE)
    st.warning("The models have not been trained yet. Run the training step, then reload this page.")
    st.code("python train.py", language="powershell")


def show_figure(filename: str, caption: str) -> None:
    path = DARK_DIR / filename
    if not path.exists():
        path = FIG_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not found ({filename}). Run `python train.py` to generate it.")


def f3(v: float) -> str:
    return f"{v:.3f}"


# --------------------------------------------------------------------------- #
# 01 Home
# --------------------------------------------------------------------------- #
def page_home() -> None:
    schema = get_json(config.REPORTS_DIR / config.SCHEMA_FILE)
    metrics = get_json(config.REPORTS_DIR / config.METRICS_FILE)
    md(
        "<div class='hero'><span class='tag'>Recorded data, not a diagnosis</span>"
        "<h1>Can a few lifestyle clues <em>estimate</em> how well someone sleeps?</h1>"
        "<p class='lead'>Hours of sleep alone do not tell us how well a person slept. SleepWise AI trains models on a public "
        "sleep and lifestyle dataset to estimate the recorded sleep-quality score, shows which clues moved the estimate, and "
        "keeps public health guidance about sleep duration in a separate, clearly labelled section.</p></div>"
    )
    b1, b2, _ = st.columns([1.1, 1.3, 3])
    with b1:
        nav_button("Analyze your sleep", PAGES[1], "hero_a", primary=True)
    with b2:
        nav_button("See how the model did", PAGES[4], "hero_b")
    md("<p class='small'>A guide to a model, not a medical tool.</p>")

    c1, c2, c3 = st.columns(3)
    cards = [
        ("01 · The data", "What was recorded?",
         "A public dataset of people with a sleep duration, a recorded sleep-quality score and lifestyle clues such as stress, "
         "activity and heart rate.", PAGES[2], "Explore the clues"),
        ("02 · The model", "What does the network say?",
         "A neural network estimates the recorded quality score from the clues. Simpler models are compared against it.",
         PAGES[1], "Read an estimate"),
        ("03 · The guidance", "What does public guidance say?",
         "The CDC, NIH and the AASM/SRS describe reference ranges and associated effects. This part is educational and is not "
         "predicted by the model.", PAGES[3], "Read the guidance"),
    ]
    for col, (k, title, text, page, label) in zip((c1, c2, c3), cards):
        with col:
            md(f"<div class='card'><div class='k'>{k}</div><h3>{title}</h3><p>{text}</p></div>")
            nav_button(label, page, f"card_{k[:2]}")

    eyebrow("How it works")
    st.markdown("## From a spreadsheet to an estimate")
    show_figure("system_architecture.png", "Workflow: data, cleaning, grouped split, model comparison, saved model, this app.")

    if schema and metrics:
        sel = metrics["model_selection"]["selected_model"]
        test = metrics["test_set"]["selected_model"]
        n_feat = len(metrics["features"]["numeric"]) + len(metrics["features"]["categorical"])
        md(
            "<div class='stats'>"
            f"<div><b>{schema['rows']}</b><span>people recorded in the dataset</span></div>"
            f"<div><b>{n_feat}</b><span>input clues available to the model</span></div>"
            f"<div><b>{schema['distinct_model_feature_combinations']}</b><span>distinct input combinations (many rows repeat)</span></div>"
            f"<div><b>{test['r2']:.2f}</b><span>test R² of the deployed model</span></div></div>"
        )
        st.caption(f"Deployed model: {sel}. Trained {metrics['generated_at_utc']} (UTC). Missing values in the file: "
                   f"{schema['missing_values_total']}. Metrics come from running the pipeline, never typed in.")
    else:
        show_train_instruction()

    callout("<b>“Estimated quality” is a model output.</b> It is not a diagnosis and not a forecast. The dataset is small, many "
            "rows repeat, and its provenance is not documented in the file, so it should not be treated as clinically validated.")
    eyebrow("Important limitations")
    st.markdown(
        "- The task is to estimate a *recorded* score from other recorded characteristics.\n"
        "- Associations in the data are not causes.\n"
        "- Sleep duration alone is a weaker predictor than all clues together (see Model Performance).\n"
        "- Educational sleep guidance comes from public health sources, not from the model."
    )


# --------------------------------------------------------------------------- #
# 02 Sleep Analysis
# --------------------------------------------------------------------------- #
def _render_input(spec: dict, container):
    name = spec["name"]
    unit = f" ({spec['unit']})" if spec.get("unit") else ""
    label = f"{name}{unit}"
    if spec["kind"] == "categorical":
        cats = spec["categories"]
        return container.selectbox(label, cats, index=cats.index(spec["default"]))
    lo, hi = spec["hard_min"], spec["hard_max"]
    help_text = f"Typical range in the training data: {spec['observed_min']:g} to {spec['observed_max']:g}."
    if spec["is_integer"]:
        lo_i, hi_i, med = int(lo), int(hi), int(round(spec["median"]))
        if hi_i - lo_i <= 10:
            return container.slider(label, lo_i, hi_i, med, step=1, help=help_text)
        return container.number_input(label, min_value=lo_i, max_value=hi_i, value=med, step=1, help=help_text)
    return container.number_input(label, min_value=float(lo), max_value=float(hi), value=float(spec["median"]),
                                  step=0.1, format="%.1f", help=help_text)


def _scale_bar(value: float, lo: float, hi: float) -> str:
    pos = 0.0 if hi == lo else min(max((value - lo) / (hi - lo), 0.0), 1.0) * 100
    return (f"<div class='scale'><i style='left:calc({pos:.1f}% - 2px)'></i></div>"
            f"<div class='scale-l'><span>{lo:g}</span><span>recorded sleep-quality scale</span><span>{hi:g}</span></div>")


def page_analysis() -> None:
    eyebrow("02 · Sleep analysis")
    st.markdown("## Enter your sleep, then read the estimate")
    if not (model_ready("full") and model_ready("duration_only")):
        show_train_instruction()
        return

    meta_full = get_json(config.MODELS_DIR / config.METADATA_FILE)
    meta_dur = get_json(config.MODELS_DIR / config.DURATION_METADATA_FILE)
    metrics = get_json(config.REPORTS_DIR / config.METRICS_FILE)
    dur_spec = next(s for s in meta_full["features"] if s["name"] == DURATION)
    target = meta_full["target"]
    st.write(
        "Only sleep duration is required. Adding lifestyle details lets the multivariable model use more information; "
        "leaving them off uses the simpler sleep-duration-only model. This is a model estimate, not a diagnosis."
    )

    use_more = st.toggle("Add lifestyle details (optional, uses the multivariable model)", value=True, key="use_more")
    left, right = st.columns([1.05, 1])
    with left:
        with st.form("analysis_form"):
            default_hours = round(dur_spec["median"] * 4) / 4
            hours = st.slider("Average sleep duration (hours per night)", 0.0, 16.0, float(default_hours), 0.25)
            st.caption("The 0 to 16 hour range is for the interface only. It is not a clinical threshold.")
            values = {DURATION: hours}
            extra_specs = [s for s in meta_full["features"] if s["name"] != DURATION]
            if use_more:
                cols = st.columns(2)
                for i, spec in enumerate(extra_specs):
                    values[spec["name"]] = _render_input(spec, cols[i % 2])
            submitted = st.form_submit_button("Analyze Sleep", type="primary")
        if not use_more:
            st.caption("Only sleep duration will be used. Other details are ignored.")

    with right:
        if not submitted:
            md("<div class='result'><div class='eyebrow' style='margin-top:0'>Your estimate</div>"
               "<p class='small'>Set your sleep duration and choose <b>Analyze Sleep</b>. The estimate appears here.</p></div>")
            return
        variant = "full" if use_more else "duration_only"
        meta = meta_full if use_more else meta_dur
        try:
            pipeline = _load_model(str(config.MODELS_DIR), variant,
                                   _mtime(pred.artifact_paths(config.MODELS_DIR, variant)[0]))
            result = pred.predict_quality(values, pipeline=pipeline, metadata=meta, variant=variant)
        except pred.InputValidationError as exc:
            st.error("Please fix the following:")
            for problem in exc.problems:
                st.write(f"- {problem}")
            return

        score = result.displayed_prediction
        md(
            "<div class='result'><div class='eyebrow' style='margin-top:0'>Estimated sleep quality</div>"
            f"<div class='big'>{score:.1f}<small> / {target['scale_max']:g}</small></div>"
            + _scale_bar(score, target["scale_min"], target["scale_max"])
            + f"<p class='small' style='margin-top:.9rem'>Model: {h(meta['model_name'])} "
              f"({'multivariable' if use_more else 'sleep duration only'}). Training scores ranged from "
              f"{target['observed_min']:g} to {target['observed_max']:g}. Unrounded output: {result.raw_prediction:.2f}.</p></div>"
        )
        if result.was_clipped:
            st.info("The raw model output was outside the 1 to 10 scale and was clipped for display. Clipping says nothing about accuracy.")
        for note in result.warnings:
            st.warning(note)

    # ---- explanation, performance and education (full width, below the form) ----
    exps = (metrics or {}).get("experiments") or {}
    exp_a, exp_b = exps.get("A_duration_only"), exps.get("B_multivariable")
    eyebrow("What the model did")
    if use_more:
        importance = get_csv(config.REPORTS_DIR / config.IMPORTANCE_FILE)
        top = ", ".join(importance.head(3)["feature"]) if importance is not None else "not available"
        st.write(f"The multivariable model received all of your inputs and returned **{score:.1f}**. In this dataset it relies most "
                 f"on: **{top}** (permutation importance). That describes the model, not what causes good or poor sleep.")
        defaults = [s["name"] for s in extra_specs
                    if values.get(s["name"]) == (s["default"] if s["kind"] == "categorical" else
                                                 (int(round(s["median"])) if s["is_integer"] else float(s["median"])))]
        if defaults:
            st.caption("Left at their default values: " + ", ".join(defaults) + ". Defaults are training-data medians or most common categories.")
        perf = exp_b["test_set"]["selected_model"] if exp_b else None
    else:
        st.write(f"The sleep-duration-only model received just **{hours:g} hours** and returned **{score:.1f}**. It ignores every other "
                 "clue, so the estimate is cruder than the multivariable one.")
        perf = exp_a["test_set"]["selected_model"] if exp_a else None
    if perf:
        st.write(f"On held-out rows this setup had a typical error of about **{perf['mae']:.2f}** points (MAE), RMSE **{perf['rmse']:.2f}**, "
                 f"R² **{perf['r2']:.2f}**. No probability or confidence score is produced.")
    if exp_a and exp_b:
        st.caption(f"For comparison: sleep duration only R² {exp_a['test_set']['selected_model']['r2']:.2f}, all clues R² "
                   f"{exp_b['test_set']['selected_model']['r2']:.2f} on the same held-out rows.")

    eyebrow("General guidance, not a model output")
    age = values.get("Age") if use_more else None
    info = edu.describe_duration(hours, age)
    md(f"<div class='eff'><h4>{h(info['label'])}</h4><p>{h(info['message'])}</p>"
       + "".join(f"<p class='m'>{h(n)}</p>" for n in info["notes"])
       + "<p class='m'>Sources: " + ", ".join(f"<a href='{edu.SOURCES[s]['url']}' target='_blank'>{h(edu.SOURCES[s]['publisher'])}</a>"
                                              for s in info["sources"]) + "</p></div>")
    callout(f"<b>Not a diagnosis.</b> {h(edu.DISCLAIMER)} {h(edu.SAFETY_NOTE)}")
    nav_button("Read the potential effects guidance", PAGES[3], "to_effects")


# --------------------------------------------------------------------------- #
# 03 Behavioural Insights
# --------------------------------------------------------------------------- #
def page_insights() -> None:
    eyebrow("03 · Behavioural insights")
    st.markdown("## What the dataset shows, and what it cannot show")
    summary = get_json(config.REPORTS_DIR / config.EDA_SUMMARY_FILE)
    if summary is None:
        show_train_instruction()
        return
    st.write("Three kinds of statement are kept apart on this page: **A. Observed in the dataset**, **B. What the trained model "
             "relies on**, and **C. General sleep-health information** (see Potential Effects). Nothing here proves causation.")
    md(
        "<div class='stats'>"
        f"<div><b>{summary['shape']['rows']}</b><span>rows</span></div>"
        f"<div><b>{summary['shape']['columns']}</b><span>columns</span></div>"
        f"<div><b>{sum(summary['missing_values'].values())}</b><span>missing values</span></div>"
        f"<div><b>{summary['duplicates']['ignoring_person_id']}</b><span>rows repeating another row (ignoring Person ID)</span></div></div>"
    )
    eyebrow("A · Observed in the dataset")
    show = [("quality_vs_duration", "Sleep duration versus recorded sleep quality."),
            ("quality_vs_stress", "Stress level versus recorded sleep quality."),
            ("quality_vs_activity", "Physical activity versus recorded sleep quality."),
            ("target_distribution", "Distribution of recorded sleep-quality scores.")]
    for i in range(0, len(show), 2):
        cols = st.columns(2)
        for col, (key, cap) in zip(cols, show[i:i + 2]):
            with col:
                show_figure(FIGURE_CAPTIONS[key][0], cap)
    st.caption("What these cannot show: why the pattern exists. People who sleep longer may differ in many other ways, and the "
               "dataset records one value per person, not changes over time.")
    corr = summary.get("pearson_correlation_with_target")
    if corr:
        st.markdown("**Pearson correlation with recorded sleep quality**")
        st.dataframe(pd.Series(corr, name="Correlation").to_frame())
    eyebrow("B · What the trained model relies on")
    show_figure("feature_importance.png", "Permutation importance on held-out rows: how much the error rises when a column is shuffled.")
    md(f"<div class='callout'><div class='q'>?</div><p>{h(EXPLAINABILITY_NOTE)}</p></div>")
    with st.expander("More charts"):
        rest = [k for k in FIGURE_CAPTIONS if k not in dict(show) and k != "target_distribution"]
        for key in rest:
            filename, caption = FIGURE_CAPTIONS[key]
            show_figure(filename, caption)
    eyebrow("C · General information")
    st.write("Reference ranges and associated effects from public health sources are on the Potential Effects page. They are not "
             "outputs of the model.")
    nav_button("Open Potential Effects", PAGES[3], "ins_to_eff")


# --------------------------------------------------------------------------- #
# 04 Potential Effects
# --------------------------------------------------------------------------- #
def page_effects() -> None:
    eyebrow("04 · General educational guidance")
    st.markdown("## Potential Effects Associated With Sleep Patterns")
    callout(f"<b>This section is general educational guidance, not an ML-generated prediction.</b> {h(edu.DISCLAIMER)}")
    st.write("Sleep duration is one factor in sleep health, and the relationship between sleep and behaviour is not necessarily causal. "
             "The statements below paraphrase public guidance and describe associations in groups of people.")

    eyebrow("Recommended ranges by age (CDC)")
    table = pd.DataFrame([{"Age group": f"{r['group']}, {r['ages']}", "Recommended sleep": r["text"]} for r in edu.AGE_TABLE])
    st.dataframe(table, hide_index=True)
    st.caption("Ranges for ages 65 and over and for younger children are on the CDC page. Needs vary between individuals.")

    for group, items in edu.effects_by_group().items():
        eyebrow(group)
        for e in items:
            src = edu.SOURCES[e.source]
            md(f"<div class='eff'><h4>{h(e.effect)}</h4><p>{h(e.explanation)}</p>"
               f"<p class='m'><span class='pillless'>EVIDENCE</span>{h(e.evidence)}</p>"
               f"<p class='m'><span class='pillless'>SOURCE</span><a href='{src['url']}' target='_blank'>{h(src['publisher'])}: {h(src['title'])}</a></p>"
               f"<p class='m'>{h(edu.DISCLAIMER)}</p></div>")

    callout(f"<b>When to talk to someone.</b> {h(edu.SAFETY_NOTE)}")
    eyebrow("References")
    for key in ("cdc", "nhlbi", "aasm", "aasm_news"):
        s = edu.SOURCES[key]
        st.markdown(f"- {s['publisher']}. *{s['title']}*. [{s['url']}]({s['url']})")
    st.caption(f"Wording and links were checked on {edu.CHECKED_ON}. Web pages change, so verify them before citing.")


# --------------------------------------------------------------------------- #
# 05 Model Performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    eyebrow("05 · Model performance")
    st.markdown("## How well did the models do?")
    metrics = get_json(config.REPORTS_DIR / config.METRICS_FILE)
    if metrics is None:
        show_train_instruction()
        return
    ds, split, sel = metrics["dataset"], metrics["split"], metrics["model_selection"]
    st.write(
        f"Rows used: **{ds['rows_used']}** ({ds['distinct_feature_combinations']} distinct feature combinations). "
        f"Split {int((1 - split['test_size']) * 100)}/{int(split['test_size'] * 100)}, random_state={split['random_state']}: "
        f"**{split['train_rows']}** training rows, **{split['test_rows']}** test rows. "
        f"Features used: {', '.join(metrics['features']['numeric'] + metrics['features']['categorical'])}."
    )
    st.caption(f"Split strategy: {split['strategy']}. All metrics below come from executing the pipeline.")

    exps = metrics.get("experiments") or {}
    exp_a, exp_b = exps.get("A_duration_only"), exps.get("B_multivariable")
    if exp_a and exp_b:
        eyebrow("Experiment A versus Experiment B")
        a, b = exp_a["test_set"]["selected_model"], exp_b["test_set"]["selected_model"]
        base = exp_a["test_set"]["baseline_training_mean"]
        rows = pd.DataFrame(
            {"Mean baseline": base,
             f"A: sleep duration only ({exp_a['selected_model']})": a,
             f"B: all features ({exp_b['model']})": b}).T.rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"})
        st.dataframe(rows.style.format("{:.3f}"))
        show_figure("experiment_comparison.png", "Held-out RMSE and R² for the baseline and both experiments.")
        st.markdown(
            f"- Sleep duration alone beats the baseline (R² {a['r2']:.2f} versus {base['r2']:.2f}), so duration carries real signal in this dataset.\n"
            f"- Adding the other clues improves held-out R² to {b['r2']:.2f} and cuts RMSE from {a['rmse']:.2f} to {b['rmse']:.2f}. "
            "The duration-only model should be read as a rough estimate.\n"
            f"- The dataset has only {exp_a['distinct_input_values_in_training']} distinct sleep-duration values in the training rows, and the file is highly structured, so even good "
            "scores say little about reliable individual predictions."
        )
        with st.expander("Cross-validation for Experiment A (training split only)"):
            st.dataframe(pd.DataFrame({k: {"RMSE (mean ± std)": f"{v['rmse_mean']:.3f} ± {v['rmse_std']:.3f}",
                                           "R² (mean)": f"{v['r2_mean']:.3f}"} for k, v in exp_a["cv_results"].items()}).T)
        show_figure("actual_vs_predicted_duration_only.png", "Experiment A: recorded versus predicted scores on test rows.")

    net = metrics.get("network")
    if net:
        eyebrow("Neural network architecture")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_figure("network_architecture.png", "Fully connected feed-forward network (a sample of nodes is drawn).")
        with c2:
            st.markdown(
                f"- **Implementation:** {net['implementation']}\n"
                f"- **Layers:** {' → '.join(str(n) for n in net['layer_sizes'])}\n"
                f"- **Activation:** {net['activation'].upper()} (hidden), {net['output_activation']} output\n"
                f"- **Optimiser:** {net['optimizer']}, learning rate {net['learning_rate_init']}\n"
                f"- **Regularisation:** L2 penalty {net['l2_alpha']}" + (", early stopping" if net["early_stopping"] else "") + "\n"
                f"- **Epochs run:** {net['epochs_run']}\n"
                f"- **Trainable parameters:** {net['trainable_parameters']:,}"
            )
        show_figure("training_curve.png", "Training loss and validation R² by epoch.")

    eyebrow("Cross-validation (training split only, used to compare models)")
    cv_rows = {name: {"RMSE (mean ± std)": f"{v['rmse_mean']:.3f} ± {v['rmse_std']:.3f}",
                      "MAE (mean ± std)": f"{v['mae_mean']:.3f} ± {v['mae_std']:.3f}",
                      "R² (mean ± std)": f"{v['r2_mean']:.3f} ± {v['r2_std']:.3f}"} for name, v in sel["cv_results"].items()}
    st.dataframe(pd.DataFrame(cv_rows).T)
    st.write(f"**Deployed model:** {sel['selected_model']}. **Best by cross-validation:** {sel['best_by_cv']}.")
    st.caption(f"Why: {sel['deployed_reason']}. Cross-validation: {sel['cv']}.")
    show_figure("model_comparison.png", "Cross-validated RMSE by model (training split).")

    eyebrow("Final held-out test results (evaluated once)")
    test = metrics["test_set"]
    table = pd.DataFrame({sel["selected_model"] + " (deployed)": test["selected_model"],
                          "Baseline (training mean)": test["baseline_training_mean"]}
                         ).T.rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"})
    st.dataframe(table.style.format("{:.3f}"))
    st.caption("MAE and RMSE are in sleep-quality points. R² compares the model with always predicting the mean.")
    with st.expander("Test results for every candidate model (for transparency only)"):
        st.dataframe(pd.DataFrame(test["all_models"]).T.rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"}).style.format("{:.3f}"))
        st.caption(test["all_models_note"])
    c1, c2 = st.columns(2)
    with c1:
        show_figure("actual_vs_predicted.png", "Recorded versus predicted scores on test rows.")
    with c2:
        show_figure("residuals.png", "Residuals on test rows.")

    eyebrow("How reliable are these numbers?")
    naive = metrics["diagnostics"]["naive_random_split_80_20"]
    fit = metrics["training_set_fit"]
    st.markdown(
        f"- The test set holds only **{split['test_distinct_feature_combinations']}** distinct feature combinations, so the estimate is uncertain; "
        "cross-validation standard deviations show how much results vary.\n"
        f"- Training R² is {fit['r2']:.3f} and test R² is {test['selected_model']['r2']:.3f}, so the model fits seen data better than unseen data.\n"
        f"- A plain random split (not used) would give {naive['share_of_test_rows_with_identical_feature_twin_in_train'] * 100:.0f}% of test rows an identical twin in training "
        f"and report an optimistic R² of {naive['metrics']['r2']:.3f}.\n"
        "- Predictions are clipped to 1 to 10 for display only; metrics use raw model output."
    )


# --------------------------------------------------------------------------- #
# 06 About
# --------------------------------------------------------------------------- #
def page_about() -> None:
    eyebrow("06 · About the project")
    st.markdown("## About SleepWise AI")
    st.markdown(
        """
**Objective.** Estimate a recorded sleep-quality score from sleep duration and lifestyle clues, show what the model relies on, and keep general sleep-health education separate from the model.

**Dataset.** The Kaggle *Sleep Health and Lifestyle Dataset* CSV supplied by the project author (see `DATA_SOURCES.md`). The file carries no documentation of how its values were collected, and the project does not claim they are verified clinical observations.

**Methodology.** Validate and clean the CSV, split blood pressure into systolic and diastolic values, build a scikit-learn `Pipeline` (median imputation and scaling for numbers; most-frequent imputation and one-hot encoding for categories, unseen categories ignored), keep rows with identical inputs on the same side of an 80/20 split, compare a mean baseline, Linear Regression, Decision Tree, Random Forest and two neural networks with grouped 5-fold cross-validation on the training split, run Experiment A (sleep duration only) against Experiment B (all features), then evaluate once on held-out rows. The deployed multivariable model is the scikit-learn neural network (two hidden ReLU layers, Adam, L2 penalty, early stopping).

**Technology.** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib, pytest.

**Excluded columns.** Person ID (identifier), Quality of Sleep (target) and Sleep Disorder (a recorded outcome that could leak information about the target).

**Ethics.** Sleep quality is health-related. Outputs are estimates for learning, not diagnoses or advice. Gender and occupation are inputs because they are in the dataset; that does not make them appropriate for decisions about individuals.

**Known limitations.** Small dataset with many repeated rows; associations, not causes; narrow value ranges; no uncertainty estimate; one record per person; the model cannot forecast future sleep or detect sleep disorders; educational guidance is paraphrased from public sources and should be verified.

**Future improvements.** Independent validation data, prediction intervals, nested cross-validation, richer behavioural features if a suitable dataset exists, fairness checks, optional SHAP explanations.
"""
    )


# --------------------------------------------------------------------------- #
# Shell: brand, navigation, page, footer
# --------------------------------------------------------------------------- #
md("<div class='sw-brand'><div><b>SleepWise AI</b><span>SLEEP, EXPLAINED</span></div>"
   "<span style='margin:0'>EDUCATIONAL PROTOTYPE</span></div>")
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGES[0]
page = st.radio("Navigate", PAGES, key="nav", horizontal=True, label_visibility="collapsed")

{
    PAGES[0]: page_home,
    PAGES[1]: page_analysis,
    PAGES[2]: page_insights,
    PAGES[3]: page_effects,
    PAGES[4]: page_performance,
    PAGES[5]: page_about,
}[page]()

md("<div class='sw-foot'><span><b>SleepWise AI</b> / EDUCATIONAL SLEEP ANALYSIS</span>"
   "<span>A course project, built for learning. Not a medical device.</span></div>")

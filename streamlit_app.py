import json
import os
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GeneScreen AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"

with open(f"{MODEL_DIR}/feature_schema.json", "r") as f:
    schema = json.load(f)

FEATURES = schema["features"]
NUMERIC = schema.get("numeric_features", [])
CATEGORICAL = schema.get("categorical_features", [])
CLASSES = schema.get("classes", [])

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #142b3b;
    --muted: #687f8c;
    --teal: #078f88;
    --pale: #eaf8f6;
    --line: #deecef;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #f5f9fb;
    color: var(--ink);
}
.block-container {
    max-width: 1450px;
    padding-top: 1.3rem;
    padding-bottom: 2.5rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#102a3a,#123e4a);
}
[data-testid="stSidebar"] * {
    color: #f0fbfc;
}
.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -.5px;
}
.brand-sub {
    color: #b4d3dc;
    font-size: .78rem;
    margin-top: -4px;
}
.hero {
    background: linear-gradient(120deg,#ffffff 0%,#e4f8f6 100%);
    border: 1px solid #d7eeec;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 20px;
}
.eyebrow {
    color: #078f88;
    font-weight: 700;
    font-size: .76rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    color: #142b3b;
    font-size: clamp(1.8rem,3vw,2.8rem);
    line-height: 1.12;
    margin: 10px 0;
}
.hero p {
    color: #5f7784;
    font-size: 1rem;
    max-width: 670px;
}
.panel {
    background: white;
    border: 1px solid #e1ecef;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
}
.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #173447;
}
.panel-sub {
    font-size: .84rem;
    color: #708692;
    margin: 4px 0 12px;
}
.step {
    background: white;
    border: 1px solid #deeaed;
    border-radius: 14px;
    text-align: center;
    padding: 12px 6px;
    color: #708692;
    font-size: .82rem;
}
.step-active {
    background: #e3f7f4;
    border: 1px solid #76d2c8;
    color: #087e78;
    font-weight: 700;
}
.result {
    background: linear-gradient(135deg,#e7faf6,#f9ffff);
    border: 1px solid #b9e8df;
    border-radius: 20px;
    padding: 26px;
}
.result-name {
    font-family: 'Manrope', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #116b66;
}
.small-note {
    background: #fff8e8;
    border: 1px solid #f3dfac;
    border-radius: 12px;
    color: #775a20;
    padding: 12px 15px;
    font-size: .84rem;
}
div.stButton > button[kind="primary"] {
    background: #078f88;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 2.7rem;
}
div.stButton > button {
    border-radius: 10px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1ecef;
    padding: 16px;
    border-radius: 15px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="brand">🧬 GeneScreen AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Genetic Subclass Screening</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Home", "New Screening", "Model Performance", "About"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("**🔬 Research Prototype**")
    st.caption("Educational use only. Not a medical diagnosis.")

# ---------------- HELPERS ----------------
def load_model(kernel):
    path = os.path.join(MODEL_DIR, f"svm_{kernel.lower()}.joblib")
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)

def field_kind(name):
    return "numeric" if name in NUMERIC else "categorical"

# Keep each dataset feature exactly once.
groups = {
    "Patient & Family Information": [
        "Patient Id", "Patient Age", "Gender",
        "Genes in mother's side", "Inherited from father",
        "Maternal gene", "Paternal gene", "Mother's age", "Father's age",
    ],
    "Medical Measurements": [
        "Blood cell count (mcL)",
        "White Blood cell count (thousand per microliter)",
        "Respiratory Rate (breaths/min)", "Heart Rate (rates/min",
    ],
    "Tests & Blood Results": [
        "Test 1", "Test 2", "Test 3", "Test 4", "Test 5",
        "Blood test result", "Status",
    ],
    "Medical History": [
        "Parental consent", "Follow-up", "Birth asphyxia",
        "Autopsy shows birth defect (if applicable)",
        "Folic acid details (peri-conceptional)",
        "H/O serious maternal illness", "H/O radiation exposure (x-ray)",
        "H/O substance abuse", "Assisted conception IVF/ART",
        "History of anomalies in previous pregnancies",
        "No. of previous abortion", "Birth defects",
    ],
    "Symptoms": [
        "Symptom 1", "Symptom 2", "Symptom 3",
        "Symptom 4", "Symptom 5",
    ],
}

# Category choices are common dataset values. A blank choice is always available.
CATEGORY_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Genes in mother's side": ["Yes", "No"],
    "Inherited from father": ["Yes", "No"],
    "Maternal gene": ["Yes", "No"],
    "Paternal gene": ["Yes", "No"],
    "Status": ["Alive", "Deceased"],
    "Respiratory Rate (breaths/min)": ["Normal", "Tachypnea"],
    "Heart Rate (rates/min": ["Normal", "Tachycardia"],
    "Parental consent": ["Yes", "No"],
    "Follow-up": ["High", "Low"],
    "Birth asphyxia": ["Yes", "No"],
    "Autopsy shows birth defect (if applicable)": [
        "Not applicable", "Yes", "No"
    ],
    "Folic acid details (peri-conceptional)": ["Yes", "No"],
    "H/O serious maternal illness": ["Yes", "No"],
    "H/O radiation exposure (x-ray)": ["Yes", "No"],
    "H/O substance abuse": ["Yes", "No"],
    "Assisted conception IVF/ART": ["Yes", "No"],
    "History of anomalies in previous pregnancies": ["Yes", "No"],
    "Birth defects": ["Singular", "Multiple"],
    "Blood test result": [
        "Normal", "Abnormal", "Inconclusive", "Not available"
    ],
}

def render_input(name, key_prefix):
    key = f"{key_prefix}_{name}"
    if field_kind(name) == "numeric":
        # Blank values are handled as missing values by the trained pipeline.
        return st.number_input(
            name, value=None, placeholder="Leave blank if unknown",
            key=key, step=1.0, format="%g"
        )
    if name == "Patient Id":
        return st.text_input(
            name, placeholder="Optional sample ID",
            key=key
        )
    options = ["— Select if known —"] + CATEGORY_OPTIONS.get(name, [])
    choice = st.selectbox(name, options, key=key)
    return None if choice == options[0] else choice

def render_group(title, fields, key_prefix, columns=2):
    fields = [f for f in fields if f in FEATURES]
    if not fields:
        return {}
    values = {}
    st.markdown(
        f'<div class="panel"><div class="panel-title">{html.escape(title)}</div>'
        '<div class="panel-sub">Fill in what is available. Unknown fields can be left blank.</div>',
        unsafe_allow_html=True
    )
    cols = st.columns(columns)
    for i, name in enumerate(fields):
        with cols[i % columns]:
            values[name] = render_input(name, key_prefix)
    st.markdown("</div>", unsafe_allow_html=True)
    return values

def reset_form():
    for key in list(st.session_state.keys()):
        if key.startswith("form_"):
            del st.session_state[key]
    st.session_state.pop("last_prediction", None)

# ---------------- HOME ----------------
if page == "Home":
    left, right = st.columns([1.5, 1], vertical_alignment="center")
    with left:
        st.markdown("""
        <div class="hero">
          <div class="eyebrow">AI · Machine Learning · Research</div>
          <h1>Genetic Disorder<br>Subclass Screening</h1>
          <p>Explore an experimental machine-learning model that predicts a
          genetic-disorder subclass from dataset features.</p>
          <p><b>Educational research prototype — not a medical diagnosis.</b></p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("＋  Start New Screening", type="primary", use_container_width=True):
            st.session_state["page_override"] = "New Screening"
            st.rerun()
    with right:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:35px 18px">
          <div style="font-size:5rem">🧬</div>
          <div class="panel-title">Machine Learning Screening</div>
          <div class="panel-sub">SVM · Linear · Polynomial · RBF</div>
          <hr>
          <p style="color:#078f88;font-weight:700">Simple · Guided · Research-focused</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### What you can do")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><h3>🧾 Guided Input</h3><p>Enter sample features in clear, separate steps.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><h3>🧠 SVM Prediction</h3><p>Compare predictions from the available trained models.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><h3>📊 Model Metrics</h3><p>Review saved model comparison metrics.</p></div>', unsafe_allow_html=True)
    st.info("For your first test, use a row from your existing dataset. Avoid entering real patient details.")

# ---------------- NEW SCREENING ----------------
elif page == "New Screening":
    st.markdown("## New Screening")
    st.caption("Use an existing dataset row for testing. Leave unknown values blank.")
    if "form_step" not in st.session_state:
        st.session_state.form_step = 0

    step_names = [
        "Patient & Family",
        "Measurements & Tests",
        "History & Symptoms",
        "Review & Predict",
    ]
    step_cols = st.columns(4)
    for i, label in enumerate(step_names):
        with step_cols[i]:
            cls = "step step-active" if i == st.session_state.form_step else "step"
            st.markdown(
                f'<div class="{cls}"><b>{i+1}</b><br>{html.escape(label)}</div>',
                unsafe_allow_html=True
            )
    st.write("")

    if st.session_state.form_step == 0:
        render_group(
            "Patient & Family Information",
            groups["Patient & Family Information"], "form"
        )
    elif st.session_state.form_step == 1:
        c1, c2 = st.columns(2)
        with c1:
            render_group("Medical Measurements", groups["Medical Measurements"], "form", 1)
        with c2:
            render_group("Tests & Blood Results", groups["Tests & Blood Results"], "form", 2)
    elif st.session_state.form_step == 2:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            render_group("Medical History", groups["Medical History"], "form", 2)
        with c2:
            render_group("Symptoms", groups["Symptoms"], "form", 2)
    else:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Review & Predict</div>
          <div class="panel-sub">Choose the SVM model. Check your entries before running a prediction.</div>
        </div>
        """, unsafe_allow_html=True)
        kernel = st.selectbox("SVM model", ["RBF", "Linear", "Polynomial"])
        st.session_state["chosen_kernel"] = kernel
        st.caption("Unknown inputs will be passed as missing values; model preprocessing handles them only if supported by the saved pipeline.")

        if st.button("Run Prediction", type="primary", use_container_width=True):
            model = load_model(kernel.lower())
            row = {}
            for name in FEATURES:
                value = st.session_state.get(f"form_{name}")
                if value == "— Select if known —":
                    value = None
                row[name] = value

            X = pd.DataFrame([row], columns=FEATURES)
            try:
                prediction = model.predict(X)[0]
                st.session_state["last_prediction"] = {
                    "prediction": str(prediction),
                    "kernel": kernel,
                }
                st.success("Prediction generated.")
            except Exception as e:
                st.error("The model could not process these inputs. Please check that the saved model is compatible with the feature schema.")
                st.code(str(e))

        result = st.session_state.get("last_prediction")
        if result:
            st.markdown(f"""
            <div class="result">
              <div class="eyebrow">Prediction Result · {html.escape(result['kernel'])} SVM</div>
              <div style="font-size:1rem;color:#426d72;margin-top:12px">Predicted subclass</div>
              <div class="result-name">{html.escape(result['prediction'])}</div>
              <p style="color:#58767b">Experimental model output only. It is not a confirmed diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Do not use this result for treatment or medical decisions.")

    left, right = st.columns([1, 1])
    with left:
        if st.session_state.form_step > 0:
            if st.button("← Previous Step", use_container_width=True):
                st.session_state.form_step -= 1
                st.rerun()
    with right:
        if st.session_state.form_step < 3:
            if st.button("Continue →", type="primary", use_container_width=True):
                st.session_state.form_step += 1
                st.rerun()
        else:
            if st.button("＋ Start Over", use_container_width=True):
                reset_form()
                st.session_state.form_step = 0
                st.rerun()

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.caption("Metrics saved during your model evaluation. Values are not recalculated here.")
    metrics_path = "model_comparison.csv"
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        numeric_cols = metrics.select_dtypes(include=np.number).columns.tolist()
        label_col = next(
            (c for c in metrics.columns if c.lower() in ["model", "kernel", "svm model"]),
            metrics.columns[0]
        )
        metric_col = next(
            (c for c in numeric_cols if "accuracy" in c.lower()),
            numeric_cols[0] if numeric_cols else None
        )
        if metric_col:
            chart = metrics[[label_col, metric_col]].copy().set_index(label_col)
            st.markdown("### Model comparison")
            st.bar_chart(chart)
    else:
        st.info("model_comparison.csv was not found. Run your model evaluation cell and save the comparison table first.")

    st.markdown("""
    <div class="small-note">
      <b>Interpretation:</b> Accuracy alone is not enough for imbalanced multi-class data.
      Review macro F1-score and per-class performance as well.
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown("## About GeneScreen AI")
    st.markdown("""
    <div class="panel">
      <div class="panel-title">Project Overview</div>
      <p>This educational research prototype uses trained Support Vector Machine
      models to predict a disorder subclass from input features.</p>
      <p><b>Available models:</b> Linear SVM, Polynomial SVM, RBF SVM.</p>
      <p><b>Purpose:</b> Machine-learning experimentation and educational demonstration.</p>
    </div>
    <div class="small-note">
      This prototype is not clinically validated, does not diagnose genetic disorders,
      and must not be used to make medical or treatment decisions.
    </div>
    """, unsafe_allow_html=True)

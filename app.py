# =============================================================
#  NCNSI (Neurosurgical Central Nervous System Infection)
#  Risk Prediction — Streamlit App
#  Model: Gradient Boosting (sklearn), 11 selected features
# =============================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Risk Prediction of Intracranial Infection",
    page_icon="🧠",
    layout="wide",
)

# ---------------- Custom CSS (参考 Shiny 风格) ----------------
st.markdown("""
<style>
.stApp { background-color:#eef1f5; }
.app-header{
  background:linear-gradient(120deg,#163a6e 0%,#1c5a8f 45%,#1f7a6d 100%);
  color:#fff;padding:28px 20px;border-radius:0 0 14px 14px;
  margin-bottom:22px;box-shadow:0 4px 14px rgba(20,50,90,.25);text-align:center;
}
.app-header h1{font-size:26px;font-weight:700;margin:0;}
.app-header p{font-size:13.5px;opacity:.9;margin:8px 0 0 0;}
.card{background:#fff;border-radius:14px;padding:18px 22px;
  box-shadow:0 3px 14px rgba(30,60,110,.07);border:1px solid #e6ebf1;margin-bottom:18px;}
.card-title{font-size:15px;font-weight:700;color:#163a6e;margin-bottom:14px;
  padding-bottom:9px;border-bottom:2px solid #eaf0f7;}
.result-prob{font-size:44px;font-weight:800;margin:6px 0;}
.badge{display:inline-block;padding:8px 18px;border-radius:22px;color:#fff;
  font-weight:700;font-size:16px;}
.legend-item{font-size:13px;color:#4b5566;margin:5px 0;}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px;}
.dot-low{background:#3aa76d;} .dot-mid{background:#e6b800;} .dot-high{background:#d9534f;}
.disclaimer{font-size:11.5px;color:#c98a2b;margin-top:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>Risk Prediction of Intracranial Infection</h1>
  <p>An interactive tool based on a Gradient Boosting model · For research use only</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Feature definitions ----------------
TARGET = "HAI"

FEATURES = [
    "Duration of Surgery",
    "Post Operative WBC",
    "Post Operative Neutrophil Percentage",
    "Cerebrospinal Fluid WBC",
    "Cerebrospinal Fluid Characteristic",
    "Post Operative Cranial Hypertension",
    "Cerebrospinal Fluid Glucose Abnormality",
    "Cerebrospinal Fluid Culture",
    "EVD",
    "LD",
    "Use of Immuno-suppressive Drugs",
]

# 选项映射：label -> 编码值
CATEGORICAL = {
    "Duration of Surgery": {
        "< 2 h": 1, "2–3 h": 2, "3–4 h": 3, "4–5 h": 4, "≥ 5 h": 5},
    "Post Operative WBC": {
        "< 10": 1, "10–20": 2, "≥ 20": 3},
    "Post Operative Neutrophil Percentage": {
        "40–70 %": 1, "70–80 %": 2, "≥ 80 %": 3},
    "Cerebrospinal Fluid WBC": {
        "< 10": 1, "10–100": 2, "100–1000": 3, "1000–10000": 4, "≥ 10000": 5},
    "Cerebrospinal Fluid Characteristic": {
        "Clear": 0, "Turbid": 1},
    "Post Operative Cranial Hypertension": {"No": 0, "Yes": 1},
    "Cerebrospinal Fluid Glucose Abnormality": {"No": 0, "Yes": 1},
    "Cerebrospinal Fluid Culture": {"No": 0, "Yes": 1},
    "EVD": {"No": 0, "Yes": 1},
    "LD": {"No": 0, "Yes": 1},
    "Use of Immuno-suppressive Drugs": {"No": 0, "Yes": 1},
}

# ---------------- Train / cache model ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("LUNEI_SELECTED_variables.csv")
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_tr

model, explainer, X_background = load_model()

# ---------------- Layout ----------------
col_in, col_out = st.columns([1.05, 0.95])

with col_in:
    st.markdown('<div class="card"><div class="card-title">Predictor Variables</div>',
                unsafe_allow_html=True)
    inputs = {}
    for feat in FEATURES:
        opts = list(CATEGORICAL[feat].keys())
        choice = st.selectbox(feat, opts, key=feat)
        inputs[feat] = CATEGORICAL[feat][choice]
    predict = st.button("🩺  Predict Risk", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_out:
    st.markdown('<div class="card"><div class="card-title">Prediction Result</div>',
                unsafe_allow_html=True)
    if predict:
        x = pd.DataFrame([inputs])[FEATURES]
        prob = float(model.predict_proba(x)[0, 1])

        if prob < 0.30:
            level, color = "Low risk", "#3aa76d"
        elif prob < 0.60:
            level, color = "Intermediate risk", "#e6b800"
        else:
            level, color = "High risk", "#d9534f"

        st.markdown(
            f'<div class="result-prob" style="color:{color}">{prob*100:.1f}%</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<span class="badge" style="background:{color}">{level}</span>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(prob, 1.0))

        st.markdown(
            '<div class="legend-item"><span class="dot dot-low"></span>Low risk: &lt; 30%</div>'
            '<div class="legend-item"><span class="dot dot-mid"></span>Intermediate risk: 30% – 60%</div>'
            '<div class="legend-item"><span class="dot dot-high"></span>High risk: &gt; 60%</div>',
            unsafe_allow_html=True)

        # ---- SHAP individual explanation ----
        st.markdown("---")
        st.markdown("**SHAP explanation (individual)**")
        sv = explainer(x)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        shap.plots.waterfall(sv[0], max_display=11, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Set the predictors on the left, then click **Predict Risk**.")

    st.markdown(
        '<div class="disclaimer">⚠️ This tool is intended for research purposes only '
        'and does not constitute clinical diagnosis or treatment advice.</div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------
st.markdown(
    """
    <style>
    body { background-color: #f8fafc; }
    .card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .card-title {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 600;
    }
    .risk-high { color: #b91c1c; }
    .risk-low { color: #047857; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, encoder, scaler, feature_columns

model, target_encoder, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# ENCODING FUNCTIONS
# --------------------------------------------------
def yes_no(v): return 1 if v == "Yes" else 0
def good_poor(v): return 1 if v == "Poor" else 0
def encode_smoking(v): return {"Never": 0, "Former": 1, "Current": 2}[v]
def encode_activity(v): return {"Low": 0, "Moderate": 1, "High": 2}[v]
def encode_sediment(v): return {"Normal": 0, "Abnormal": 1}[v]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction")

st.caption(
    "A machine learning–based clinical decision support system designed to assist "
    "healthcare professionals in early detection and risk assessment of chronic kidney disease "
    "using patient clinical, biochemical, and lifestyle indicators."
)

# --------------------------------------------------
# KIDNEY IMAGE
# --------------------------------------------------
try:
    img = Image.open(BASE_DIR / "kidney.png")
    st.image(img, caption="Human Kidney Anatomy", use_container_width=True)
except:
    pass

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Clinical Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_data["Age of the patient"] = st.number_input("Age", 0, 120)
        input_data["Blood pressure (mm/Hg)"] = st.number_input("Blood Pressure")
    with c2:
        input_data["Specific gravity of urine"] = st.number_input("Specific Gravity")
        input_data["Albumin in urine"] = st.number_input("Albumin")
    with c3:
        input_data["Sugar in urine"] = st.number_input("Sugar")
        input_data["Random blood glucose level (mg/dl)"] = st.number_input("Random Blood Glucose")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_data["Blood urea (mg/dl)"] = st.number_input("Blood Urea")
        input_data["Serum creatinine (mg/dl)"] = st.number_input("Serum Creatinine")
    with c2:
        input_data["Sodium level (mEq/L)"] = st.number_input("Sodium")
        input_data["Potassium level (mEq/L)"] = st.number_input("Potassium")
    with c3:
        input_data["Hemoglobin level (gms)"] = st.number_input("Hemoglobin")
        input_data["Packed cell volume (%)"] = st.number_input("Packed Cell Volume")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_data["White blood cell count (cells/cumm)"] = st.number_input("WBC Count")
    with c2:
        input_data["Red blood cell count (millions/cumm)"] = st.number_input("RBC Count")
    with c3:
        input_data["Estimated Glomerular Filtration Rate (eGFR)"] = st.number_input("eGFR")

    st.subheader("Medical History")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_data["Hypertension (yes/no)"] = yes_no(st.selectbox("Hypertension", ["No", "Yes"]))
        input_data["Diabetes mellitus (yes/no)"] = yes_no(st.selectbox("Diabetes", ["No", "Yes"]))
    with c2:
        input_data["Coronary artery disease (yes/no)"] = yes_no(st.selectbox("CAD", ["No", "Yes"]))
        input_data["Anemia (yes/no)"] = yes_no(st.selectbox("Anemia", ["No", "Yes"]))
    with c3:
        input_data["Smoking status"] = encode_smoking(st.selectbox("Smoking", ["Never", "Former", "Current"]))
        input_data["Physical activity level"] = encode_activity(st.selectbox("Activity", ["Low", "Moderate", "High"]))

    st.subheader("Inflammatory Markers")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_data["Cystatin C level"] = st.number_input("Cystatin C")
    with c2:
        input_data["C-reactive protein (CRP) level"] = st.number_input("CRP")
    with c3:
        input_data["Interleukin-6 (IL-6) level"] = st.number_input("IL-6")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submit:
    # Create DataFrame from user input
    df = pd.DataFrame([input_data])

    # Reindex to match feature columns of the trained model
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Convert to numpy array before scaling
    try:
        df_scaled = scaler.transform(df.values)
        df = pd.DataFrame(df_scaled, columns=feature_columns)
    except ValueError as e:
        st.error(f"Error in data preprocessing: {e}")
        st.stop()


    # --------------------------------------------------
    # RESULT DASHBOARD
    # --------------------------------------------------
    st.subheader("Prediction Outcome")

    risk_class = "risk-high" if top_category.lower() in ["ckd", "yes", "positive"] else "risk-low"
    conf_label = "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low"

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><div class='card-title'>Clinical Assessment</div><div class='card-value {risk_class}'>{top_category}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='card-title'>Risk Probability</div><div class='card-value'>{top_prob}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='card-title'>Model Confidence</div><div class='card-value'>{conf_label}</div></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # PROBABILITY TABLE
    # --------------------------------------------------
    prob_df = pd.DataFrame({
        "Risk Category": target_encoder.classes_,
        "Probability (%)": np.round(probs, 2)
    }).sort_values("Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(prob_df, use_container_width=True)

    # --------------------------------------------------
    # VISUALIZATION (VERTICAL, NORMAL PIE)
    # --------------------------------------------------
    st.subheader("📈 Risk Probability Distribution")

    pie_chart = alt.Chart(prob_df).mark_arc().encode(
        theta="Probability (%):Q",
        color="Risk Category:N",
        tooltip=["Risk Category", "Probability (%)"]
    ).properties(width=350, height=350)

    st.altair_chart(pie_chart, use_container_width=False)

    bar_chart = alt.Chart(prob_df).mark_bar().encode(
        x="Risk Category:N",
        y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Risk Category:N", legend=None),
        tooltip=["Risk Category", "Probability (%)"]
    ).properties(width=700, height=350)

    st.altair_chart(bar_chart, use_container_width=False)


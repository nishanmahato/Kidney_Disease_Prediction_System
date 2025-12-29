import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="centered",
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
        color: #111827;
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
def yes_no(val): return 1 if val == "Yes" else 0
def good_poor(val): return 1 if val == "Poor" else 0
def encode_smoking(val): return {"Never": 0, "Former": 1, "Current": 2}[val]
def encode_activity(val): return {"Low": 0, "Moderate": 1, "High": 2}[val]
def encode_sediment(val): return {"Normal": 0, "Abnormal": 1}[val]

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction")

st.caption(
    "An intelligent clinical decision support system that leverages machine learning "
    "to assist in the early detection and risk assessment of chronic kidney disease "
    "based on patient clinical, laboratory, and lifestyle data."
)

# --------------------------------------------------
# KIDNEY IMAGE (ATTRACTIVE UI)
# --------------------------------------------------
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Kidney_diagram_en.svg/512px-Kidney_diagram_en.svg.png",
    caption="Human Kidney Anatomy",
    width=300,
)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Clinical Information")

    c1, c2 = st.columns(2)

    with c1:
        input_data["Age of the patient"] = st.number_input("Age (years)", 0, 120)
        input_data["Blood pressure (mm/Hg)"] = st.number_input("Blood Pressure")
        input_data["Specific gravity of urine"] = st.number_input("Specific Gravity")
        input_data["Albumin in urine"] = st.number_input("Albumin in Urine")
        input_data["Sugar in urine"] = st.number_input("Sugar in Urine")
        input_data["Random blood glucose level (mg/dl)"] = st.number_input("Random Blood Glucose")
        input_data["Blood urea (mg/dl)"] = st.number_input("Blood Urea")
        input_data["Serum creatinine (mg/dl)"] = st.number_input("Serum Creatinine")
        input_data["Sodium level (mEq/L)"] = st.number_input("Sodium Level")
        input_data["Potassium level (mEq/L)"] = st.number_input("Potassium Level")
        input_data["Hemoglobin level (gms)"] = st.number_input("Hemoglobin")
        input_data["Packed cell volume (%)"] = st.number_input("Packed Cell Volume")

    with c2:
        input_data["White blood cell count (cells/cumm)"] = st.number_input("WBC Count")
        input_data["Red blood cell count (millions/cumm)"] = st.number_input("RBC Count")
        input_data["Estimated Glomerular Filtration Rate (eGFR)"] = st.number_input("eGFR")
        input_data["Urine protein-to-creatinine ratio"] = st.number_input("Protein/Creatinine Ratio")
        input_data["Urine output (ml/day)"] = st.number_input("Urine Output")
        input_data["Serum albumin level"] = st.number_input("Serum Albumin")
        input_data["Cholesterol level"] = st.number_input("Cholesterol")
        input_data["Parathyroid hormone (PTH) level"] = st.number_input("PTH")
        input_data["Serum calcium level"] = st.number_input("Calcium")
        input_data["Serum phosphate level"] = st.number_input("Phosphate")
        input_data["Body Mass Index (BMI)"] = st.number_input("BMI")

    st.subheader("Medical History")

    c3, c4 = st.columns(2)

    with c3:
        input_data["Hypertension (yes/no)"] = yes_no(st.selectbox("Hypertension", ["No", "Yes"]))
        input_data["Diabetes mellitus (yes/no)"] = yes_no(st.selectbox("Diabetes", ["No", "Yes"]))
        input_data["Coronary artery disease (yes/no)"] = yes_no(st.selectbox("CAD", ["No", "Yes"]))
        input_data["Pedal edema (yes/no)"] = yes_no(st.selectbox("Pedal Edema", ["No", "Yes"]))
        input_data["Anemia (yes/no)"] = yes_no(st.selectbox("Anemia", ["No", "Yes"]))

    with c4:
        input_data["Family history of chronic kidney disease"] = yes_no(st.selectbox("Family History of CKD", ["No", "Yes"]))
        input_data["Appetite (good/poor)"] = good_poor(st.selectbox("Appetite", ["Good", "Poor"]))
        input_data["Smoking status"] = encode_smoking(st.selectbox("Smoking Status", ["Never", "Former", "Current"]))
        input_data["Physical activity level"] = encode_activity(st.selectbox("Physical Activity", ["Low", "Moderate", "High"]))
        input_data["Urinary sediment microscopy results"] = encode_sediment(st.selectbox("Urinary Sediment", ["Normal", "Abnormal"]))

    st.subheader("Inflammatory Markers")
    input_data["Cystatin C level"] = st.number_input("Cystatin C")
    input_data["C-reactive protein (CRP) level"] = st.number_input("CRP")
    input_data["Interleukin-6 (IL-6) level"] = st.number_input("IL-6")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    df = pd.DataFrame(scaler.transform(df), columns=feature_columns)

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0] * 100

    top_idx = np.argmax(probs)
    top_category = target_encoder.classes_[top_idx]
    top_prob = round(probs[top_idx], 2)

    st.subheader("Prediction Outcome")
    st.success(f"Predicted Condition: **{top_category}**")
    st.info(f"Risk Probability: **{top_prob}%**")


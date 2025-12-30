import warnings
warnings.filterwarnings("ignore")

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
    layout="wide"
)

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    target_encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, target_encoder, scaler, feature_columns

model, target_encoder, scaler, feature_columns = load_artifacts()

# --------------------------------------------------
# ENCODING MAPS (MATCH TRAINING)
# --------------------------------------------------
binary_map = {
    "yes": 1, "no": 0,
    "present": 1, "not present": 0,
    "abnormal": 1, "normal": 0,
    "poor": 1, "good": 0
}

activity_map = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction 🩺")
st.markdown(
    "Machine-learning–based system for early assessment of chronic kidney disease using clinical and lifestyle data."
)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["Age of the patient"] = st.number_input("Age (years)", 0, 120)
        input_data["Blood pressure (mm/Hg)"] = st.number_input("Blood Pressure")
        input_data["Specific gravity of urine"] = st.number_input("Specific Gravity")
        input_data["Albumin in urine"] = st.number_input("Albumin")
        input_data["Sugar in urine"] = st.number_input("Sugar")

    with col2:
        input_data["Random blood glucose level (mg/dl)"] = st.number_input("Random Blood Glucose")
        input_data["Blood urea (mg/dl)"] = st.number_input("Blood Urea")
        input_data["Serum creatinine (mg/dl)"] = st.number_input("Serum Creatinine")
        input_data["Sodium level (mEq/L)"] = st.number_input("Sodium")
        input_data["Potassium level (mEq/L)"] = st.number_input("Potassium")

    with col3:
        input_data["Hemoglobin level (gms)"] = st.number_input("Hemoglobin")
        input_data["Packed cell volume (%)"] = st.number_input("Packed Cell Volume")
        input_data["White blood cell count (cells/cumm)"] = st.number_input("WBC Count")
        input_data["Red blood cell count (millions/cumm)"] = st.number_input("RBC Count")
        input_data["Estimated Glomerular Filtration Rate (eGFR)"] = st.number_input("eGFR")

    st.subheader("Clinical Conditions")

    col4, col5, col6 = st.columns(3)

    with col4:
        input_data["Red blood cells in urine"] = binary_map[
            st.selectbox("Red Blood Cells in Urine", ["normal", "abnormal"])
        ]
        input_data["Pus cells in urine"] = binary_map[
            st.selectbox("Pus Cells in Urine", ["normal", "abnormal"])
        ]
        input_data["Pus cell clumps in urine"] = binary_map[
            st.selectbox("Pus Cell Clumps", ["not present", "present"])
        ]

    with col5:
        input_data["Bacteria in urine"] = binary_map[
            st.selectbox("Bacteria in Urine", ["not present", "present"])
        ]
        input_data["Hypertension (yes/no)"] = binary_map[
            st.selectbox("Hypertension", ["no", "yes"])
        ]
        input_data["Diabetes mellitus (yes/no)"] = binary_map[
            st.selectbox("Diabetes Mellitus", ["no", "yes"])
        ]

    with col6:
        input_data["Coronary artery disease (yes/no)"] = binary_map[
            st.selectbox("Coronary Artery Disease", ["no", "yes"])
        ]
        input_data["Appetite (good/poor)"] = binary_map[
            st.selectbox("Appetite", ["good", "poor"])
        ]
        input_data["Pedal edema (yes/no)"] = binary_map[
            st.selectbox("Pedal Edema", ["no", "yes"])
        ]

    st.subheader("Lifestyle & History")

    col7, col8, col9 = st.columns(3)

    with col7:
        input_data["Anemia (yes/no)"] = binary_map[
            st.selectbox("Anemia", ["no", "yes"])
        ]
        input_data["Family history of chronic kidney disease"] = binary_map[
            st.selectbox("Family History of CKD", ["no", "yes"])
        ]
        input_data["Smoking status"] = binary_map[
            st.selectbox("Smoking Status", ["no", "yes"])
        ]

    with col8:
        input_data["Physical activity level"] = activity_map[
            st.selectbox("Physical Activity", ["low", "moderate", "high"])
        ]
        input_data["Urinary sediment microscopy results"] = binary_map[
            st.selectbox("Urinary Sediment", ["normal", "abnormal"])
        ]

    with col9:
        input_data["Body Mass Index (BMI)"] = st.number_input("BMI")
        input_data["Urine output (ml/day)"] = st.number_input("Urine Output")
        input_data["Cholesterol level"] = st.number_input("Cholesterol")

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])

    # Ensure exact feature set & order
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric features (as trained)
    df_scaled = scaler.transform(df.to_numpy())

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df_scaled)[0]
        label = target_encoder.inverse_transform([pred])[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Outcome: **{label}**")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0]
        prob_df = pd.DataFrame({
            "Class": target_encoder.classes_,
            "Probability (%)": np.round(prob * 100, 2)
        })

        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x="Class:N",
                y="Probability (%):Q",
                tooltip=["Class", "Probability (%)"]
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

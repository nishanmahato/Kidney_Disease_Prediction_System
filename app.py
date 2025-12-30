import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path


# PAGE CONFIG
st.set_page_config(
    page_title="Kidney Disease Prediction System",
    layout="wide"
)

st.title("🩺 Kidney Disease Prediction System")
st.caption(
    "A machine-learning–based tool for early assessment of chronic kidney disease risk using clinical and lifestyle data."
)


# BASE DIRECTORY
BASE_DIR = Path(__file__).parent.resolve()


# LOAD SAVED MODELS & OBJECTS
model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
target_encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")


# CATEGORICAL MAPPINGS
binary_map = {"yes": 1, "no": 0}
normal_abnormal_map = {"normal": 0, "abnormal": 1}
present_map = {"not present": 0, "present": 1}
appetite_map = {"good": 1, "poor": 0}
activity_map = {"low": 0, "moderate": 1, "high": 2}


# COLUMN TYPES
CATEGORICAL_COLS = [
    "Red blood cells in urine",
    "Pus cells in urine",
    "Pus cell clumps in urine",
    "Bacteria in urine",
    "Hypertension (yes/no)",
    "Diabetes mellitus (yes/no)",
    "Coronary artery disease (yes/no)",
    "Appetite (good/poor)",
    "Pedal edema (yes/no)",
    "Anemia (yes/no)",
    "Family history of chronic kidney disease",
    "Smoking status",
    "Physical activity level",
    "Urinary sediment microscopy results",
]

NUMERIC_COLS = [c for c in feature_columns if c not in CATEGORICAL_COLS]


# INPUT FORM
st.subheader("🧾 Patient Clinical Information")

input_data = {}

with st.form("patient_form"):
    cols = st.columns(3)

    for i, col in enumerate(feature_columns):
        with cols[i % 3]:

            if col in [
                "Red blood cells in urine",
                "Pus cells in urine",
                "Urinary sediment microscopy results",
            ]:
                input_data[col] = st.selectbox(col, ["normal", "abnormal"])

            elif col in [
                "Pus cell clumps in urine",
                "Bacteria in urine",
                "Hypertension (yes/no)",
                "Diabetes mellitus (yes/no)",
                "Coronary artery disease (yes/no)",
                "Pedal edema (yes/no)",
                "Anemia (yes/no)",
                "Family history of chronic kidney disease",
                "Smoking status",
            ]:
                input_data[col] = st.selectbox(col, ["yes", "no"])

            elif col == "Appetite (good/poor)":
                input_data[col] = st.selectbox(col, ["good", "poor"])

            elif col == "Physical activity level":
                input_data[col] = st.selectbox(col, ["low", "moderate", "high"])

            else:
                input_data[col] = st.number_input(col, value=0.0)

    submit = st.form_submit_button("🔍 Predict Risk")


# PREDICTION & RESULTS
if submit:
    df = pd.DataFrame([input_data]).reindex(columns=feature_columns)

    for col in df.columns:

        if col in [
            "Red blood cells in urine",
            "Pus cells in urine",
            "Urinary sediment microscopy results",
        ]:
            df[col] = df[col].map(normal_abnormal_map)

        elif col in ["Pus cell clumps in urine", "Bacteria in urine"]:
            df[col] = df[col].map(present_map)

        elif col in [
            "Hypertension (yes/no)",
            "Diabetes mellitus (yes/no)",
            "Coronary artery disease (yes/no)",
            "Pedal edema (yes/no)",
            "Anemia (yes/no)",
            "Family history of chronic kidney disease",
            "Smoking status",
        ]:
            df[col] = df[col].map(binary_map)

        elif col == "Appetite (good/poor)":
            df[col] = df[col].map(appetite_map)

        elif col == "Physical activity level":
            df[col] = df[col].map(activity_map)

    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = scaler.transform(df_scaled[NUMERIC_COLS])

    X_final = df_scaled.to_numpy()

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(X_final)[0]
        label = target_encoder.inverse_transform([pred])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_final)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0

    risk_class = (
        "High Risk"
        if top_category.lower() in ["ckd", "yes", "positive"]
        else "Low Risk"
    )

    st.subheader("Prediction Outcome")

    if risk_class == "High Risk":
        st.error("High Risk of Chronic Kidney Disease detected")
    else:
        st.success("Low Risk of Chronic Kidney Disease detected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clinical Assessment", top_category)
    c2.metric("Risk Probability", f"{top_prob}%")
    c3.metric(
        "Model Confidence",
        "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low",
    )

    # VISUALIZATION
    prob_df = pd.DataFrame({
        "Risk Category": target_encoder.classes_,
        "Probability (%)": np.round(probs, 2),
    }).sort_values("Probability (%)", ascending=False)

    st.subheader("Risk Probability Distribution")

    bar_chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x="Risk Category:N",
            y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Risk Category:N", scale=alt.Scale(scheme="set2")),
            tooltip=["Risk Category", "Probability (%)"],
        )
        .properties(height=350)
    )

    pie_chart = (
        alt.Chart(prob_df)
        .mark_arc(innerRadius=40, stroke="white")
        .encode(
            theta="Probability (%):Q",
            color=alt.Color("Risk Category:N", scale=alt.Scale(scheme="set2")),
            tooltip=["Risk Category", "Probability (%)"],
        )
        .properties(height=350)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(bar_chart, use_container_width=True)
    with col2:
        st.altair_chart(pie_chart, use_container_width=True)


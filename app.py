import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction System",
    layout="wide"
)

st.title("🩺 Kidney Disease Prediction System")
st.caption(
    "A machine-learning–based tool for early assessment of chronic kidney disease risk using clinical and lifestyle data."
)

# --------------------------------------------------
# LOAD SAVED MODELS & OBJECTS
# --------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --------------------------------------------------
# CATEGORICAL MAPPINGS (MATCH TRAINING)
# --------------------------------------------------
binary_map = {"yes": 1, "no": 0}
normal_abnormal_map = {"normal": 0, "abnormal": 1}
present_map = {"not present": 0, "present": 1}
appetite_map = {"good": 1, "poor": 0}
activity_map = {"low": 0, "moderate": 1, "high": 2}

# --------------------------------------------------
# CATEGORICAL COLUMN LIST
# --------------------------------------------------
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

NUMERIC_COLS = [col for col in feature_columns if col not in CATEGORICAL_COLS]

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
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

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_columns)

    # Encode categorical values
    for col in df.columns:
        if col in ["Red blood cells in urine", "Pus cells in urine", "Urinary sediment microscopy results"]:
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

    # Scale ONLY numeric columns
    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = scaler.transform(df_scaled[NUMERIC_COLS])

    X_final = df_scaled.to_numpy()

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(X_final)[0]
        label = target_encoder.inverse_transform([pred])[0]

    # --------------------------------------------------
    # RESULT DASHBOARD
    # --------------------------------------------------
    st.subheader("📊 Prediction Outcome")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_final)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0

    risk_class = "High Risk" if top_category.lower() in ["ckd", "yes", "positive"] else "Low Risk"

    c1, c2, c3 = st.columns(3)

    c1.metric("Clinical Assessment", top_category)
    c2.metric("Risk Probability", f"{top_prob}%")

    conf_label = "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low"
    c3.metric("Model Confidence", conf_label)

    # --------------------------------------------------
    # PROBABILITY TABLE & CHARTS
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        prob_df = pd.DataFrame({
            "Risk Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2),
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📈 Risk Probability Distribution")

        st.dataframe(prob_df, use_container_width=True)

        bar_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Risk Category:N", title="Risk Category"),
                y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Risk Category", "Probability (%)"],
            )
            .properties(height=350)
        )

        st.altair_chart(bar_chart, use_container_width=True)

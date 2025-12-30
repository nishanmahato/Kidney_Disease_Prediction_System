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
# CATEGORICAL COLUMN GROUPS (MATCH DATASET)
# --------------------------------------------------
YES_NO_COLS = [
    'Hypertension (yes/no)',
    'Diabetes mellitus (yes/no)',
    'Coronary artery disease (yes/no)',
    'Pedal edema (yes/no)',
    'Anemia (yes/no)',
    'Family history of chronic kidney disease',
    'Bacteria in urine',
    'Pus cell clumps in urine'
]

GOOD_POOR_COL = 'Appetite (good/poor)'
SMOKING_COL = 'Smoking status'
ACTIVITY_COL = 'Physical activity level'
SEDIMENT_COL = 'Urinary sediment microscopy results'

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction 🩺")
st.markdown(
    "A machine-learning–based system for early detection of chronic kidney disease "
    "using clinical, biochemical, and lifestyle indicators."
)

# --------------------------------------------------
# INPUT FORM (AUTO-GENERATED)
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Information")

    cols = st.columns(3)
    col_idx = 0

    for feature in feature_columns:

        if feature == "Target":
            continue

        with cols[col_idx]:

            # YES / NO
            if feature in YES_NO_COLS:
                input_data[feature] = (
                    1 if st.selectbox(feature, ["no", "yes"]) == "yes" else 0
                )

            # GOOD / POOR
            elif feature == GOOD_POOR_COL:
                input_data[feature] = (
                    1 if st.selectbox(feature, ["good", "poor"]) == "poor" else 0
                )

            # SMOKING STATUS
            elif feature == SMOKING_COL:
                smoking_map = {"never": 0, "former": 1, "current": 2}
                input_data[feature] = smoking_map[
                    st.selectbox(feature, smoking_map.keys())
                ]

            # PHYSICAL ACTIVITY
            elif feature == ACTIVITY_COL:
                activity_map = {"low": 0, "moderate": 1, "high": 2}
                input_data[feature] = activity_map[
                    st.selectbox(feature, activity_map.keys())
                ]

            # URINARY SEDIMENT
            elif feature == SEDIMENT_COL:
                input_data[feature] = (
                    1 if st.selectbox(feature, ["normal", "abnormal"]) == "abnormal" else 0
                )

            # NUMERIC FEATURES
            else:
                input_data[feature] = st.number_input(feature, value=0.0)

        col_idx = (col_idx + 1) % 3

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:

    df = pd.DataFrame([input_data])
    df = df[feature_columns]  # exact order

    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_columns
    )

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df_scaled)[0]
        label = target_encoder.inverse_transform([pred])[0]

    # --------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------
    st.subheader("Prediction Outcome")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_scaled)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0.0

    risk_class = "risk-high" if top_category.lower() in ["ckd", "yes", "positive"] else "risk-low"

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Clinical Assessment</div>"
        f"<div class='card-value {risk_class}'>{top_category}</div></div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Risk Probability</div>"
        f"<div class='card-value'>{top_prob}%</div></div>",
        unsafe_allow_html=True
    )

    confidence = "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low"

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>{confidence}</div></div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):

        prob_df = pd.DataFrame({
            "Risk Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2)
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        st.subheader("📈 Risk Probability Distribution")

        pie = (
            alt.Chart(prob_df)
            .mark_arc()
            .encode(
                theta="Probability (%):Q",
                color="Risk Category:N",
                tooltip=["Risk Category", "Probability (%)"]
            )
            .properties(width=400, height=400)
        )

        st.altair_chart(pie, use_container_width=True)

        bar = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Risk Category:N", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                color="Risk Category:N",
                tooltip=["Risk Category", "Probability (%)"]
            )
            .properties(width=700, height=350)
        )

        st.altair_chart(bar, use_container_width=True)

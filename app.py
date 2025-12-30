import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Kidney Disease Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# GLOBAL STYLE
# ==================================================
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

# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    target_encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, target_encoder, scaler, feature_columns

model, target_encoder, scaler, feature_columns = load_artifacts()

# ==================================================
# ENCODING FUNCTIONS (DATASET-ALIGNED)
# ==================================================
def yes_no(val): return 1 if val == "yes" else 0
def normal_abnormal(val): return 1 if val == "abnormal" else 0
def present_absent(val): return 1 if val == "present" else 0
def good_poor(val): return 1 if val == "poor" else 0
def physical_activity(val):
    return {"low": 0, "moderate": 1, "high": 2}[val]

# ==================================================
# APP HEADER
# ==================================================
st.title("Kidney Disease Risk Prediction 🩺")
st.markdown(
    "A machine-learning–based clinical decision support system for early "
    "assessment of chronic kidney disease using laboratory, clinical, and "
    "lifestyle parameters."
)

# ==================================================
# INPUT FORM (AUTO-GENERATED FROM MODEL FEATURES)
# ==================================================
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Clinical & Laboratory Data")

    cols = st.columns(3)
    col_idx = 0

    for feature in feature_columns:

        if feature == "Target":
            continue

        with cols[col_idx]:

            if feature in [
                'Red blood cells in urine',
                'Pus cells in urine',
                'Urinary sediment microscopy results'
            ]:
                input_data[feature] = normal_abnormal(
                    st.selectbox(feature, ["normal", "abnormal"])
                )

            elif feature in [
                'Pus cell clumps in urine',
                'Bacteria in urine'
            ]:
                input_data[feature] = present_absent(
                    st.selectbox(feature, ["not present", "present"])
                )

            elif feature in [
                'Hypertension (yes/no)',
                'Diabetes mellitus (yes/no)',
                'Coronary artery disease (yes/no)',
                'Pedal edema (yes/no)',
                'Anemia (yes/no)',
                'Family history of chronic kidney disease',
                'Smoking status'
            ]:
                input_data[feature] = yes_no(
                    st.selectbox(feature, ["no", "yes"])
                )

            elif feature == 'Appetite (good/poor)':
                input_data[feature] = good_poor(
                    st.selectbox(feature, ["good", "poor"])
                )

            elif feature == 'Physical activity level':
                input_data[feature] = physical_activity(
                    st.selectbox(feature, ["low", "moderate", "high"])
                )

            else:
                input_data[feature] = st.number_input(
                    feature, value=0.0
                )

        col_idx = (col_idx + 1) % 3

    submit = st.form_submit_button("Predict Risk")

# ==================================================
# PREDICTION PIPELINE
# ==================================================
if submit:

    df = pd.DataFrame([input_data])
    df = df[feature_columns]

    df_scaled = scaler.transform(df)

    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df_scaled)[0]
        label = target_encoder.inverse_transform([pred])[0]

    # ==================================================
    # DASHBOARD
    # ==================================================
    st.subheader("Prediction Outcome")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_scaled)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0

    risk_class = (
        "risk-high"
        if str(top_category).lower() in ["ckd", "yes", "positive"]
        else "risk-low"
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'><div class='card-title'>Clinical Assessment</div>"
        f"<div class='card-value {risk_class}'>{top_category}</div></div>",
        unsafe_allow_html=True,
    )

    c2.markdown(
        f"<div class='card'><div class='card-title'>Risk Probability</div>"
        f"<div class='card-value'>{top_prob}%</div></div>",
        unsafe_allow_html=True,
    )

    conf_label = "High" if top_prob >= 80 else "Moderate" if top_prob >= 60 else "Low"

    c3.markdown(
        f"<div class='card'><div class='card-title'>Model Confidence</div>"
        f"<div class='card-value'>{conf_label}</div></div>",
        unsafe_allow_html=True,
    )

    # ==================================================
    # PROBABILITY VISUALIZATION
    # ==================================================
    if hasattr(model, "predict_proba"):

        prob_df = pd.DataFrame({
            "Risk Category": target_encoder.classes_,
            "Probability (%)": np.round(probs, 2),
        }).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        st.subheader("📈 Risk Probability Distribution")

        pie_chart = (
            alt.Chart(prob_df)
            .mark_arc()
            .encode(
                theta="Probability (%):Q",
                color="Risk Category:N",
                tooltip=["Risk Category", "Probability (%)"]
            )
            .properties(width=400, height=400)
        )

        bar_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Risk Category:N", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Risk Category:N", legend=None),
                tooltip=["Risk Category", "Probability (%)"]
            )
            .properties(width=700, height=350)
        )

        st.altair_chart(pie_chart, use_container_width=True)
        st.altair_chart(bar_chart, use_container_width=True)

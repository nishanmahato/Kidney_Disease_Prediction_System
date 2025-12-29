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
        color: #111827;
    }
    .risk-high { color: #b91c1c; }
    .risk-low { color: #047857; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
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
def yes_no(val):
    return 1 if val == "Yes" else 0

def good_poor(val):
    return 1 if val == "Poor" else 0

def encode_smoking(val):
    return {"Never": 0, "Former": 1, "Current": 2}[val]

def encode_activity(val):
    return {"Low": 0, "Moderate": 1, "High": 2}[val]

def encode_sediment(val):
    return {"Normal": 0, "Abnormal": 1}[val]

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("Kidney Disease Risk Prediction 🩺")
st.markdown(
    "A machine-learning–based tool for early assessment of chronic kidney disease risk using clinical and lifestyle data. "
    "Provide accurate patient data to get a reliable prediction."
)


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
input_data = {}

with st.form("patient_form"):
    st.subheader("Patient Clinical Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["Age"] = st.number_input("Age (years)", 0, 120)
        input_data["Blood pressure"] = st.number_input("Blood Pressure (mm/Hg)")
        input_data["Specific gravity"] = st.number_input("Specific Gravity")
        input_data["Albumin"] = st.number_input("Albumin")

    with col2:
        input_data["Sugar"] = st.number_input("Sugar")
        input_data["Blood glucose random"] = st.number_input("Random Blood Glucose")
        input_data["Blood urea"] = st.number_input("Blood Urea")
        input_data["Serum creatinine"] = st.number_input("Serum Creatinine")

    with col3:
        input_data["Sodium"] = st.number_input("Sodium")
        input_data["Potassium"] = st.number_input("Potassium")
        input_data["Hemoglobin"] = st.number_input("Hemoglobin")
        input_data["Packed cell volume"] = st.number_input("Packed Cell Volume")

    st.subheader("Medical History & Lifestyle")
    col4, col5, col6 = st.columns(3)

    with col4:
        input_data["Hypertension"] = yes_no(st.selectbox("Hypertension", ["No", "Yes"]))
        input_data["Diabetes mellitus"] = yes_no(st.selectbox("Diabetes", ["No", "Yes"]))
        input_data["Coronary artery disease"] = yes_no(st.selectbox("CAD", ["No", "Yes"]))

    with col5:
        input_data["Pedal edema"] = yes_no(st.selectbox("Pedal Edema", ["No", "Yes"]))
        input_data["Anemia"] = yes_no(st.selectbox("Anemia", ["No", "Yes"]))
        input_data["Family history of CKD"] = yes_no(st.selectbox("Family History of CKD", ["No", "Yes"]))

    with col6:
        input_data["Appetite"] = good_poor(st.selectbox("Appetite", ["Good", "Poor"]))
        input_data["Smoking status"] = encode_smoking(
            st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        )
        input_data["Physical activity"] = encode_activity(
            st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        )

    input_data["Urinary sediment"] = encode_sediment(
        st.selectbox("Urinary Sediment", ["Normal", "Abnormal"])
    )

    submit = st.form_submit_button("Predict Risk")

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
if submit:
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale features if scaler is available
    try:
        df = pd.DataFrame(
            scaler.transform(df),
            columns=feature_columns
        )
    except Exception:
        pass

    # ---------------- PREDICTION ----------------
    with st.spinner("Analyzing patient data..."):
        pred = model.predict(df)[0]
        label = target_encoder.inverse_transform([pred])[0]

    # ---------------- DASHBOARD ----------------
    st.subheader("Prediction Outcome")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0] * 100
        top_idx = np.argmax(probs)
        top_category = target_encoder.classes_[top_idx]
        top_prob = round(probs[top_idx], 2)
    else:
        top_category = label
        top_prob = 0

    risk_class = (
        "risk-high"
        if top_category.lower() in ["ckd", "yes", "positive"]
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

    # --------------------------------------------------
    # PREDICTION PROBABILITIES + VISUALIZATION
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):

        prob_df = pd.DataFrame(
            {
                "Risk Category": target_encoder.classes_,
                "Probability (%)": np.round(probs, 2),
            }
        ).sort_values("Probability (%)", ascending=False)

        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        st.subheader("📈 Risk Probability Distribution")

        # -------- PIE CHART (NORMAL PIE) --------
        pie_chart = (
            alt.Chart(prob_df)
            .mark_arc()
            .encode(
                theta=alt.Theta("Probability (%):Q"),
                color=alt.Color("Risk Category:N"),
                tooltip=["Risk Category", "Probability (%)"],
            )
            .properties(width=400, height=400)
        )

        st.altair_chart(pie_chart, use_container_width=True)

        # -------- BAR CHART (VERTICAL, BELOW PIE) --------
        bar_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Risk Category:N",
                    title="Risk Category",
                    axis=alt.Axis(labelAngle=-30),
                ),
                y=alt.Y(
                    "Probability (%):Q",
                    title="Probability (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color("Risk Category:N", legend=None),
                tooltip=["Risk Category", "Probability (%)"],
            )
            .properties(width=700, height=350)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    else:
        st.info("This model does not support probability predictions.")





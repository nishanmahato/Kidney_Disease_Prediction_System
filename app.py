import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL & PREPROCESSING OBJECTS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "rf_kidney_disease_model.pkl")
    encoder = joblib.load(BASE_DIR / "target_label_encoder.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, encoder, scaler, feature_columns

# Feature columns used during training (MUST match exactly)
feature_columns = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
st.title("🩺 Chronic Kidney Disease Prediction System")

st.markdown(
    """
    **This clinical decision-support system applies machine learning techniques
    to assess the probability of chronic kidney disease (CKD) based on patient
    medical parameters.**  
    The system is designed to assist healthcare professionals by providing
    early risk estimation, improving diagnosis efficiency, and supporting
    preventive care strategies.
    """
)

# --------------------------------------------------
# KIDNEY IMAGE
# --------------------------------------------------
try:
    kidney_img = Image.open("kidney.png")  # place kidney.png in same folder
    st.image(kidney_img, caption="Human Kidney Anatomy", use_container_width=True)
except:
    st.info("ℹ️ Kidney image not found. Add 'kidney.png' to enhance UI.")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("🧪 Patient Medical Parameters")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Age", 1, 120, 45)
    bp = col2.number_input("Blood Pressure (mmHg)", 50, 200, 80)
    sg = col1.number_input("Specific Gravity", 1.005, 1.030, 1.020)
    al = col2.number_input("Albumin", 0, 5, 1)
    su = col1.number_input("Sugar", 0, 5, 0)
    bgr = col2.number_input("Blood Glucose Random", 50, 500, 120)
    bu = col1.number_input("Blood Urea", 1.0, 300.0, 40.0)
    sc = col2.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
    sod = col1.number_input("Sodium", 100.0, 170.0, 135.0)
    pot = col2.number_input("Potassium", 2.0, 7.0, 4.5)
    hemo = col1.number_input("Hemoglobin", 3.0, 20.0, 13.0)
    pcv = col2.number_input("Packed Cell Volume", 20, 60, 40)
    wc = col1.number_input("White Blood Cell Count", 3000, 20000, 8000)
    rc = col2.number_input("Red Blood Cell Count", 2.0, 6.5, 4.5)

    submitted = st.form_submit_button("🔍 Predict CKD Risk")

# --------------------------------------------------
# PREDICTION LOGIC (ERROR FIX APPLIED HERE)
# --------------------------------------------------
if submitted:
    input_data = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc
    }

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Ensure column order
    df = df.reindex(columns=feature_columns)

    # 🔧 FIX: Convert DataFrame to NumPy array before scaling
    df_scaled = scaler.transform(df.values)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)

    # Predict probabilities
    probs = best_model.predict_proba(df_scaled)[0]
    classes = target_encoder.inverse_transform(np.arange(len(probs)))

    prob_df = pd.DataFrame({
        "Risk Category": classes,
        "Probability (%)": np.round(probs * 100, 2)
    })

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.subheader("📊 Prediction Results")

    predicted_class = classes[np.argmax(probs)]
    st.success(f"🩸 **Predicted Outcome:** {predicted_class}")

    # --------------------------------------------------
    # VISUALIZATION (VERTICAL)
    # --------------------------------------------------
    st.subheader("📈 Risk Probability Distribution")

    # Normal Pie Chart
    pie_chart = (
        alt.Chart(prob_df)
        .mark_arc()
        .encode(
            theta="Probability (%):Q",
            color="Risk Category:N",
            tooltip=["Risk Category", "Probability (%)"]
        )
        .properties(width=350, height=350)
    )

    st.altair_chart(pie_chart, use_container_width=False)

    # Bar Chart (Below)
    bar_chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("Risk Category:N", title="Risk Category"),
            y=alt.Y(
                "Probability (%):Q",
                title="Probability (%)",
                scale=alt.Scale(domain=[0, 100])
            ),
            color=alt.Color("Risk Category:N", legend=None),
            tooltip=["Risk Category", "Probability (%)"]
        )
        .properties(width=700, height=350)
    )

    st.altair_chart(bar_chart, use_container_width=False)

    # --------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------
    st.warning(
        "⚠️ **Disclaimer:** This system is intended for educational and decision-support purposes only. "
        "It does not replace professional medical diagnosis or clinical judgment."
    )


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ===============================================================
# 1. LOAD TRAINED MODEL
# ===============================================================

MODEL_PATH = "kidney_logreg_randomsearch.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success(f"Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    st.error(f"❌ Could not load saved model: {e}")
    st.stop()


# ===============================================================
# 2. LOAD ORIGINAL DATASET TO REBUILD PREPROCESSORS
# ===============================================================

DATA_URL = "https://raw.githubusercontent.com/nishanmahato/Kidney_Disease_Prediction_System/refs/heads/main/kidney_disease_dataset.csv"

df = pd.read_csv(DATA_URL)

# Identify actual target column in your dataset
target_col = "Target"   # Update only if different

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset!")
    st.stop()

# Split features & labels
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric feature columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Store training feature order
training_features = X.columns.tolist()


# ===============================================================
# 3. REBUILD LABEL ENCODERS (for categorical features)
# ===============================================================

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le


# ===============================================================
# 4. REBUILD SCALER (for numeric features)
# ===============================================================

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


# ===============================================================
# 5. REBUILD TARGET LABEL ENCODER (for reverse decoding)
# ===============================================================

target_encoder = LabelEncoder()
target_encoder.fit(df[target_col].astype(str))


# ===============================================================
# 6. STREAMLIT INPUT UI
# ===============================================================

st.title("🩺 Kidney Disease Prediction System")
st.write("AI-powered prediction model trained using Logistic Regression with RandomizedSearchCV.")

st.subheader("Enter Patient Information:")

user_data = {}

for col in training_features:

    if col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        median = float(df[col].median())

        user_data[col] = st.number_input(
            f"{col} (numeric)",
            value=median,
            min_value=min_val,
            max_value=max_val
        )

    elif col in categorical_cols:
        options = sorted(df[col].dropna().unique().tolist())
        user_data[col] = st.selectbox(f"{col} (categorical)", options)


# Convert user input → DataFrame
input_df = pd.DataFrame([user_data])


# ===============================================================
# 7. APPLY LABEL ENCODING ON USER INPUT (same as training)
# ===============================================================

for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col].astype(str))


# ===============================================================
# 8. APPLY SCALING ON USER INPUT (same as training)
# ===============================================================

input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])


# ===============================================================
# 9. FIX FEATURE ORDER — must match training exactly
# ===============================================================

input_df = input_df[training_features]


# ===============================================================
# 10. MAKE PREDICTION WITH REVERSE DECODING
# ===============================================================

if st.button("🔍 Predict"):
    try:
        pred_encoded = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted Class: **{pred_label}**")

        # Show class probabilities (decoded)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            st.subheader("Prediction Probabilities:")

            decoded_probs = {
                target_encoder.inverse_transform([cls])[0]: float(prob)
                for cls, prob in zip(model.classes_, probs)
            }

            st.json(decoded_probs)

    except Exception as e:
        st.error(f"Prediction error: {e}")

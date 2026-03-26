import streamlit as st
import pandas as pd
import joblib

# Load models
model_proc = joblib.load("processing_model.pkl")
model_fraud = joblib.load("fraud_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Returns Analytics", layout="centered")

st.title("📦 E-Commerce Returns Analytics")
st.subheader("Operations + Fraud Prediction Dashboard")

st.write("Predict processing category and fraud risk based on return details.")

# Inputs
category = st.selectbox(
    "Product Category",
    encoders["product_category_name"].classes_
)

reason = st.selectbox(
    "Return Reason",
    encoders["return_reason"].classes_
)

load = st.selectbox(
    "Warehouse Load",
    encoders["warehouse_load"].classes_
)
inspection = st.selectbox(
    "Inspection Level",
    encoders["inspection_level"].classes_
)

if st.button("Predict"):

    # ---------------- PROCESSING INPUT (3 features) ----------------
    proc_input = pd.DataFrame([[
        encoders["product_category_name"].transform([category])[0],
        encoders["return_reason"].transform([reason])[0],
        encoders["warehouse_load"].transform([load])[0]
    ]], columns=[
        "product_category_name",
        "return_reason",
        "warehouse_load"
    ])

    # ---------------- FRAUD INPUT (4 features) ----------------
    fraud_input = pd.DataFrame([[
        encoders["product_category_name"].transform([category])[0],
        encoders["return_reason"].transform([reason])[0],
        encoders["inspection_level"].transform([inspection])[0],
        encoders["warehouse_load"].transform([load])[0]
    ]], columns=[
        "product_category_name",
        "return_reason",
        "inspection_level",
        "warehouse_load"
    ])

    # Predictions
    proc_pred = model_proc.predict(proc_input)[0]
    proc_label = target_encoder.inverse_transform([proc_pred])[0]

    fraud_pred = model_fraud.predict(fraud_input)[0]

    st.markdown("### 🔍 Results")

    st.success(f"📊 Processing Category: **{proc_label}**")

    if fraud_pred == 1:
        st.error("⚠️ High Fraud Risk")
    else:
        st.success("✅ Likely Genuine Return")

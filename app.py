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

if st.button("Predict"):

    input_data = [
        encoders["product_category_name"].transform([category])[0],
        encoders["return_reason"].transform([reason])[0],
        encoders["warehouse_load"].transform([load])[0]
    ]

    input_df = pd.DataFrame([input_data], columns=[
        "product_category_name",
        "return_reason",
        "warehouse_load"
    ])

    proc_pred = model_proc.predict(input_df)[0]
    proc_label = target_encoder.inverse_transform([proc_pred])[0]

    fraud_pred = model_fraud.predict(input_df)[0]

    st.markdown("### 🔍 Results")

    st.success(f"📊 Processing Category: **{proc_label}**")

    if fraud_pred == 1:
        st.error("⚠️ High Fraud Risk")
    else:
        st.success("✅ Likely Genuine Return")

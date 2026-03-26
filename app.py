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

# ---------------- HUMAN READABLE MAPPINGS ----------------

category_map = {
    0: "Electronics",
    1: "Groceries",
    2: "Clothing"
}

reason_map = {
    0: "Damaged",
    1: "Wrong Item",
    2: "Not Needed"
}

load_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

inspection_map = {
    0: "Basic",
    1: "Manual",
    2: "Intensive"
}

processing_map = {
    0: "Low Processing",
    1: "Medium Processing",
    2: "High Processing"
}

# ---------------- INPUTS ----------------

category = st.selectbox("Product Category", list(category_map.values()))
reason = st.selectbox("Return Reason", list(reason_map.values()))
load = st.selectbox("Warehouse Load", list(load_map.values()))
inspection = st.selectbox("Inspection Level", list(inspection_map.values()))

# Convert back to encoded values
def get_key(val, dictionary):
    return list(dictionary.keys())[list(dictionary.values()).index(val)]

category_encoded = get_key(category, category_map)
reason_encoded = get_key(reason, reason_map)
load_encoded = get_key(load, load_map)
inspection_encoded = get_key(inspection, inspection_map)

# ---------------- PREDICTION ----------------

if st.button("Predict"):

    # Processing model input (3 features)
    proc_input = pd.DataFrame([[
        category_encoded,
        reason_encoded,
        load_encoded
    ]], columns=[
        "product_category_name",
        "return_reason",
        "warehouse_load"
    ])

    # Fraud model input (4 features)
    fraud_input = pd.DataFrame([[
        category_encoded,
        reason_encoded,
        inspection_encoded,
        load_encoded
    ]], columns=[
        "product_category_name",
        "return_reason",
        "inspection_level",
        "warehouse_load"
    ])

    # Predictions
    proc_pred = model_proc.predict(proc_input)[0]
    proc_label = target_encoder.inverse_transform([proc_pred])[0]

    fraud_prob = model_fraud.predict_proba(fraud_input)[0][1]

    # ---------------- OUTPUT ----------------

    st.markdown("### 🔍 Results")

    # Processing output
    st.success(f"📊 Processing Category: **{processing_map[proc_pred]}**")

    # Fraud output (IMPROVED LOGIC)
    st.metric("Fraud Risk Score", f"{fraud_prob:.6f}")

    if fraud_prob > 0.05:
        st.error(f"⚠️ High Fraud Risk")
    elif fraud_prob > 0.01:
        st.success(f"Medium Fraud Risk")
    else: 
        st.success(f"Likely Genuine return")

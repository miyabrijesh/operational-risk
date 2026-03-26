import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Returns Analytics", layout="wide")

# ---------------- LOAD HTML DASHBOARD ----------------
with open("ecommerce_returns_dashboard.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# ---------------- LOAD MODELS ----------------
model_proc = joblib.load("processing_model.pkl")
model_fraud = joblib.load("fraud_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Prediction System", "📊 Advanced Dashboard"])

# =====================================================
# 🔮 TAB 1: ML PREDICTION
# =====================================================
with tab1:

    st.title("📦 E-Commerce Returns Analytics")
    st.subheader("Operations + Fraud Prediction Dashboard")

    st.write("Predict processing category and fraud risk based on return details.")

    # ---------------- INPUTS (USING ORIGINAL ENCODERS ✅) ----------------
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

    # Encode inputs (MATCHES TRAINING ✅)
    category_encoded = encoders["product_category_name"].transform([category])[0]
    reason_encoded = encoders["return_reason"].transform([reason])[0]
    load_encoded = encoders["warehouse_load"].transform([load])[0]
    inspection_encoded = encoders["inspection_level"].transform([inspection])[0]

    # Optional: Processing label mapping (if needed)
    processing_map = {
        0: "Low Processing",
        1: "Medium Processing",
        2: "High Processing"
    }

    # ---------------- PREDICTION ----------------
    if st.button("Predict"):

        # Processing input
        proc_input = pd.DataFrame([[
            category_encoded,
            reason_encoded,
            load_encoded
        ]], columns=[
            "product_category_name",
            "return_reason",
            "warehouse_load"
        ])

        # Fraud input
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
        fraud_prob = model_fraud.predict_proba(fraud_input)[0][1]

        # ---------------- OUTPUT ----------------
        st.markdown("### 🔍 Results")

        st.success(f"📊 Processing Category: **{processing_map.get(proc_pred, proc_pred)}**")

        st.metric("Fraud Risk Score", f"{fraud_prob:.6f}")

        if fraud_prob > 0.05:
            st.error("⚠️ High Fraud Risk")
        elif fraud_prob > 0.01:
            st.warning("⚠️ Medium Fraud Risk")
        else:
            st.success("✅ Likely Genuine Return")


# =====================================================
# 📊 TAB 2: DASHBOARD
# =====================================================
with tab2:

    st.markdown("## 📊 Advanced Dashboard")

    components.html(
        html_content,
        height=1200,
        scrolling=True
    )

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

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Prediction System", "📊 Advanced Dashboard"])

# =====================================================
# 🔮 TAB 1: ML PREDICTION
# =====================================================
with tab1:

    st.title("📦 E-Commerce Returns Analytics")
    st.subheader("Operations + Fraud Prediction Dashboard")

    st.write("Predict processing category and fraud risk based on return details.")

    # ---------------- CLEAN DISPLAY MAPPINGS ----------------
    category_map = {
        "Electronics": 0,
        "Groceries": 1,
        "Clothing": 2
    }

    reason_map = {
        "Damaged": 0,
        "Wrong Item": 1,
        "Not Needed": 2
    }

    load_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    inspection_map = {
        "Basic": 0,
        "Manual": 1,
        "Intensive": 2
    }

    # ✅ FIX 1: Processing labels
    processing_map = {
        0: "Low Processing",
        1: "Medium Processing",
        2: "High Processing"
    }

    # ---------------- INPUT UI ----------------
    category = st.selectbox("Product Category", list(category_map.keys()))
    reason = st.selectbox("Return Reason", list(reason_map.keys()))
    load = st.selectbox("Warehouse Load", list(load_map.keys()))
    inspection = st.selectbox("Inspection Level", list(inspection_map.keys()))

    # ---------------- ENCODING ----------------
    encoders = joblib.load("encoders.pkl")

    category_encoded = encoders["product_category_name"].transform([category])[0]
    reason_encoded = encoders["return_reason"].transform([reason])[0]
    load_encoded = encoders["warehouse_load"].transform([load])[0]
    inspection_encoded = encoders["inspection_level"].transform([inspection])[0]

    # ---------------- PREDICTION ----------------
    if st.button("Predict"):

        # Processing model input
        proc_input = pd.DataFrame([[
            category_encoded,
            reason_encoded,
            load_encoded
        ]], columns=[
            "product_category_name",
            "return_reason",
            "warehouse_load"
        ])

        # Fraud model input
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

        # ✅ FIX 1 APPLIED
        st.success(f"📊 Processing Category: **{processing_map[proc_pred]}**")

        # Show fraud score cleanly
        st.metric("Fraud Risk Score", f"{fraud_prob:.4f}")

        # ✅ FIX 2 APPLIED (better thresholds)
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

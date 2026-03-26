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

    # ---------------- HUMAN FRIENDLY MAPPINGS ----------------
    category_display = {
        "Electronics": 0,
        "Groceries": 1,
        "Clothing": 2
    }

    reason_display = {
        "Damaged": 0,
        "Wrong Item": 1,
        "Not Needed": 2
    }

    load_display = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    inspection_display = {
        "Basic": 0,
        "Manual": 1,
        "Intensive": 2
    }

    processing_map = {
        0: "Low Processing",
        1: "Medium Processing",
        2: "High Processing"
    }

    # ---------------- INPUTS ----------------
    category = st.selectbox("Product Category", list(category_display.keys()))
    reason = st.selectbox("Return Reason", list(reason_display.keys()))
    load = st.selectbox("Warehouse Load", list(load_display.keys()))
    inspection = st.selectbox("Inspection Level", list(inspection_display.keys()))

    # Encode values for model
    category_encoded = category_display[category]
    reason_encoded = reason_display[reason]
    load_encoded = load_display[load]
    inspection_encoded = inspection_display[inspection]

    # ---------------- PREDICTION ----------------
    if st.button("Predict"):

        proc_input = pd.DataFrame([[
            category_encoded,
            reason_encoded,
            load_encoded
        ]], columns=[
            "product_category_name",
            "return_reason",
            "warehouse_load"
        ])

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

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
encoders = joblib.load("encoders.pkl")  # ✅ FIXED POSITION

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Prediction System", "📊 Advanced Dashboard"])

# =====================================================
# 🔮 TAB 1: ML PREDICTION
# =====================================================
with tab1:

    st.title("📦 E-Commerce Returns Analytics")
    st.subheader("Operations + Fraud Prediction Dashboard")

    st.write("Predict processing category and fraud risk based on return details.")

    # ---------------- PROCESSING LABELS ----------------
    processing_map = {
        0: "Low Processing",
        1: "Medium Processing",
        2: "High Processing"
    }

    # ----------- CLEAN LABEL FUNCTION -----------
    def clean_label(x):
        return x.replace("_", " ").title()

    # ----------- DROPDOWNS USING ENCODERS -----------

    category_options = encoders["product_category_name"].classes_
    category_display = {clean_label(x): x for x in category_options}
    category = st.selectbox("Product Category", list(category_display.keys()))

    reason_options = encoders["return_reason"].classes_
    reason_display = {clean_label(x): x for x in reason_options}
    reason = st.selectbox("Return Reason", list(reason_display.keys()))

    load_options = encoders["warehouse_load"].classes_
    load_display = {clean_label(x): x for x in load_options}
    load = st.selectbox("Warehouse Load", list(load_display.keys()))

    inspection_options = encoders["inspection_level"].classes_
    inspection_display = {clean_label(x): x for x in inspection_options}
    inspection = st.selectbox("Inspection Level", list(inspection_display.keys()))

    # ----------- ENCODING -----------

    category_encoded = encoders["product_category_name"].transform([category_display[category]])[0]
    reason_encoded = encoders["return_reason"].transform([reason_display[reason]])[0]
    load_encoded = encoders["warehouse_load"].transform([load_display[load]])[0]
    inspection_encoded = encoders["inspection_level"].transform([inspection_display[inspection]])[0]

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

        st.success(f"📊 Processing Category: **{processing_map[proc_pred]}**")

        st.metric("Fraud Risk Score", f"{fraud_prob:.4f}")

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

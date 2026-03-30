import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Returns Fraud & Warehouse Analytics",
    page_icon="📦",
    layout="wide"
)

# ─── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_proc  = joblib.load("processing_model.pkl")
    model_fraud = joblib.load("fraud_model.pkl")
    encoders    = joblib.load("encoders.pkl")
    return model_proc, model_fraud, encoders

model_proc, model_fraud, encoders = load_models()

# ─── LOAD DASHBOARD HTML ───────────────────────────────────────
@st.cache_data
def load_html():
    with open("ecommerce_returns_dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

html_content = load_html()

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediction System", "📊 Analytics Dashboard"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — ML PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:

    st.title("📦 E-Commerce Returns: Fraud & Processing Predictor")
    st.write(
        "Enter the details of a return request below. "
        "The model will predict the **processing category** and estimate the **fraud risk**."
    )
    st.divider()

    col1, col2 = st.columns(2)

    # ── Dropdown options come directly from the LabelEncoders ──
    # This guarantees dropdowns always show human-readable labels
    # and encoding is always consistent with training.

    with col1:
        st.subheader("Return Details")

        product_category = st.selectbox(
            "Product Category",
            options=list(encoders["Product_Category"].classes_),
            help="Type of product being returned"
        )

        return_reason = st.selectbox(
            "Return Reason",
            options=list(encoders["Return_Reason"].classes_),
            help="Reason stated by the customer for the return"
        )

        inspection_level = st.selectbox(
            "Inspection Level",
            options=list(encoders["Inspection_Level"].classes_),
            help="Level of inspection assigned at the warehouse"
        )

        warehouse_load = st.selectbox(
            "Warehouse Load",
            options=list(encoders["Warehouse_Load"].classes_),
            help="Current operational load at the warehouse"
        )

    with col2:
        st.subheader("Prediction Results")

        if st.button("🔍 Predict", use_container_width=True, type="primary"):

            # ── Encode inputs using the same LabelEncoders from training ──
            cat_enc  = encoders["Product_Category"].transform([product_category])[0]
            reas_enc = encoders["Return_Reason"].transform([return_reason])[0]
            insp_enc = encoders["Inspection_Level"].transform([inspection_level])[0]
            load_enc = encoders["Warehouse_Load"].transform([warehouse_load])[0]

            # ── Processing Category Prediction ──
            proc_input = pd.DataFrame(
                [[cat_enc, reas_enc, load_enc]],
                columns=["Product_Category", "Return_Reason", "Warehouse_Load"]
            )
            proc_pred_encoded = model_proc.predict(proc_input)[0]
            proc_pred_label   = encoders["Processing_Category"].inverse_transform([proc_pred_encoded])[0]

            # ── Fraud Risk Prediction ──
            fraud_input = pd.DataFrame(
                [[cat_enc, reas_enc, insp_enc, load_enc]],
                columns=["Product_Category", "Return_Reason", "Inspection_Level", "Warehouse_Load"]
            )
            fraud_prob = model_fraud.predict_proba(fraud_input)[0][1]

            # ── Display Results ──────────────────────────────────────────

            # Processing category with colour coding
            proc_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            proc_avg   = {"Low": "< 20 min", "Medium": "20–50 min", "High": "> 50 min"}
            st.metric(
                label="📊 Processing Category",
                value=f"{proc_color.get(proc_pred_label, '')} {proc_pred_label}",
                delta=proc_avg.get(proc_pred_label, "")
            )

            st.metric(
                label="🎯 Fraud Probability Score",
                value=f"{fraud_prob:.1%}"
            )

            # Risk verdict
            if fraud_prob >= 0.55:
                st.error(
                    f"⚠️ **HIGH FRAUD RISK** ({fraud_prob:.1%})  \n"
                    "Recommend: Intensive inspection before accepting this return."
                )
            elif fraud_prob >= 0.30:
                st.warning(
                    f"⚡ **MEDIUM FRAUD RISK** ({fraud_prob:.1%})  \n"
                    "Recommend: Manual inspection — verify product condition and customer history."
                )
            else:
                st.success(
                    f"✅ **LOW FRAUD RISK** ({fraud_prob:.1%})  \n"
                    "Recommend: Basic inspection — likely a genuine return."
                )

            st.divider()

            # ── Summary table ─────────────────────────────────────────
            st.markdown("**Input Summary**")
            summary = pd.DataFrame({
                "Field":  ["Product Category", "Return Reason", "Inspection Level", "Warehouse Load"],
                "Value":  [product_category, return_reason, inspection_level, warehouse_load],
                "Encoded": [cat_enc, reas_enc, insp_enc, load_enc]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.info("👈 Fill in the return details on the left and click **Predict**.")

    st.divider()

    # ── Reference table ───────────────────────────────────────
    with st.expander("📖 Model Information & Thresholds"):
        st.markdown("""
        **Models used:**
        - **Processing Category**: Random Forest Classifier (200 trees, balanced class weights)
          - `Low` = typically < 20 min processing time
          - `Medium` = 20–50 min processing time  
          - `High` = > 50 min processing time

        - **Fraud Risk**: Random Forest Classifier (300 trees, fraud class weighted 3×)
          - Training data: 250 QuickCommerce records (Blinkit + Swiggy Instamart)
          - Fraud rate in training data: 34% (85 suspected fraud, 165 genuine)
          - Model accuracy: 91% | Fraud recall: 98%

        **Fraud Risk Thresholds:**
        | Score | Risk Level | Recommended Action |
        |-------|-----------|-------------------|
        | ≥ 55% | 🔴 High | Intensive inspection — hold return |
        | 30–55% | 🟡 Medium | Manual verification required |
        | < 30% | 🟢 Low | Basic inspection — process normally |

        **Key insight from data:** Fraudulent returns take on average **74.2 min** to process 
        vs **20.7 min** for genuine returns (t = −17.44, p < 0.001).
        """)

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Full Analytics Dashboard")
    st.markdown(
        "Interactive dashboard covering all three datasets: "
        "QuickCommerce (N=250), Olist/Kaggle (N=111,702), and Consumer Survey (N=80)."
    )
    components.html(html_content, height=1100, scrolling=True)

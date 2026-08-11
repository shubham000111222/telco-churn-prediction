"""
streamlit_demo/app.py
---------------------
Interactive dashboard for churn prediction.
Run: streamlit run streamlit_demo/app.py
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ── Load ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    xgb   = joblib.load("models/xgb_model.pkl")
    lgbm  = joblib.load("models/lgbm_model.pkl")
    fnames= joblib.load("models/feature_names.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return xgb, lgbm, fnames, metrics


xgb, lgbm, feature_names, metrics = load_models()
THRESHOLD = metrics.get("threshold", 0.30)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Telco Customer Churn Prediction")
st.caption(
    "XGBoost + LightGBM ensemble | IBM Telco Dataset (7,043 customers) | "
    f"ROC-AUC: **{metrics['roc_auc']}** | Threshold: **{THRESHOLD}**"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Score a Customer", "📈 Model Performance", "📊 Data Insights"])

# ── Tab 1: Predict ─────────────────────────────────────────────────────────

with tab1:
    st.subheader("Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges= st.number_input("Monthly Charges ($)", 18.8, 118.8, 70.0, step=0.5)
        contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        senior         = st.checkbox("Senior Citizen")

    with col3:
        tech_support   = st.checkbox("Tech Support")
        online_security= st.checkbox("Online Security")
        paperless      = st.checkbox("Paperless Billing", value=True)
        partner        = st.checkbox("Has Partner")

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        total_charges = monthly_charges * tenure + 50

        row = {
            "SeniorCitizen": int(senior), "Partner": int(partner),
            "Dependents": 0, "tenure": tenure,
            "PhoneService": 1, "MultipleLines": 0,
            "OnlineSecurity": int(online_security), "OnlineBackup": 0,
            "DeviceProtection": 0, "TechSupport": int(tech_support),
            "StreamingTV": 0, "StreamingMovies": 0,
            "PaperlessBilling": int(paperless),
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            "gender": 0,
            "Contract_One year": int(contract == "One year"),
            "Contract_Two year": int(contract == "Two year"),
            "InternetService_Fiber optic": int(internet == "Fiber optic"),
            "InternetService_No": int(internet == "No"),
            "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
            "PaymentMethod_Electronic check": int(payment == "Electronic check"),
            "PaymentMethod_Mailed check": int(payment == "Mailed check"),
            "charge_per_tenure": monthly_charges / (tenure + 1),
            "service_count": int(online_security) + int(tech_support),
            "tenure_bucket_1_2yr": int(12 <= tenure < 24),
            "tenure_bucket_2_4yr": int(24 <= tenure < 48),
            "tenure_bucket_gt_4yr": int(tenure >= 48),
        }

        X = pd.DataFrame([row])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        xgb_p  = xgb.predict_proba(X)[:, 1][0]
        lgbm_p = lgbm.predict_proba(X)[:, 1][0]
        prob   = 0.55 * xgb_p + 0.45 * lgbm_p
        churn  = prob >= THRESHOLD
        risk   = "🔴 HIGH" if prob >= 0.65 else ("🟡 MEDIUM" if prob >= 0.35 else "🟢 LOW")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Prediction", "WILL CHURN ⚠️" if churn else "WILL RETAIN ✅")
        c3.metric("Risk Tier", risk)

        # SHAP waterfall
        explainer  = shap.TreeExplainer(xgb)
        shap_vals  = explainer.shap_values(X)
        top_idx    = np.argsort(np.abs(shap_vals[0]))[::-1][:8]
        top_feat   = [feature_names[i] for i in top_idx]
        top_shap   = [shap_vals[0][i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors  = ["#DC2626" if v > 0 else "#2563EB" for v in top_shap]
        ax.barh(range(len(top_feat)), top_shap, color=colors)
        ax.set_yticks(range(len(top_feat)))
        ax.set_yticklabels(top_feat, fontsize=10)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value (impact on churn probability)")
        ax.set_title("Why this prediction? (SHAP reason codes)", fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

# ── Tab 2: Performance ────────────────────────────────────────────────────────

with tab2:
    st.subheader("Model Performance — IBM Telco Test Set (n=1,409)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",   metrics["roc_auc"])
    m2.metric("Precision", metrics["precision"])
    m3.metric("Recall",    metrics["recall"])
    m4.metric("F1-Score",  metrics["f1"])

    st.info(
        f"Threshold tuned to **{THRESHOLD}** on validation set to maximise F1 on the minority "
        "churn class. Higher recall means fewer at-risk customers slip through undetected."
    )

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image("results/roc_curve.png", caption="ROC Curve", use_column_width=True)
        except:
            st.warning("Run generate_results.py to see plots.")
    with col2:
        try:
            st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
        except:
            st.warning("Run generate_results.py to see plots.")

    try:
        st.image("results/shap_summary.png", caption="SHAP Feature Importance", use_column_width=True)
    except:
        pass

# ── Tab 3: Insights ───────────────────────────────────────────────────────────

with tab3:
    st.subheader("Churn Rate by Customer Segment")
    try:
        st.image("results/churn_by_segment.png", use_column_width=True)
    except:
        st.warning("Run generate_results.py to see segment analysis.")

    from src.data.load_data import load_telco
    df = load_telco()

    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        rates = df.groupby("Contract")["Churn"].mean() * 100
        st.write("**Month-to-month customers churn at:**",
                 f"{rates.get('Month-to-month', 0):.1f}% vs "
                 f"{rates.get('Two year', 0):.1f}% for 2-year contracts")
        rates2 = df.groupby("InternetService")["Churn"].mean() * 100
        st.write("**Fiber optic churn rate:**",
                 f"{rates2.get('Fiber optic', 0):.1f}%")

    with col2:
        rates3 = df.groupby("PaymentMethod")["Churn"].mean() * 100
        st.write("**Electronic check churn rate:**",
                 f"{rates3.get('Electronic check', 0):.1f}%")
        st.write("**Overall churn rate:**", f"{df['Churn'].mean()*100:.1f}%")

"""
api/main.py
-----------
FastAPI endpoint for real-time churn prediction.
Returns churn probability, risk tier, and top SHAP reason codes.

Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

import sys, os, time
sys.path.insert(0, ".")

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── Load models at startup ────────────────────────────────────────────────────
MODEL_DIR = "models"

try:
    xgb_model      = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    lgbm_model     = joblib.load(f"{MODEL_DIR}/lgbm_model.pkl")
    feature_names  = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    explainer      = shap.TreeExplainer(xgb_model)
    MODELS_LOADED  = True
except Exception as e:
    MODELS_LOADED  = False
    LOAD_ERROR     = str(e)

app = FastAPI(
    title="Churn Prediction API",
    description="XGBoost + LightGBM ensemble for customer churn prediction with SHAP explainability.",
    version="1.0.0",
)

# ── Request / Response schemas ────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    tenure: int                     = Field(..., ge=0, le=72, example=14)
    monthly_charges: float          = Field(..., ge=18.8, le=118.8, example=89.5)
    total_charges: float            = Field(..., ge=0, example=1253.0)
    senior_citizen: int             = Field(0, ge=0, le=1, example=0)
    partner: int                    = Field(0, ge=0, le=1, example=1)
    dependents: int                 = Field(0, ge=0, le=1, example=0)
    phone_service: int              = Field(1, ge=0, le=1, example=1)
    multiple_lines: int             = Field(0, ge=0, le=1, example=0)
    online_security: int            = Field(0, ge=0, le=1, example=0)
    online_backup: int              = Field(0, ge=0, le=1, example=0)
    device_protection: int          = Field(0, ge=0, le=1, example=0)
    tech_support: int               = Field(0, ge=0, le=1, example=0)
    streaming_tv: int               = Field(0, ge=0, le=1, example=1)
    streaming_movies: int           = Field(0, ge=0, le=1, example=1)
    paperless_billing: int          = Field(1, ge=0, le=1, example=1)
    contract_month_to_month: int    = Field(1, ge=0, le=1, example=1)
    contract_two_year: int          = Field(0, ge=0, le=1, example=0)
    internet_fiber_optic: int       = Field(1, ge=0, le=1, example=1)
    internet_no: int                = Field(0, ge=0, le=1, example=0)
    payment_electronic_check: int   = Field(1, ge=0, le=1, example=1)
    payment_mailed_check: int       = Field(0, ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    top_reasons: list[dict]
    latency_ms: float


def build_feature_vector(req: CustomerFeatures) -> pd.DataFrame:
    """Map API request to the exact feature vector the model expects."""
    row = {
        "SeniorCitizen": req.senior_citizen,
        "Partner": req.partner,
        "Dependents": req.dependents,
        "tenure": req.tenure,
        "PhoneService": req.phone_service,
        "MultipleLines": req.multiple_lines,
        "OnlineSecurity": req.online_security,
        "OnlineBackup": req.online_backup,
        "DeviceProtection": req.device_protection,
        "TechSupport": req.tech_support,
        "StreamingTV": req.streaming_tv,
        "StreamingMovies": req.streaming_movies,
        "PaperlessBilling": req.paperless_billing,
        "MonthlyCharges": req.monthly_charges,
        "TotalCharges": req.total_charges,
        "gender": 0,  # not sent by client; default neutral
        # One-hot encoded columns
        "Contract_One year": 0,
        "Contract_Two year": req.contract_two_year,
        "InternetService_Fiber optic": req.internet_fiber_optic,
        "InternetService_No": req.internet_no,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": req.payment_electronic_check,
        "PaymentMethod_Mailed check": req.payment_mailed_check,
        # Engineered
        "charge_per_tenure": req.monthly_charges / (req.tenure + 1),
        "service_count": (req.multiple_lines + req.online_security + req.online_backup
                          + req.device_protection + req.tech_support
                          + req.streaming_tv + req.streaming_movies),
    }

    # Tenure bucket dummies
    for bucket in ["tenure_bucket_1_2yr", "tenure_bucket_2_4yr", "tenure_bucket_gt_4yr"]:
        row[bucket] = 0
    if 12 <= req.tenure < 24:
        row["tenure_bucket_1_2yr"] = 1
    elif 24 <= req.tenure < 48:
        row["tenure_bucket_2_4yr"] = 1
    elif req.tenure >= 48:
        row["tenure_bucket_gt_4yr"] = 1

    df = pd.DataFrame([row])
    # Align to training columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


THRESHOLD = 0.30  # tuned on validation set for best F1


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {LOAD_ERROR}")

    t0 = time.perf_counter()
    X = build_feature_vector(customer)

    xgb_p  = xgb_model.predict_proba(X)[:, 1][0]
    lgbm_p = lgbm_model.predict_proba(X)[:, 1][0]
    prob   = float(0.55 * xgb_p + 0.45 * lgbm_p)

    churn = prob >= THRESHOLD
    risk  = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")

    # SHAP reason codes
    shap_vals = explainer.shap_values(X)[0]
    reasons   = sorted(
        [{"feature": f, "impact": round(float(v), 4)}
         for f, v in zip(feature_names, shap_vals)],
        key=lambda x: abs(x["impact"]), reverse=True
    )[:5]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=bool(churn),
        risk_tier=risk,
        top_reasons=reasons,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    """Score up to 100 customers in a single call."""
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Max 100 customers per batch.")
    return [predict(c) for c in customers]

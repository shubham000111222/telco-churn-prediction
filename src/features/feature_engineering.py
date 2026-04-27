"""
feature_engineering.py
-----------------------
Transforms raw Telco columns into model-ready features.

Key decisions:
  - Binary Yes/No → 0/1
  - Multi-category → one-hot (drop_first to avoid multicollinearity)
  - Tenure buckets  → ordinal signal the tree models can use cleanly
  - Payment risk proxy → electronic check correlates with higher churn
"""

import pandas as pd
import numpy as np


BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

GENDER_COL = "gender"  # Male/Female → 1/0

# These have a third level ("No internet service" / "No phone service")
TRINARY_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Gender ---
    df["gender"] = (df["gender"] == "Male").astype(int)

    # --- Binary columns ---
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # --- Trinary: collapse "No X service" → 0, "No" → 0, "Yes" → 1 ---
    for col in TRINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # --- One-hot encode ---
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True, dtype=int)

    # --- Engineered features ---
    # Tenure buckets (non-linear churn risk by lifecycle stage)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["lt_1yr", "1_2yr", "2_4yr", "gt_4yr"],
    )
    df = pd.get_dummies(df, columns=["tenure_bucket"], drop_first=True, dtype=int)

    # Charge-to-tenure ratio (high monthly spend + short tenure → churn risk)
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Service count (customers with more services churn less)
    service_cols = [c for c in df.columns if c in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ]]
    df["service_count"] = df[service_cols].sum(axis=1)

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    exclude = {"Churn"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load_data import load_telco

    df = load_telco()
    X = df.drop(columns=["Churn"])
    X_enc = encode_features(X)
    print(f"Features after encoding: {X_enc.shape[1]}")
    print(X_enc.dtypes.value_counts())
    print(X_enc.head(2))

"""
load_data.py
------------
Loads and performs minimal cleaning on the IBM Telco Customer Churn dataset.
Dataset: data/raw/telco_churn.csv  (7,043 rows × 21 columns)
Source : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_telco(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges is stored as string; spaces appear for 0-tenure customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Drop non-predictive ID column
    df.drop(columns=["customerID"], inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def split(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    seed: int = 42,
):
    """Stratified 80/20 split preserving class ratio."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    df = load_telco()
    print(f"Shape      : {df.shape}")
    print(f"Churn rate : {df['Churn'].mean():.1%}")
    print(df.head(3))

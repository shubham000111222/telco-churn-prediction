"""
generate_results.py
-------------------
Generates and saves all plots to results/ for the README.
Run AFTER train.py has saved models.
"""

import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from sklearn.metrics import (
    RocCurveDisplay, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)

from src.data.load_data import load_telco, split
from src.features.feature_engineering import encode_features

# ── palette ──────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GRAY   = "#6B7280"
BG     = "#F9FAFB"
FONT   = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
})


def load_artifacts():
    ensemble     = joblib.load("models/ensemble_model.pkl")
    xgb          = joblib.load("models/xgb_model.pkl")
    feature_names= joblib.load("models/feature_names.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)

    df = load_telco()
    _, X_raw_test, _, y_test = split(df)
    X_test = encode_features(X_raw_test)
    X_test, _ = X_test.align(pd.DataFrame(columns=feature_names), join="right", axis=1, fill_value=0)
    X_test = X_test[feature_names]

    return ensemble, xgb, X_test, y_test, feature_names, metrics


# ── 1. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc(ensemble, X_test, y_test):
    proba = ensemble.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Ensemble (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1], "--", color=GRAY, lw=1.2, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Churn Prediction Ensemble", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig("results/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ roc_curve.png")


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion(ensemble, X_test, y_test):
    pred = ensemble.predict(X_test)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix (threshold = 0.50)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ confusion_matrix.png")


# ── 3. SHAP Summary ───────────────────────────────────────────────────────────

def plot_shap(xgb, X_test, feature_names, max_display=15):
    explainer   = shap.TreeExplainer(xgb)
    sample      = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      max_display=max_display, show=False, plot_size=None)
    plt.title("SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ shap_summary.png")


# ── 4. Churn Rate by Segment ──────────────────────────────────────────────────

def plot_churn_by_segment():
    df = load_telco()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    segments = [
        ("Contract", "Contract Type"),
        ("InternetService", "Internet Service"),
        ("PaymentMethod", "Payment Method"),
    ]

    for ax, (col, title) in zip(axes, segments):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        colors = [BLUE if i == 0 else GRAY for i in range(len(rates))]
        bars = ax.bar(range(len(rates)), rates.values * 100, color=colors, width=0.6)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Churn Rate (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for bar, val in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Churn Rate by Customer Segment", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("results/churn_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ churn_by_segment.png")


# ── 5. Metrics Summary Card ───────────────────────────────────────────────────

def plot_metrics_card(metrics):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")

    labels  = ["ROC-AUC", "Precision", "Recall", "F1-Score"]
    values  = [metrics["roc_auc"], metrics["precision"],
                metrics["recall"], metrics["f1"]]
    colors  = [BLUE, "#059669", "#D97706", "#7C3AED"]

    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        x = 0.12 + i * 0.24
        ax.text(x, 0.72, f"{val:.3f}", transform=ax.transAxes,
                fontsize=22, fontweight="bold", color=col, ha="center")
        ax.text(x, 0.35, lbl, transform=ax.transAxes,
                fontsize=11, color=GRAY, ha="center")

    ax.set_title("Model Performance — Held-out Test Set (n=1,409)", 
                 fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig("results/metrics_card.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ metrics_card.png")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading artifacts …")
    ensemble, xgb, X_test, y_test, feature_names, metrics = load_artifacts()

    print("Generating plots …")
    plot_roc(ensemble, X_test, y_test)
    plot_confusion(ensemble, X_test, y_test)
    plot_shap(xgb, X_test, feature_names)
    plot_churn_by_segment()
    plot_metrics_card(metrics)

    print(f"\n✓ All plots saved to results/")

"""
train.py
--------
Full training pipeline:
  1. Load & encode Telco data
  2. Handle class imbalance with SMOTE
  3. Tune XGBoost + LightGBM with Optuna (Bayesian search)
  4. Blend ensemble predictions
  5. Evaluate on held-out test set
  6. Log everything to MLflow
  7. Save model artifacts
"""

import sys, os, warnings, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.load_data import load_telco, split
from src.features.feature_engineering import encode_features

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
os.makedirs("models", exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def prepare_data():
    df = load_telco()
    X_raw, X_raw_test, y_train, y_test = split(df)

    X_train_enc = encode_features(X_raw)
    X_test_enc  = encode_features(X_raw_test)

    # Align columns (one-hot can differ if a category is missing in a split)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    feature_names = list(X_train_enc.columns)

    # SMOTE on training set only
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_enc, y_train)

    print(f"Train (after SMOTE): {X_train_res.shape}  |  churn: {y_train_res.mean():.1%}")
    print(f"Test               : {X_test_enc.shape}   |  churn: {y_test.mean():.1%}")

    return X_train_res, y_train_res, X_test_enc, y_test, feature_names


def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    print(f"\n── {name} ──────────────────────────────")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, proba):.4f}")
    print(f"  Precision : {precision_score(y_test, pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, pred):.4f}")
    print(f"  F1        : {f1_score(y_test, pred):.4f}")
    return roc_auc_score(y_test, proba)


# ── Optuna objectives ─────────────────────────────────────────────────────────

def tune_xgb(X_train, y_train, X_val, y_val, n_trials=40):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0, 5),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            random_state=SEED, eval_metric="auc", use_label_encoder=False,
            n_jobs=-1,
        )
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best AUC (val): {study.best_value:.4f}")
    return study.best_params


def tune_lgbm(X_train, y_train, X_val, y_val, n_trials=40):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves       = trial.suggest_int("num_leaves", 20, 150),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m = LGBMClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[])
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  LGBM best AUC (val): {study.best_value:.4f}")
    return study.best_params


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_experiment("churn-prediction")

    print("Loading & encoding data …")
    X_train, y_train, X_test, y_test, feature_names = prepare_data()

    # Use 15% of training set as Optuna validation
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_val, y_tr, y_val = tts(X_train, y_train, test_size=0.15,
                                    random_state=SEED, stratify=y_train)

    with mlflow.start_run(run_name="xgb_lgbm_ensemble"):

        # ── Tune ────────────────────────────────────────────────────────────
        print("\nTuning XGBoost …")
        xgb_params = tune_xgb(X_tr, y_tr, X_val, y_val, n_trials=40)

        print("\nTuning LightGBM …")
        lgbm_params = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=40)

        # ── Retrain on full training set ─────────────────────────────────
        print("\nRetraining on full training set …")
        xgb  = XGBClassifier(**xgb_params,  random_state=SEED,
                              eval_metric="auc", use_label_encoder=False, n_jobs=-1)
        lgbm = LGBMClassifier(**lgbm_params, random_state=SEED, n_jobs=-1, verbose=-1)

        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        # ── Ensemble (simple average) ─────────────────────────────────────
        class EnsembleModel:
            def __init__(self, models, weights=None):
                self.models  = models
                self.weights = weights or [1/len(models)] * len(models)

            def predict_proba(self, X):
                probas = np.array([m.predict_proba(X)[:, 1] for m in self.models])
                avg    = np.average(probas, axis=0, weights=self.weights)
                return np.column_stack([1-avg, avg])

            def predict(self, X, threshold=0.5):
                return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        ensemble = EnsembleModel([xgb, lgbm], weights=[0.55, 0.45])

        # ── Evaluate ──────────────────────────────────────────────────────
        evaluate("XGBoost",  xgb,      X_test, y_test)
        evaluate("LightGBM", lgbm,     X_test, y_test)
        auc = evaluate("Ensemble",   ensemble, X_test, y_test)

        proba     = ensemble.predict_proba(X_test)[:, 1]
        pred      = ensemble.predict(X_test)
        precision = precision_score(y_test, pred)
        recall    = recall_score(y_test, pred)
        f1        = f1_score(y_test, pred)

        print("\n── Final Test Metrics ───────────────────────────────────")
        print(classification_report(y_test, pred, target_names=["No Churn","Churn"]))

        # ── Log to MLflow ─────────────────────────────────────────────────
        mlflow.log_params({**{f"xgb_{k}": v for k, v in xgb_params.items()},
                           **{f"lgbm_{k}": v for k, v in lgbm_params.items()}})
        mlflow.log_metrics({
            "test_roc_auc":  round(auc, 4),
            "test_precision": round(precision, 4),
            "test_recall":   round(recall, 4),
            "test_f1":       round(f1, 4),
        })

        # ── Save artifacts ────────────────────────────────────────────────
        joblib.dump(xgb,      "models/xgb_model.pkl")
        joblib.dump(lgbm,     "models/lgbm_model.pkl")
        joblib.dump(ensemble, "models/ensemble_model.pkl")
        joblib.dump(feature_names, "models/feature_names.pkl")

        metrics = {
            "roc_auc":  round(auc, 4),
            "precision": round(precision, 4),
            "recall":   round(recall, 4),
            "f1":       round(f1, 4),
        }
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Models saved to models/")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  F1      : {f1:.4f}")

    return ensemble, xgb, lgbm, X_test, y_test, feature_names


if __name__ == "__main__":
    main()

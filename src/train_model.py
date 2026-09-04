"""
================================================================================
JAL SANKETH - MACHINE LEARNING TRAINING PIPELINE (XGBOOST CLASSIFIER)
Module Path: src/train_model.py
================================================================================

DISCLAIMER & PROTOTYPE STATUS:
------------------------------
This machine learning pipeline is developed as a PROTOTYPE for the Smart India
Hackathon 2026. It is trained on a synthetic/demo meteorological dataset
(data/raw/weather_demo.csv) to demonstrate automated heavy-rainfall early
warning capabilities.

THIS MODEL IS NOT PRODUCTION-READY AND MUST NOT BE USED FOR OPERATIONAL
CIVIL PROTECTION OR DISASTER DEPLOYMENT WITHOUT EXTENSIVE VALIDATION ON
OFFICIAL MULTI-YEAR OBSERVATIONAL DATASETS (E.G., IMD AWS/DWR FEEDS).
================================================================================
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Headless matplotlib configuration for headless script execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from src.feature_engineering import FeatureEngineer, load_engineered_data


def train_heavy_rain_classifier(
    raw_data_path: str = "data/raw/weather_demo.csv",
    model_dir: str = "models",
    figures_dir: str = "reports/figures",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Trains an XGBClassifier to predict heavy rainfall events (rainfall >= 64.5 mm),
    evaluates classification performance metrics, and saves model artifacts.

    Parameters:
    -----------
    raw_data_path : str
        Path to the raw/demo weather dataset.
    model_dir : str
        Directory to persist trained model and feature metadata.
    figures_dir : str
        Directory to save evaluation visualizations.
    test_size : float
        Proportion of dataset reserved for testing (default: 0.20).
    random_state : int
        Seed for reproducible train-test split and XGBoost training.

    Returns:
    --------
    Tuple[XGBClassifier, Dict[str, Any]]: Trained model instance and evaluation metrics.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 80)
    print("      JAL SANKETH - XGBOOST HEAVY RAIN CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"[*] Loading & engineering features from: {raw_data_path}")

    # 1. Load engineered feature matrix and binary target
    X, y_reg, y_clf, full_df = load_engineered_data(raw_data_path=raw_data_path)
    feature_names = list(X.columns)

    print(f"[+] Total Observations: {X.shape[0]:,} records")
    print(f"[+] Features Engineered: {X.shape[1]} variables")
    
    # Class distribution
    n_pos = int(y_clf.sum())
    n_neg = int(len(y_clf) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[+] Class Balance: Class 0 = {n_neg:,} ({n_neg/len(y_clf)*100:.2f}%) | Class 1 (Heavy Rain) = {n_pos:,} ({n_pos/len(y_clf)*100:.2f}%)")
    print(f"[+] Computed Imbalance scale_pos_weight: {scale_pos_weight:.2f}")

    # 2. Train-Test Split (Stratified to preserve heavy-rain class proportion in test set)
    print(f"\n[*] Splitting dataset (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )
    print(f"  * Training Set Size : {X_train.shape[0]:,} samples")
    print(f"  * Testing Set Size  : {X_test.shape[0]:,} samples")

    # 3. Configure and Train XGBoost Classifier
    print("\n[*] Initializing and Training XGBClassifier...")
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.15,
        scale_pos_weight=scale_pos_weight * 0.75,  # Balanced cost-sensitive weighting
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    print("  [OK] Model training completed successfully.")

    # 4. Model Inference on Test Set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 5. Comprehensive Metric Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 80)
    print("                 MODEL PERFORMANCE EVALUATION METRICS")
    print("=" * 80)
    print(f"  * Accuracy                     : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  * Precision (Heavy Rain >=64.5): {prec:.4f} ({prec * 100:.2f}%)")
    print(f"  * Recall (Hazard Capture Rate) : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  * F1-Score (Harmonic Mean)     : {f1:.4f}")
    print(f"  * ROC-AUC Score                : {roc_auc:.4f}")
    print("-" * 80)

    print("\n--- CONFUSION MATRIX ---")
    print(f"                Predicted Normal (0)   Predicted Heavy Rain (1)")
    print(f"Actual Normal (0)      {cm[0, 0]:>8,}               {cm[0, 1]:>8,}")
    print(f"Actual Heavy (1)       {cm[1, 0]:>8,}               {cm[1, 1]:>8,}")

    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["Normal Rain (0)", "Heavy Rain (1)"], digits=4))

    # 6. Feature Importance Extraction
    importances = model.feature_importances_
    feat_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- TOP 10 MOST INFLUENTIAL METEOROLOGICAL / TERRAIN FEATURES ---")
    for i, row in feat_importance_df.head(10).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<26} : {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # 7. Generate and Save Visualizations
    # Plot A: Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_feats = feat_importance_df.head(15)
    sns.barplot(
        data=top_feats,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="Blues_r",
        edgecolor="#1e293b",
        ax=ax
    )
    ax.set_title("XGBoost Feature Importance (Top 15 Predictors)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Weight (Gain)", fontsize=11)
    ax.set_ylabel("Engineered Feature", fontsize=11)
    plt.tight_layout()
    feat_imp_path = os.path.join(figures_dir, "06_xgboost_feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved feature importance plot : {feat_imp_path}")

    # Plot B: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (<64.5mm)", "Heavy Rain (>=64.5mm)"],
        yticklabels=["Normal (<64.5mm)", "Heavy Rain (>=64.5mm)"],
        cbar=False,
        ax=ax
    )
    ax.set_title("Confusion Matrix: Heavy Rain Classification", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Ground Truth Class", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "07_xgboost_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix plot    : {cm_path}")

    # 8. Persist Model Artifacts
    model_save_path = os.path.join(model_dir, "heavy_rain_model.joblib")
    feature_cols_save_path = os.path.join(model_dir, "feature_columns.json")
    metrics_save_path = os.path.join(model_dir, "model_metrics.json")

    # Save joblib model
    joblib.dump(model, model_save_path)
    print(f"[+] Saved trained model to        : {model_save_path}")

    # Save feature column order
    with open(feature_cols_save_path, "w") as f:
        json.dump(feature_names, f, indent=4)
    print(f"[+] Saved feature column order to : {feature_cols_save_path}")

    # Save metrics metadata
    metrics_dict = {
        "model_architecture": "XGBClassifier (Extreme Gradient Boosting)",
        "prototype_status": "Hackathon Demo Prototype - Trained on Synthetic Meteorological Data",
        "dataset_records_total": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1])
            }
        },
        "top_features": feat_importance_df.head(10).to_dict(orient="records"),
        "features_used": feature_names
    }
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[+] Saved evaluation metrics to   : {metrics_save_path}")

    print("\n" + "=" * 80)
    print("NOTICE: Machine learning training completed for Jal Sanketh Prototype.")
    print("========================================================================\n")

    return model, metrics_dict


if __name__ == "__main__":
    trained_model, eval_metrics = train_heavy_rain_classifier()

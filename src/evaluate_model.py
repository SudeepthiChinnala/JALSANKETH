"""
================================================================================
JAL SANKETH - MODEL EVALUATION & SCENARIO INFERENCE TEST SUITE
Module Path: src/evaluate_model.py
================================================================================

This module loads the trained XGBClassifier (models/heavy_rain_model.joblib),
re-evaluates it on the held-out test split, prints authentic performance
metrics, and evaluates the model across 5+ diverse meteorological test scenarios.

Zero Data Leakage / Authentic Metrics Guarantee:
------------------------------------------------
1. All metrics are computed dynamically at runtime using scikit-learn on the
   isolated test split (test_size=0.20, random_state=42).
2. Probability calibration and classification decisions for manual weather
   scenarios are computed directly via model.predict_proba().
================================================================================
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

from src.feature_engineering import FeatureEngineer, load_engineered_data


def load_model_and_metadata(
    model_path: str = "models/heavy_rain_model.joblib",
    features_path: str = "models/feature_columns.json"
):
    """
    Loads serialized XGBoost model and feature column order.
    """
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"[!] Model or feature columns missing. Please run src/train_model.py first.")

    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


def evaluate_on_test_set(
    model,
    feature_columns: List[str],
    raw_data_path: str = "data/raw/weather_demo.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Reconstructs the isolated test split and executes model evaluation.
    """
    # 1. Load engineered dataset
    X, y_reg, y_clf, full_df = load_engineered_data(raw_data_path=raw_data_path)
    X = X[feature_columns]

    # 2. Stratified train-test split (reproducible seed 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf
    )

    # 3. Model inference on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. Metric computation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    clf_rep = classification_report(y_test, y_pred, target_names=["Normal Rain (<64.5mm)", "Heavy Rain (>=64.5mm)"], digits=4)

    # 5. Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("=" * 85)
    print("      JAL SANKETH - AUTHENTIC TEST SET EVALUATION REPORT (HELD-OUT 20% DATA)")
    print("=" * 85)
    print(f"[*] Total Test Observations  : {X_test.shape[0]:,} samples")
    print(f"[*] True Normal Rain Count   : {int((y_test == 0).sum()):,} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"[*] True Heavy Rain Count    : {int((y_test == 1).sum()):,} ({(y_test == 1).mean()*100:.2f}%)")
    print("-" * 85)

    print("\n--- 1. OVERALL CLASSIFICATION PERFORMANCE METRICS ---")
    print(f"  * Accuracy                     : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  * Precision (Heavy Rain >=64.5): {prec:.4f} ({prec * 100:.2f}%)")
    print(f"  * Recall (Hazard Capture Rate) : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  * F1-Score (Harmonic Mean)     : {f1:.4f}")
    print(f"  * ROC-AUC Score                : {roc_auc:.4f}")

    print("\n--- 2. CONFUSION MATRIX ---")
    print(f"                          Predicted Normal (0)   Predicted Heavy Rain (1)")
    print(f"  Actual Normal (<64.5mm)         {cm[0, 0]:>8,}               {cm[0, 1]:>8,}")
    print(f"  Actual Heavy  (>=64.5mm)        {cm[1, 0]:>8,}               {cm[1, 1]:>8,}")
    print(f"  -> Correctly Detected Heavy Events : {cm[1, 1]} / {cm[1, 0] + cm[1, 1]} ({rec*100:.1f}%)")
    print(f"  -> False Alarm Count               : {cm[0, 1]} / {cm[0, 0] + cm[0, 1]} ({cm[0, 1]/(cm[0, 0]+cm[0, 1])*100:.1f}%)")

    print("\n--- 3. FULL CLASSIFICATION REPORT ---")
    print(clf_rep)

    print("\n--- 4. FEATURE IMPORTANCE RANKING (TOP 12 PREDICTORS) ---")
    for i, row in feat_imp.head(12).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<26} : {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "feature_importances": feat_imp
    }


def create_scenario_vector(
    feature_columns: List[str],
    temp: float,
    hum: float,
    pres: float,
    wind: float,
    month: int,
    day_of_year: int,
    rain_lag_1d: float = 0.0,
    rain_lag_3d_mean: float = 0.0,
    rain_lag_7d_sum: float = 0.0,
    pres_trend_24h: float = 0.0,
    temp_trend_24h: float = 0.0,
    elevation: float = 10.0,
    drainage_factor: float = 0.40
) -> pd.DataFrame:
    """
    Constructs a fully engineered single-row feature vector matching the exact model schema.
    """
    # 1. Cyclical seasonality
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    is_sw = 1 if month in [6, 7, 8, 9] else 0
    is_ne = 1 if month in [10, 11, 12] else 0

    # 2. Thermodynamics & Indices
    dew_point_est = temp - ((100.0 - hum) / 5.0)
    dew_point_depression = max(temp - dew_point_est, 0.0)
    e_sat = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    e_actual = e_sat * (hum / 100.0)
    vapor_pressure_deficit = max(e_sat - e_actual, 0.0)

    norm_hum = hum / 100.0
    norm_wind = np.clip(wind / 40.0, 0.1, 2.0)
    norm_press = np.clip(1013.25 / max(pres, 800.0), 0.9, 1.3)
    instability_index = norm_hum * norm_wind * norm_press

    # 3. Terrain
    drainage_deficit = 1.0 - drainage_factor
    elevation_vulnerability = np.clip(1.0 - (elevation / 350.0), 0.05, 1.0)
    runoff_potential = drainage_deficit * elevation_vulnerability

    row_data = {
        "temperature": temp,
        "humidity": hum,
        "pressure": pres,
        "wind_speed": wind,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "is_monsoon_sw": is_sw,
        "is_monsoon_ne": is_ne,
        "rainfall_lag_1d": rain_lag_1d,
        "rainfall_lag_3d_mean": rain_lag_3d_mean,
        "rainfall_lag_7d_sum": rain_lag_7d_sum,
        "pressure_trend_24h": pres_trend_24h,
        "temp_trend_24h": temp_trend_24h,
        "dew_point_est": dew_point_est,
        "dew_point_depression": dew_point_depression,
        "vapor_pressure_deficit": vapor_pressure_deficit,
        "instability_index": instability_index,
        "elevation": elevation,
        "drainage_factor": drainage_factor,
        "drainage_deficit": drainage_deficit,
        "elevation_vulnerability": elevation_vulnerability,
        "runoff_potential_index": runoff_potential
    }

    df_row = pd.DataFrame([row_data])[feature_columns]
    return df_row


def run_scenario_simulations(model, feature_columns: List[str]):
    """
    Evaluates model inference on 5+ diverse, realistic meteorological scenarios.
    """
    scenarios = [
        {
            "id": "SCENARIO-1",
            "name": "Dry Winter / Clear Sky Conditions",
            "description": "Mild January day in Delhi plains with low moisture, high barometric pressure, and calm winds.",
            "params": {
                "temp": 18.5,
                "hum": 38.0,
                "pres": 1018.0,
                "wind": 6.5,
                "month": 1,
                "day_of_year": 15,
                "rain_lag_1d": 0.0,
                "rain_lag_3d_mean": 0.0,
                "rain_lag_7d_sum": 0.0,
                "pres_trend_24h": 0.5,
                "temp_trend_24h": 0.2,
                "elevation": 216.0,
                "drainage_factor": 0.48
            },
            "expected_outcome": "Near-Zero Probability (Normal/Dry)"
        },
        {
            "id": "SCENARIO-2",
            "name": "Pre-Monsoon Convective Heatwave / Dry Squall",
            "description": "Hot May afternoon in North-Central India with moderate humidity and high convective heat.",
            "params": {
                "temp": 41.0,
                "hum": 48.0,
                "pres": 1005.0,
                "wind": 18.0,
                "month": 5,
                "day_of_year": 135,
                "rain_lag_1d": 0.0,
                "rain_lag_3d_mean": 2.0,
                "rain_lag_7d_sum": 5.0,
                "pres_trend_24h": -1.5,
                "temp_trend_24h": 1.2,
                "elevation": 53.0,
                "drainage_factor": 0.42
            },
            "expected_outcome": "Low Probability (<15%)"
        },
        {
            "id": "SCENARIO-3",
            "name": "Typical Moderate Monsoon Rain Day",
            "description": "Overcast July morning in Bengaluru with steady moderate monsoon drizzle and elevated humidity.",
            "params": {
                "temp": 26.5,
                "hum": 76.0,
                "pres": 998.0,
                "wind": 20.0,
                "month": 7,
                "day_of_year": 195,
                "rain_lag_1d": 12.0,
                "rain_lag_3d_mean": 18.0,
                "rain_lag_7d_sum": 45.0,
                "pres_trend_24h": -2.0,
                "temp_trend_24h": -1.5,
                "elevation": 920.0,
                "drainage_factor": 0.45
            },
            "expected_outcome": "Moderate Probability (20% - 50%)"
        },
        {
            "id": "SCENARIO-4",
            "name": "Active Coastal Monsoonal Heavy Downpour (Mumbai Trough)",
            "description": "Mid-July coastal low in Mumbai with high atmospheric saturation, moderate depression, and antecedent downpours.",
            "params": {
                "temp": 27.0,
                "hum": 89.0,
                "pres": 992.0,
                "wind": 36.0,
                "month": 7,
                "day_of_year": 205,
                "rain_lag_1d": 48.0,
                "rain_lag_3d_mean": 52.0,
                "rain_lag_7d_sum": 160.0,
                "pres_trend_24h": -5.5,
                "temp_trend_24h": -2.2,
                "elevation": 8.0,
                "drainage_factor": 0.35
            },
            "expected_outcome": "High Probability (>60%)"
        },
        {
            "id": "SCENARIO-5",
            "name": "Severe Cyclonic Cloudburst Deluge (Chennai NE Monsoon Peak)",
            "description": "November cyclonic depression off Tamil Nadu coast with deep barometric collapse, extreme humidity (98%), and gale winds.",
            "params": {
                "temp": 24.5,
                "hum": 98.0,
                "pres": 978.0,
                "wind": 58.0,
                "month": 11,
                "day_of_year": 320,
                "rain_lag_1d": 95.0,
                "rain_lag_3d_mean": 110.0,
                "rain_lag_7d_sum": 280.0,
                "pres_trend_24h": -12.0,
                "temp_trend_24h": -3.8,
                "elevation": 7.0,
                "drainage_factor": 0.30
            },
            "expected_outcome": "Critical / Very High Probability (>85%)"
        },
        {
            "id": "SCENARIO-6",
            "name": "Western Ghats Orographic Flash Flood Surge (Wayanad)",
            "description": "August orographic monsoon surge against Western Ghats foothills with saturated soil and sustained squalls.",
            "params": {
                "temp": 22.0,
                "hum": 96.0,
                "pres": 922.0,
                "wind": 44.0,
                "month": 8,
                "day_of_year": 225,
                "rain_lag_1d": 80.0,
                "rain_lag_3d_mean": 92.0,
                "rain_lag_7d_sum": 240.0,
                "pres_trend_24h": -7.0,
                "temp_trend_24h": -2.5,
                "elevation": 750.0,
                "drainage_factor": 0.62
            },
            "expected_outcome": "High/Critical Probability (>75%)"
        }
    ]

    print("\n" + "=" * 85)
    print("      JAL SANKETH - SCENARIO-BASED INFERENCE & PROBABILITY TEST SUITE")
    print("=" * 85)

    for sc in scenarios:
        p = sc["params"]
        X_sc = create_scenario_vector(
            feature_columns=feature_columns,
            temp=p["temp"],
            hum=p["hum"],
            pres=p["pres"],
            wind=p["wind"],
            month=p["month"],
            day_of_year=p["day_of_year"],
            rain_lag_1d=p["rain_lag_1d"],
            rain_lag_3d_mean=p["rain_lag_3d_mean"],
            rain_lag_7d_sum=p["rain_lag_7d_sum"],
            pres_trend_24h=p["pres_trend_24h"],
            temp_trend_24h=p["temp_trend_24h"],
            elevation=p["elevation"],
            drainage_factor=p["drainage_factor"]
        )

        pred_class = int(model.predict(X_sc)[0])
        pred_prob_heavy = float(model.predict_proba(X_sc)[0, 1])

        # Risk Classification Badge
        if pred_prob_heavy >= 0.75:
            risk_tier = "[CRITICAL ALERT - HIGH FLOOD RISK]"
        elif pred_prob_heavy >= 0.50:
            risk_tier = "[HIGH ALERT - SEVERE WARNING]"
        elif pred_prob_heavy >= 0.25:
            risk_tier = "[MODERATE ADVISORY - WATCH]"
        else:
            risk_tier = "[NORMAL - LOW HAZARD]"

        print(f"\n>> [{sc['id']}] {sc['name']}")
        print(f"  * Description      : {sc['description']}")
        print(f"  * Key Inputs       : Temp={p['temp']} C | Hum={p['hum']}% | Press={p['pres']} hPa | Wind={p['wind']} km/h | 24h Drop={p['pres_trend_24h']} hPa")
        print(f"  * Antecedent Rain  : Lag 1d={p['rain_lag_1d']} mm | 7d Sum={p['rain_lag_7d_sum']} mm | Elevation={p['elevation']} m")
        print(f"  * Model Prediction : Class {pred_class} {'(Heavy Rain Event >=64.5mm)' if pred_class == 1 else '(Normal / Light Rain)'}")
        print(f"  * Prob(Heavy Rain) : {pred_prob_heavy:.4f} ({pred_prob_heavy * 100:.2f}%)")
        print(f"  * Advisory Status  : {risk_tier}")
        print(f"  * Target Benchmark : {sc['expected_outcome']}")
        print("-" * 85)


def main():
    model, feature_columns = load_model_and_metadata()
    test_metrics = evaluate_on_test_set(model, feature_columns)
    run_scenario_simulations(model, feature_columns)

    print("\n" + "=" * 85)
    print("  [OK] Model evaluation and scenario simulation completed successfully.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()

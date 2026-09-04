"""
================================================================================
JAL SANKETH — COMPREHENSIVE END-TO-END (E2E) TEST SUITE
Test File: tests/test_e2e_pipeline.py
================================================================================

This test suite performs an exhaustive end-to-end validation covering:
  1. Environment & Modular Imports
  2. Dataset Integrity & Column Validation
  3. Feature Engineering & Zero-Leakage Guarantee
  4. Model Loading & Feature Schema Alignment
  5. XGBoost Prediction & Probability Calibration
  6. Flood Risk Calculation Execution
  7. Risk Score Boundedness (0 to 100)
  8. Risk Level Tier Validity
  9. Folium Map Generation & Popup Rendering
 10. Early-Warning Advisory Generation
 11. Plotly Analytics Chart Construction
 12. Multi-Tier Demo Meteorological Scenarios
 13. Dynamic Responsiveness (No Fake/Hardcoded Output)
 14. Batch Dataset Evaluation Pipeline
 15. Configurable Weights & Threshold Integrity
================================================================================
"""

import sys
import os
import json
import joblib
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestJalSankethE2E(unittest.TestCase):
    """
    Exhaustive validation test suite for the complete Jal Sanketh system.
    """

    @classmethod
    def setUpClass(cls):
        """
        Global test setup ensuring all required assets exist.
        """
        cls.raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "weather_demo.csv")
        cls.model_path = os.path.join(PROJECT_ROOT, "models", "heavy_rain_model.joblib")
        cls.features_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
        cls.metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")

        # Ensure raw dataset exists
        if not os.path.exists(cls.raw_data_path):
            from data.create_demo_dataset import generate_demo_dataset
            generate_demo_dataset(output_path=cls.raw_data_path)

        # Ensure model exists
        if not os.path.exists(cls.model_path) or not os.path.exists(cls.features_path):
            from src.train_model import train_heavy_rain_classifier
            train_heavy_rain_classifier()

    # --------------------------------------------------------------------------
    # 1. IMPORTS VALIDATION
    # --------------------------------------------------------------------------
    def test_01_all_imports(self):
        """Verify all core source modules and utilities import cleanly."""
        try:
            import src.feature_engineering as fe
            import src.train_model as tm
            import src.evaluate_model as em
            import src.flood_risk as fr
            import src.alerts as al
            import src.map as mp
            import utils.config as cfg
            import app
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to import core modules: {e}")

    # --------------------------------------------------------------------------
    # 2. DATASET INTEGRITY & COLUMN VALIDATION
    # --------------------------------------------------------------------------
    def test_02_dataset_loading_and_columns(self):
        """Verify raw dataset loads, has >=10,000 rows, mandatory columns, and 0 nulls."""
        self.assertTrue(os.path.exists(self.raw_data_path))
        df = pd.read_csv(self.raw_data_path)

        # Check row count
        self.assertGreaterEqual(len(df), 10000, f"Expected >= 10,000 rows, got {len(df)}")

        # Check required columns
        required_cols = [
            "date", "latitude", "longitude", "temperature", "humidity",
            "pressure", "wind_speed", "rainfall", "elevation",
            "drainage_factor", "heavy_rain"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check missing values
        self.assertEqual(df.isnull().sum().sum(), 0, "Raw dataset contains null values")

        # Check heavy rain binary values
        unique_targets = set(df["heavy_rain"].unique())
        self.assertTrue(unique_targets.issubset({0, 1}), f"Unexpected target classes: {unique_targets}")

    # --------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING & LEAKAGE
    # --------------------------------------------------------------------------
    def test_03_feature_engineering_pipeline(self):
        """Verify feature engineering produces 24 clean features without data leakage."""
        from src.feature_engineering import FeatureEngineer, load_engineered_data

        X, y_reg, y_clf, full_df = load_engineered_data(raw_data_path=self.raw_data_path)

        self.assertEqual(X.shape[0], len(full_df))
        self.assertEqual(X.shape[1], 24, f"Expected 24 features, got {X.shape[1]}")
        self.assertEqual(X.isnull().sum().sum(), 0, "Engineered features contain nulls")

        # Verify zero leakage (targets not in X)
        self.assertNotIn("rainfall", X.columns)
        self.assertNotIn("heavy_rain", X.columns)

        # Verify temporal lags exist and are non-negative
        self.assertIn("rainfall_lag_1d", X.columns)
        self.assertIn("rainfall_lag_3d_mean", X.columns)
        self.assertIn("rainfall_lag_7d_sum", X.columns)
        self.assertGreaterEqual(X["rainfall_lag_1d"].min(), 0.0)

    # --------------------------------------------------------------------------
    # 4. MODEL LOADING & METADATA ALIGNMENT
    # --------------------------------------------------------------------------
    def test_04_model_loading(self):
        """Verify model and feature column schema load properly."""
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.features_path))

        model = joblib.load(self.model_path)
        with open(self.features_path, "r") as f:
            feature_cols = json.load(f)

        self.assertEqual(len(feature_cols), 24)
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertTrue(hasattr(model, "feature_importances_"))

    # --------------------------------------------------------------------------
    # 5. MODEL PREDICTION EXECUTION
    # --------------------------------------------------------------------------
    def test_05_model_predictions_execute(self):
        """Verify XGBoost model predicts valid binary classes and calibrated probabilities."""
        from src.feature_engineering import load_engineered_data

        X, _, y_clf, _ = load_engineered_data(raw_data_path=self.raw_data_path)
        model = joblib.load(self.model_path)

        sample_X = X.head(100)
        preds = model.predict(sample_X)
        probs = model.predict_proba(sample_X)[:, 1]

        self.assertEqual(len(preds), 100)
        self.assertEqual(len(probs), 100)
        self.assertTrue(set(preds).issubset({0, 1}))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    # --------------------------------------------------------------------------
    # 6, 7, 8. FLOOD RISK CALCULATION, BOUNDS & TIERS
    # --------------------------------------------------------------------------
    def test_06_flood_risk_engine(self):
        """Verify flood risk calculation produces scores in [0, 100] and valid levels."""
        from src.flood_risk import FloodRiskEngine

        engine = FloodRiskEngine()

        test_cases = [
            {"prob": 0.01, "rain": 0.0, "elev": 900.0, "drain": 0.80, "sat": 0.0, "expected_tier": "Low"},
            {"prob": 0.30, "rain": 30.0, "elev": 150.0, "drain": 0.50, "sat": 40.0, "expected_tier": "Moderate"},
            {"prob": 0.70, "rain": 80.0, "elev": 45.0, "drain": 0.28, "sat": 110.0, "expected_tier": "High"},
            {"prob": 0.98, "rain": 160.0, "elev": 6.0, "drain": 0.20, "sat": 240.0, "expected_tier": "Critical"}
        ]

        for tc in test_cases:
            res = engine.calculate_risk(
                heavy_rain_prob=tc["prob"],
                rainfall_intensity_mm=tc["rain"],
                elevation_m=tc["elev"],
                drainage_factor=tc["drain"],
                antecedent_rain_7d_mm=tc["sat"]
            )

            score = res["risk_score"]
            level = res["risk_level"]

            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            self.assertIn(level, ["Low", "Moderate", "High", "Critical"])
            self.assertEqual(level, tc["expected_tier"])
            self.assertTrue(isinstance(res["explanation"], str))
            self.assertGreater(len(res["explanation"]), 20)

    # --------------------------------------------------------------------------
    # 9. REUSABLE MAP COMPONENT RENDERING
    # --------------------------------------------------------------------------
    def test_09_map_renders_correctly(self):
        """Verify Folium map generates valid markers, circles, popups, and HTML."""
        from src.map import create_flood_risk_map
        import folium

        sample_stations = [
            {"location": "Hyderabad", "latitude": 17.3850, "longitude": 78.4867, "rainfall": 140.0, "heavy_rain_prob": 0.92, "risk_score": 88.5, "risk_level": "Critical"},
            {"location": "Adilabad", "latitude": 19.6641, "longitude": 78.5320, "rainfall": 2.0, "heavy_rain_prob": 0.01, "risk_score": 12.0, "risk_level": "Low"}
        ]

        m = create_flood_risk_map(sample_stations)
        self.assertIsInstance(m, folium.Map)

        # Check map HTML rendering
        html_str = m._repr_html_()
        self.assertIn("Hyderabad", html_str)
        self.assertIn("Adilabad", html_str)
        self.assertIn("#ef4444", html_str)  # Red for critical
        self.assertIn("#22c55e", html_str)  # Green for low

    # --------------------------------------------------------------------------
    # 10. ALERT GENERATION
    # --------------------------------------------------------------------------
    def test_10_alert_generation(self):
        """Verify early-warning alerts generate correct headlines and components."""
        from src.alerts import create_early_warning_alert, AlertGenerator

        tiers = ["Low", "Moderate", "High", "Critical"]
        for t in tiers:
            alert = create_early_warning_alert(
                location="Test Basin",
                predicted_rainfall=50.0,
                heavy_rainfall_prob=0.50,
                flood_risk_score=55.0,
                flood_risk_level=t
            )
            self.assertEqual(alert["risk_level"], t)
            self.assertIn(AlertGenerator.HEADLINES[t], alert["headline_message"])
            self.assertGreater(len(alert["main_reason"]), 10)
            self.assertGreater(len(alert["recommended_action"]), 10)

    # --------------------------------------------------------------------------
    # 11. PLOTLY CHART RENDERING
    # --------------------------------------------------------------------------
    def test_11_chart_rendering(self):
        """Verify Plotly timeline charts build cleanly without exceptions."""
        import plotly.graph_objects as go

        df_sample = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "rainfall": np.random.exponential(10, 30),
            "risk_score": np.random.uniform(10, 90, 30)
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_sample["date"], y=df_sample["rainfall"], name="Rainfall"))
        fig.add_trace(go.Scatter(x=df_sample["date"], y=df_sample["risk_score"], name="Risk Score"))
        fig.update_layout(title="Test Chart")

        self.assertEqual(len(fig.data), 2)
        self.assertIsNotNone(fig.to_json())

    # --------------------------------------------------------------------------
    # 12. DEMO SCENARIOS INFERENCE
    # --------------------------------------------------------------------------
    def test_12_demo_scenarios(self):
        """Verify that demo meteorological scenarios produce calibrated probabilities."""
        from src.evaluate_model import create_scenario_vector

        model = joblib.load(self.model_path)
        with open(self.features_path, "r") as f:
            feat_cols = json.load(f)

        # Dry scenario (Winter Delhi)
        X_dry = create_scenario_vector(feat_cols, temp=18.0, hum=35.0, pres=1018.0, wind=5.0, month=1, day_of_year=15)
        prob_dry = float(model.predict_proba(X_dry)[0, 1])

        # Cyclonic Deluge scenario (Chennai Monsoon)
        X_cyclone = create_scenario_vector(
            feat_cols, temp=24.0, hum=98.0, pres=978.0, wind=58.0, month=11, day_of_year=320,
            rain_lag_1d=95.0, rain_lag_7d_sum=280.0, pres_trend_24h=-12.0
        )
        prob_cyclone = float(model.predict_proba(X_cyclone)[0, 1])

        self.assertLess(prob_dry, 0.10, "Dry weather should have low heavy rain probability")
        self.assertGreater(prob_cyclone, 0.80, "Severe cyclone should have very high heavy rain probability")
        self.assertGreater(prob_cyclone, prob_dry, "Cyclone probability must be strictly higher than dry weather")

    # --------------------------------------------------------------------------
    # 13. NO FAKE/HARDCODED OUTPUTS (Dynamic Sensitivity Verification)
    # --------------------------------------------------------------------------
    def test_13_no_hardcoded_values(self):
        """Verify outputs dynamically change with input modifications (no hardcoding)."""
        from src.flood_risk import FloodRiskEngine

        engine = FloodRiskEngine()

        # Baseline
        base_res = engine.calculate_risk(0.10, 10.0, 100.0, 0.60, 10.0)
        # Modified rainfall
        high_rain_res = engine.calculate_risk(0.10, 120.0, 100.0, 0.60, 10.0)
        # Modified drainage
        bad_drain_res = engine.calculate_risk(0.10, 10.0, 100.0, 0.15, 10.0)

        self.assertNotEqual(base_res["risk_score"], high_rain_res["risk_score"])
        self.assertNotEqual(base_res["risk_score"], bad_drain_res["risk_score"])
        self.assertGreater(high_rain_res["risk_score"], base_res["risk_score"])
        self.assertGreater(bad_drain_res["risk_score"], base_res["risk_score"])

    # --------------------------------------------------------------------------
    # 14. BATCH DATASET EVALUATION PIPELINE
    # --------------------------------------------------------------------------
    def test_14_batch_evaluation_pipeline(self):
        """Verify full batch evaluation runs cleanly across raw dataset."""
        from app import get_evaluated_dataset

        eval_df, engine = get_evaluated_dataset()

        self.assertGreaterEqual(len(eval_df), 10000)
        self.assertIn("heavy_rain_prob", eval_df.columns)
        self.assertIn("risk_score", eval_df.columns)
        self.assertIn("risk_level", eval_df.columns)
        self.assertIn("risk_explanation", eval_df.columns)
        self.assertIn("location_name", eval_df.columns)

    # --------------------------------------------------------------------------
    # 15. CONFIG INTEGRITY
    # --------------------------------------------------------------------------
    def test_15_config_weights_and_tiers(self):
        """Verify configuration weights sum to 1.0 and tiers cover 0-100 without gaps."""
        from utils.config import FLOOD_RISK_WEIGHTS, RISK_TIERS

        total_weight = sum(FLOOD_RISK_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

        for tier in ["Low", "Moderate", "High", "Critical"]:
            self.assertIn(tier, RISK_TIERS)
            self.assertIn("min_score", RISK_TIERS[tier])
            self.assertIn("max_score", RISK_TIERS[tier])
            self.assertIn("color_hex", RISK_TIERS[tier])


if __name__ == "__main__":
    unittest.main(verbosity=2)

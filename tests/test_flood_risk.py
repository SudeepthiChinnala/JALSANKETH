"""
================================================================================
JAL SANKETH - FLOOD RISK ENGINE UNIT & SCENARIO TEST SUITE
Test File: tests/test_flood_risk.py
================================================================================
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.flood_risk import FloodRiskEngine
from utils.config import RISK_TIERS, FLOOD_RISK_WEIGHTS


class TestFloodRiskEngine(unittest.TestCase):
    """
    Unit and integration tests for the Jal Sanketh Flood Risk Engine.
    """

    def setUp(self):
        self.engine = FloodRiskEngine()

    def test_scenario_1_low_risk(self):
        """
        Scenario 1: Low rainfall + good drainage + high elevation
        Expected: Low risk level (score < 25), valid score range, complete explanation.
        """
        res = self.engine.calculate_risk(
            heavy_rain_prob=0.01,
            rainfall_intensity_mm=1.5,
            elevation_m=850.0,
            drainage_factor=0.85,
            antecedent_rain_7d_mm=4.0,
            location_name="Wayanad High Plateau (Dry Day)"
        )
        score = res["risk_score"]
        level = res["risk_level"]
        explanation = res["explanation"]

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(level, "Low")
        self.assertIn("LOW RISK", explanation)
        self.assertIn("Component Points Breakdown", explanation)
        self.assertIn("Wayanad High Plateau", explanation)
        self.assertEqual(res["color_hex"], "#28a745")

    def test_scenario_2_moderate_risk(self):
        """
        Scenario 2: Moderate rainfall + average drainage
        Expected: Moderate risk level (25 <= score < 50), valid score range.
        """
        res = self.engine.calculate_risk(
            heavy_rain_prob=0.25,
            rainfall_intensity_mm=38.0,
            elevation_m=110.0,
            drainage_factor=0.50,
            antecedent_rain_7d_mm=45.0,
            location_name="Pune Urban Basin (Moderate Shower)"
        )
        score = res["risk_score"]
        level = res["risk_level"]

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(level, "Moderate")
        self.assertGreaterEqual(score, 25.0)
        self.assertLess(score, 50.0)
        self.assertEqual(res["color_hex"], "#ffc107")
        self.assertTrue(len(res["explanation"]) > 50)

    def test_scenario_3_heavy_risk(self):
        """
        Scenario 3: Heavy rainfall + poor drainage
        Expected: High risk level (50 <= score < 75), valid score range.
        """
        res = self.engine.calculate_risk(
            heavy_rain_prob=0.72,
            rainfall_intensity_mm=85.0,
            elevation_m=45.0,
            drainage_factor=0.25,
            antecedent_rain_7d_mm=130.0,
            location_name="Patna Lowland (Monsoon Trough)"
        )
        score = res["risk_score"]
        level = res["risk_level"]

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(level, "High")
        self.assertGreaterEqual(score, 50.0)
        self.assertLess(score, 75.0)
        self.assertEqual(res["color_hex"], "#fd7e14")
        self.assertIn("SEVERE WARNING", res["explanation"])

    def test_scenario_4_extreme_risk(self):
        """
        Scenario 4: Extreme rainfall + poor drainage + low elevation
        Expected: Critical risk level (score >= 75), emergency alert advisory.
        """
        res = self.engine.calculate_risk(
            heavy_rain_prob=0.98,
            rainfall_intensity_mm=175.0,
            elevation_m=6.0,
            drainage_factor=0.20,
            antecedent_rain_7d_mm=260.0,
            location_name="Chennai Coastal Basin (Cyclonic Deluge)"
        )
        score = res["risk_score"]
        level = res["risk_level"]

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(level, "Critical")
        self.assertGreaterEqual(score, 75.0)
        self.assertEqual(res["color_hex"], "#dc3545")
        self.assertIn("EMERGENCY ALERT", res["explanation"])

    def test_monotonic_risk_progression(self):
        """
        Verifies that higher-risk hazard scenarios strictly produce progressively higher scores.
        """
        res1 = self.engine.calculate_risk(0.01, 2.0, 800.0, 0.85, 0.0)
        res2 = self.engine.calculate_risk(0.25, 35.0, 150.0, 0.50, 40.0)
        res3 = self.engine.calculate_risk(0.70, 80.0, 50.0, 0.28, 120.0)
        res4 = self.engine.calculate_risk(0.98, 170.0, 5.0, 0.15, 250.0)

        score1, score2, score3, score4 = res1["risk_score"], res2["risk_score"], res3["risk_score"], res4["risk_score"]

        self.assertLess(score1, score2, f"Scenario 1 ({score1}) should be < Scenario 2 ({score2})")
        self.assertLess(score2, score3, f"Scenario 2 ({score2}) should be < Scenario 3 ({score3})")
        self.assertLess(score3, score4, f"Scenario 3 ({score3}) should be < Scenario 4 ({score4})")

    def test_score_boundary_clamping(self):
        """
        Verifies that extreme out-of-range inputs are safely clamped within [0, 100].
        """
        # Minimum extreme (negative/zero)
        res_min = self.engine.calculate_risk(
            heavy_rain_prob=-0.5,
            rainfall_intensity_mm=-10.0,
            elevation_m=2000.0,
            drainage_factor=1.5,
            antecedent_rain_7d_mm=-5.0
        )
        self.assertGreaterEqual(res_min["risk_score"], 0.0)
        self.assertLessEqual(res_min["risk_score"], 100.0)

        # Maximum extreme
        res_max = self.engine.calculate_risk(
            heavy_rain_prob=2.0,
            rainfall_intensity_mm=1000.0,
            elevation_m=-50.0,
            drainage_factor=-0.5,
            antecedent_rain_7d_mm=2000.0
        )
        self.assertGreaterEqual(res_max["risk_score"], 0.0)
        self.assertLessEqual(res_max["risk_score"], 100.0)
        self.assertEqual(res_max["risk_level"], "Critical")

    def test_custom_weight_validation(self):
        """
        Verifies that custom weights summing to 1.0 work, and invalid weights raise ValueError.
        """
        valid_weights = {
            "heavy_rain_prob": 0.40,
            "rainfall_intensity": 0.20,
            "drainage_deficit": 0.20,
            "elevation_vulnerability": 0.10,
            "antecedent_saturation": 0.10
        }
        custom_engine = FloodRiskEngine(custom_weights=valid_weights)
        self.assertEqual(custom_engine.weights["heavy_rain_prob"], 0.40)

        # Invalid weights (sum != 1.0)
        invalid_weights = {
            "heavy_rain_prob": 0.50,
            "rainfall_intensity": 0.50,
            "drainage_deficit": 0.50,
            "elevation_vulnerability": 0.10,
            "antecedent_saturation": 0.10
        }
        with self.assertRaises(ValueError):
            FloodRiskEngine(custom_weights=invalid_weights)

    def test_batch_dataframe_evaluation(self):
        """
        Verifies batch evaluation across a pandas DataFrame.
        """
        df_sample = pd.DataFrame([
            {"location": "Station A", "heavy_rain_prob": 0.05, "rainfall": 5.0, "elevation": 500.0, "drainage_factor": 0.70, "rainfall_lag_7d_sum": 10.0},
            {"location": "Station B", "heavy_rain_prob": 0.45, "rainfall": 55.0, "elevation": 45.0, "drainage_factor": 0.35, "rainfall_lag_7d_sum": 80.0},
            {"location": "Station C", "heavy_rain_prob": 0.95, "rainfall": 160.0, "elevation": 8.0, "drainage_factor": 0.25, "rainfall_lag_7d_sum": 210.0}
        ])

        evaluated_df = self.engine.evaluate_dataframe(df_sample, name_col="location")

        self.assertIn("risk_score", evaluated_df.columns)
        self.assertIn("risk_level", evaluated_df.columns)
        self.assertIn("risk_color", evaluated_df.columns)
        self.assertIn("risk_explanation", evaluated_df.columns)
        self.assertEqual(len(evaluated_df), 3)

        # Check expected tier ordering
        self.assertLess(evaluated_df.loc[0, "risk_score"], evaluated_df.loc[1, "risk_score"])
        self.assertLess(evaluated_df.loc[1, "risk_score"], evaluated_df.loc[2, "risk_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

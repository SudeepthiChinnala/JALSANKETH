"""
================================================================================
JAL SANKETH - SCENARIO SIMULATOR UNIT TESTS
Test File: tests/test_simulator.py
================================================================================
"""

import sys
import os
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulator import ScenarioSimulator, SimulationResult


class TestScenarioSimulator(unittest.TestCase):
    """Unit tests for the interactive What-If scenario simulator."""

    @classmethod
    def setUpClass(cls):
        cls.sim = ScenarioSimulator()

    def test_dry_scenario_simulation(self):
        """Dry weather should result in low heavy-rain prob and Low risk tier."""
        res = self.sim.simulate(
            basin_name="Hyderabad Test Basin",
            simulated_rainfall_mm=0.0,
            elevation_m=500.0,
            drainage_factor=0.60,
            antecedent_rain_7d_mm=0.0,
            humidity_pct=30.0,
            temperature_c=25.0,
            pressure_hpa=1016.0,
            pressure_trend_24h=1.0,
            wind_speed_kmh=10.0
        )
        self.assertIsInstance(res, SimulationResult)
        self.assertLess(res.heavy_rain_prob, 0.20)
        self.assertEqual(res.risk_level, "Low")
        self.assertLess(res.risk_score, 25.0)
        self.assertFalse(res.is_critical)
        self.assertEqual(res.color_hex, "#28a745")

    def test_extreme_deluge_simulation(self):
        """Severe convective storm on low basin should trigger Critical risk."""
        res = self.sim.simulate(
            basin_name="Bhadrachalam Lowland",
            simulated_rainfall_mm=170.0,
            elevation_m=55.0,
            drainage_factor=0.18,
            antecedent_rain_7d_mm=220.0,
            humidity_pct=98.0,
            temperature_c=26.0,
            pressure_hpa=982.0,
            pressure_trend_24h=-14.0,
            wind_speed_kmh=55.0
        )
        self.assertGreater(res.heavy_rain_prob, 0.70)
        self.assertEqual(res.risk_level, "Critical")
        self.assertGreaterEqual(res.risk_score, 75.0)
        self.assertTrue(res.is_critical)
        self.assertEqual(res.color_hex, "#dc3545")

    def test_presets_exist(self):
        """Preset scenarios should return a non-empty list of dictionaries."""
        presets = self.sim.get_preset_scenarios()
        self.assertGreaterEqual(len(presets), 4)
        for p in presets:
            self.assertIn("name", p)
            self.assertIn("rainfall_mm", p)


if __name__ == "__main__":
    unittest.main()

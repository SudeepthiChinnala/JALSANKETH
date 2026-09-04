"""
================================================================================
JAL SANKETH - FORECAST LOADER UNIT TESTS
Test File: tests/test_forecast_loader.py
================================================================================
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.forecast_loader import (
    TELANGANA_STATIONS,
    load_raw_forecast_csv,
    aggregate_forecast_daily,
    process_forecast_for_telangana,
    resolve_forecast_csv_path
)
from src.map import is_in_telangana


class TestForecastLoader(unittest.TestCase):
    """
    Unit tests for Telangana forecast loader, aggregation, and basin synthesis.
    """

    def test_telangana_stations_integrity(self):
        """Verifies that all 20 stations exist and reside strictly inside Telangana bounds."""
        self.assertEqual(len(TELANGANA_STATIONS), 20)
        for st in TELANGANA_STATIONS:
            self.assertIn("name", st)
            self.assertIn("lat", st)
            self.assertIn("lon", st)
            self.assertIn("elev", st)
            self.assertIn("drain", st)
            # Must fall inside Telangana
            self.assertTrue(
                is_in_telangana(st["lat"], st["lon"]),
                f"Station {st['name']} at ({st['lat']}, {st['lon']}) is outside Telangana!"
            )
            # Physical validity
            self.assertGreater(st["elev"], 0.0)
            self.assertGreater(st["drain"], 0.0)
            self.assertLessEqual(st["drain"], 1.0)

    def test_resolve_forecast_path(self):
        """Verifies path resolution finds canonical data/weather_forecast.csv."""
        path = resolve_forecast_csv_path()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("weather_forecast.csv"))

    def test_load_raw_forecast_csv(self):
        """Verifies raw forecast CSV loads and formats columns properly."""
        df = load_raw_forecast_csv()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        for col in ["datetime", "date", "temperature", "humidity", "rainfall", "pressure", "wind_speed"]:
            self.assertIn(col, df.columns)

    def test_aggregate_forecast_daily(self):
        """Verifies daily aggregation from hourly forecast."""
        raw_df = load_raw_forecast_csv()
        daily = aggregate_forecast_daily(raw_df)
        self.assertIsInstance(daily, pd.DataFrame)
        self.assertGreater(len(daily), 0)
        self.assertIn("date", daily.columns)
        self.assertIn("rainfall", daily.columns)
        self.assertIn("temperature", daily.columns)

    def test_process_forecast_for_telangana(self):
        """Verifies that processing forecast synthesizes evaluations for all 20 stations."""
        eval_df, hourly_df = process_forecast_for_telangana()
        self.assertIsNotNone(eval_df)
        self.assertIsNotNone(hourly_df)
        # Should evaluate 20 unique stations across forecast horizon
        self.assertEqual(eval_df["location_name"].nunique(), 20)
        self.assertGreaterEqual(len(eval_df), 20)
        required_cols = [
            "location_name", "latitude", "longitude", "rainfall",
            "heavy_rain_prob", "risk_score", "risk_level", "risk_color"
        ]
        for col in required_cols:
            self.assertIn(col, eval_df.columns)

        # Risk score bounds
        for _, row in eval_df.iterrows():
            self.assertGreaterEqual(row["risk_score"], 0.0)
            self.assertLessEqual(row["risk_score"], 100.0)
            self.assertIn(row["risk_level"], ["Low", "Moderate", "High", "Critical"])


if __name__ == "__main__":
    unittest.main()

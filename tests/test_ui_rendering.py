"""
================================================================================
JAL SANKETH - STREAMLIT APPTEST COMPREHENSIVE UI INTEGRATION TESTS
Test File: tests/test_ui_rendering.py
================================================================================
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest


class TestStreamlitAppUI(unittest.TestCase):
    """
    End-to-end integration tests using Streamlit's AppTest framework.
    Simulates user navigation across all 7 tabs and verifies zero runtime exceptions.
    """

    @classmethod
    def setUpClass(cls):
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        cls.at = AppTest.from_file(app_path, default_timeout=30)
        cls.at.run()

    def test_01_app_initial_boot(self):
        """Verify the app boots and renders the initial Dashboard view without exceptions."""
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions on boot: {list(self.at.exception)}")
        self.assertEqual(self.at.session_state["nav_tab"], "Dashboard")

    def test_02_tab_dashboard(self):
        """Verify Dashboard tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Dashboard"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Dashboard: {list(self.at.exception)}")

    def test_03_tab_rainfall_prediction(self):
        """Verify Rainfall Prediction tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Rainfall Prediction"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Rainfall Prediction: {list(self.at.exception)}")

    def test_04_tab_flood_risk_map(self):
        """Verify Flood Risk Map tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Flood Risk Map"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Flood Risk Map: {list(self.at.exception)}")

    def test_05_tab_what_if_simulator(self):
        """Verify What-If Simulator tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "What-If Simulator"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in What-If Simulator: {list(self.at.exception)}")

    def test_06_tab_alerts(self):
        """Verify Alerts tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Alerts"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Alerts: {list(self.at.exception)}")

    def test_07_tab_reports(self):
        """Verify Reports tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Reports"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Reports: {list(self.at.exception)}")

    def test_08_tab_settings(self):
        """Verify Settings tab renders without exceptions."""
        self.at.session_state["nav_tab"] = "Settings"
        self.at.run()
        self.assertEqual(len(list(self.at.exception)), 0, f"Exceptions in Settings: {list(self.at.exception)}")


if __name__ == "__main__":
    unittest.main()

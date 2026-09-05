"""
================================================================================
JAL SANKETH - EVACUATION & SAFE PLACE UNIT TESTS
Test File: tests/test_evacuation.py
================================================================================
"""

import os
import sys
import unittest
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evacuation.safe_places import SafePlaceDatabase, find_nearest_safe_places
from evacuation.emergency import is_critical_risk, get_emergency_status
from src.location import is_in_telangana


class TestEvacuationModule(unittest.TestCase):
    """
    Unit tests for evacuation shelter routing, distance calculation, and emergency triggers.
    """

    @classmethod
    def setUpClass(cls):
        cls.db = SafePlaceDatabase()

    def test_database_loads_records(self):
        """Verifies safe place database loads the canonical CSV dataset."""
        self.assertIsNotNone(self.db.safe_places_df)
        self.assertGreaterEqual(len(self.db.safe_places_df), 50)
        required_cols = ["name", "latitude", "longitude", "type", "capacity", "elevation"]
        for col in required_cols:
            self.assertIn(col, self.db.safe_places_df.columns)

    def test_haversine_distance(self):
        """Verifies Haversine distance calculation between known landmarks."""
        # Hyderabad (17.3850, 78.4867) to Warangal (17.9689, 79.5941) ~130-135 km
        dist = self.db.calculate_distance(17.3850, 78.4867, 17.9689, 79.5941)
        self.assertGreater(dist, 125.0)
        self.assertLess(dist, 145.0)

        # Distance to self must be 0
        self_dist = self.db.calculate_distance(17.3850, 78.4867, 17.3850, 78.4867)
        self.assertAlmostEqual(self_dist, 0.0, places=4)

    def test_find_nearest_safe_places(self):
        """Verifies nearest shelters are returned sorted by ascending distance."""
        nearest = self.db.find_nearest_safe_places(17.3850, 78.4867, limit=5)
        self.assertEqual(len(nearest), 5)
        for i in range(len(nearest) - 1):
            self.assertLessEqual(nearest[i]["distance_km"], nearest[i + 1]["distance_km"])
            self.assertIn("name", nearest[i])
            self.assertIn("capacity", nearest[i])

    def test_convenience_function(self):
        """Verifies module-level find_nearest_safe_places convenience function."""
        nearest = find_nearest_safe_places(17.3850, 78.4867, limit=3)
        self.assertEqual(len(nearest), 3)

    def test_get_safe_place_by_name(self):
        """Verifies finding specific shelter by name query."""
        res = self.db.get_safe_place_by_name("Stadium")
        self.assertIsNotNone(res)
        self.assertIn("latitude", res)
        self.assertIn("capacity", res)

    def test_is_critical_risk(self):
        """Verifies critical risk detection for risk level string and numerical thresholds."""
        self.assertTrue(is_critical_risk("Critical", 50.0))
        self.assertTrue(is_critical_risk("critical", 10.0))
        self.assertTrue(is_critical_risk("CRITICAL", 0.0))
        self.assertTrue(is_critical_risk("High", 75.0))
        self.assertTrue(is_critical_risk("High", 82.5))
        self.assertFalse(is_critical_risk("High", 74.9))
        self.assertFalse(is_critical_risk("Moderate", 45.0))
        self.assertFalse(is_critical_risk("Low", 15.0))

    def test_get_emergency_status(self):
        """Verifies emergency status dictionary schema."""
        status_crit = get_emergency_status("Critical", 85.0)
        self.assertTrue(status_crit["is_emergency"])
        self.assertEqual(status_crit["severity"], "CRITICAL")
        self.assertTrue(status_crit["requires_evacuation"])

        status_norm = get_emergency_status("Low", 15.0)
        self.assertFalse(status_norm["is_emergency"])
        self.assertEqual(status_norm["severity"], "NORMAL")
        self.assertFalse(status_norm["requires_evacuation"])

    def test_telangana_boundary_validation(self):
        """Verifies coordinate bounding validation for Telangana districts."""
        # Inside Telangana
        self.assertTrue(is_in_telangana(17.3850, 78.4867))  # Hyderabad
        self.assertTrue(is_in_telangana(17.9689, 79.5941))  # Warangal
        self.assertTrue(is_in_telangana(19.6641, 78.5320))  # Adilabad
        self.assertTrue(is_in_telangana(17.2473, 80.1514))  # Khammam

        # Outside Telangana
        self.assertFalse(is_in_telangana(13.0827, 80.2707))  # Chennai
        self.assertFalse(is_in_telangana(19.0760, 72.8777))  # Mumbai
        self.assertFalse(is_in_telangana(28.6139, 77.2090))  # Delhi


if __name__ == "__main__":
    unittest.main()

"""
JAL SANKETH - Emergency Evacuation Module
Provides safe place database and evacuation routing for Telangana flood scenarios.
"""

from .safe_places import SafePlaceDatabase, find_nearest_safe_places
from .emergency import render_emergency_alert, is_critical_risk, get_emergency_status, reset_evacuation_status

__all__ = ['SafePlaceDatabase', 'find_nearest_safe_places', 'render_emergency_alert', 'is_critical_risk', 'get_emergency_status', 'reset_evacuation_status']

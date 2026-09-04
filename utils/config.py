"""
================================================================================
JAL SANKETH - SYSTEM CONFIGURATION & FLOOD RISK WEIGHTS
Module Path: utils/config.py
================================================================================

This file contains all configurable parameters, hydrological factor weights,
and risk classification thresholds for the Jal Sanketh Prototype.

Adjusting weights allows calibration for different urban topologies (e.g.,
increasing drainage weight in dense urban metros or elevation weight in deltas).
================================================================================
"""

from typing import Dict, Any

# ==============================================================================
# FLOOD RISK COMPOSITE WEIGHTS (Must sum to 1.00)
# ==============================================================================
FLOOD_RISK_WEIGHTS: Dict[str, float] = {
    # 1. Machine Learning Heavy Rain Probability (P(Rain >= 64.5mm) in [0, 1])
    "heavy_rain_prob": 0.35,

    # 2. Predicted / Observed 24-hr Rainfall Volume (mm normalized to 150mm deluge benchmark)
    "rainfall_intensity": 0.25,

    # 3. Urban Drainage Bottleneck Deficit (1.0 - drainage_factor)
    "drainage_deficit": 0.20,

    # 4. Low-Lying Elevation Vulnerability (Inverse normalized elevation)
    "elevation_vulnerability": 0.12,

    # 5. Catchment Antecedent Moisture / 7-Day Cumulative Rain Saturation
    "antecedent_saturation": 0.08
}

# Ensure weights sum to 1.0
assert abs(sum(FLOOD_RISK_WEIGHTS.values()) - 1.0) < 1e-6, "FLOOD_RISK_WEIGHTS must sum to exactly 1.0"


# ==============================================================================
# RISK LEVEL CLASSIFICATION TIERS (0 to 100 Scale)
# ==============================================================================
# 0-24   = Low
# 25-49  = Moderate
# 50-74  = High
# 75-100 = Critical
RISK_TIERS = {
    "Low": {
        "min_score": 0.0,
        "max_score": 24.99,
        "color_hex": "#28a745",
        "badge": "LOW",
        "description": "Normal seasonal conditions. Negligible pooling; routine drainage maintenance."
    },
    "Moderate": {
        "min_score": 25.0,
        "max_score": 49.99,
        "color_hex": "#ffc107",
        "badge": "MODERATE",
        "description": "Advisory alert. Localized ponding in low-lying subways and culverts."
    },
    "High": {
        "min_score": 50.0,
        "max_score": 74.99,
        "color_hex": "#fd7e14",
        "badge": "HIGH",
        "description": "Severe flood warning. Waterlogging across transit corridors; deploy dewatering pumps."
    },
    "Critical": {
        "min_score": 75.0,
        "max_score": 100.0,
        "color_hex": "#dc3545",
        "badge": "CRITICAL",
        "description": "EMERGENCY FLOOD ALERT. High inundation depth; activate SDRF/NDRF evacuation SOPs."
    }
}


# ==============================================================================
# METEOROLOGICAL & NORMALIZATION BENCHMARKS
# ==============================================================================
BENCHMARKS = {
    # Extreme 24-hr precipitation threshold used for 100% hazard normalization (mm)
    "severe_rainfall_benchmark_mm": 150.0,

    # Reference elevation ceiling above which pooling vulnerability reaches baseline minimum (m)
    "elevation_vulnerability_ceiling_m": 300.0,

    # 7-day cumulative rainfall threshold representing 100% antecedent catchment saturation (mm)
    "antecedent_saturation_benchmark_mm": 200.0
}


# ==============================================================================
# CARTO BASEMAP CONFIGURATION & ENVIRONMENT LOADER
# ==============================================================================
import os

def load_environment_variables():
    """
    Loads environment variables from .env using python-dotenv with manual fallback.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dotenv_path = os.path.join(project_root, ".env")
    
    if os.path.exists(dotenv_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=dotenv_path, override=False)
        except ImportError:
            try:
                with open(dotenv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

# Load .env on module import
load_environment_variables()

CARTO_API_KEY = os.environ.get("CARTO_API_KEY", "").strip()

# CARTO Basemap Dark Matter Tile URL Configuration
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
    '&copy; <a href="https://carto.com/attributions">CARTO</a>'
)

def get_carto_tile_config() -> Dict[str, Any]:
    """
    Returns the appropriate CARTO basemap tile URL and attribution.
    Uses the authenticated URL if CARTO_API_KEY is available;
    otherwise falls back to the standard CartoDB dark_matter tile.
    """
    api_key = os.environ.get("CARTO_API_KEY", "").strip()
    if api_key:
        return {
            "tiles": f"https://{{s}}.basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?api_key={api_key}",
            "attr": CARTO_ATTRIBUTION,
            "subdomains": "abcd",
            "name": "CARTO Dark Matter (Authenticated)",
            "is_custom_url": True
        }
    return {
        "tiles": "CartoDB dark_matter",
        "attr": CARTO_ATTRIBUTION,
        "subdomains": "abc",
        "name": "CartoDB dark_matter",
        "is_custom_url": False
    }


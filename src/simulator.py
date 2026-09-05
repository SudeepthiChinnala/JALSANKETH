"""
================================================================================
JAL SANKETH — INTERACTIVE "WHAT-IF" SCENARIO SIMULATION ENGINE
Module Path: src/simulator.py
================================================================================

This module powers the "What-If" scenario sandbox for the Jal Sanketh Early
Warning Platform. It enables disaster managers, urban hydrologists, and hackathon
evaluators to simulate arbitrary combinations of meteorological and terrain
stresses and observe real-time recalculation of heavy-rain probability, composite
flood risk scores, and standard operating procedures (SOPs).
================================================================================
"""

import sys
import os
import json
import joblib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluate_model import create_scenario_vector
from src.flood_risk import FloodRiskEngine
from src.alerts import AlertGenerator
from utils.config import FLOOD_RISK_WEIGHTS, RISK_TIERS


@dataclass
class SimulationResult:
    """Encapsulates output metrics and diagnostics from a What-If simulation run."""
    basin_name: str
    rainfall_mm: float
    heavy_rain_prob: float
    heavy_rain_prob_pct: float
    risk_score: float
    risk_level: str
    color_hex: str
    badge_icon: str
    component_breakdown: Dict[str, float]
    factor_percentages: Dict[str, float]
    explanation: str
    headline_message: str
    recommended_action: str
    is_critical: bool
    input_parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScenarioSimulator:
    """
    Interactive simulation engine combining trained XGBoost classifier inference
    with multi-criteria hydrological flood vulnerability evaluation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        self.model_path = model_path or os.path.join(PROJECT_ROOT, "models", "heavy_rain_model.joblib")
        self.features_path = feature_columns_path or os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
        self.weights = custom_weights or FLOOD_RISK_WEIGHTS.copy()

        # Load trained ML model
        if not os.path.exists(self.model_path) or not os.path.exists(self.features_path):
            from src.train_model import train_heavy_rain_classifier
            train_heavy_rain_classifier()

        self.model = joblib.load(self.model_path)
        with open(self.features_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        self.risk_engine = FloodRiskEngine(custom_weights=self.weights)

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Dynamically updates active MCDA flood risk weights."""
        self.weights = weights.copy()
        self.risk_engine = FloodRiskEngine(custom_weights=self.weights)

    def simulate(
        self,
        basin_name: str = "Hyderabad — Musi River Basin",
        simulated_rainfall_mm: float = 75.0,
        elevation_m: float = 505.0,
        drainage_factor: float = 0.38,
        antecedent_rain_7d_mm: float = 60.0,
        humidity_pct: float = 85.0,
        temperature_c: float = 26.0,
        pressure_hpa: float = 998.0,
        pressure_trend_24h: float = -6.0,
        wind_speed_kmh: float = 24.0,
        month: int = 8,
        day_of_year: int = 225
    ) -> SimulationResult:
        """
        Executes an end-to-end simulation:
        1. Synthesizes 24-feature vector with strict zero-leakage atmospheric physics.
        2. Computes XGBoost probability of extreme heavy precipitation.
        3. Evaluates composite flood risk score (0-100) via MCDA engine.
        4. Derives diagnostics and NDMA-aligned operational advisories.
        """
        sim_rain = max(0.0, float(simulated_rainfall_mm))
        elev = max(1.0, float(elevation_m))
        drain = float(np.clip(drainage_factor, 0.05, 1.0))
        sat_7d = max(0.0, float(antecedent_rain_7d_mm))
        hum = float(np.clip(humidity_pct, 10.0, 100.0))
        temp = float(np.clip(temperature_c, 5.0, 50.0))
        pres = float(np.clip(pressure_hpa, 900.0, 1050.0))
        pres_trend = float(np.clip(pressure_trend_24h, -40.0, 20.0))
        wind = max(0.0, float(wind_speed_kmh))
        m = int(np.clip(month, 1, 12))
        doy = int(np.clip(day_of_year, 1, 366))

        rain_lag_1d = min(sim_rain, sat_7d * 0.35)
        rain_lag_3d_mean = min(sim_rain, sat_7d * 0.20)

        # 1. Construct 24-feature input matrix
        X_vec = create_scenario_vector(
            feature_columns=self.feature_columns,
            temp=temp,
            hum=hum,
            pres=pres,
            wind=wind,
            month=m,
            day_of_year=doy,
            rain_lag_1d=rain_lag_1d,
            rain_lag_3d_mean=rain_lag_3d_mean,
            rain_lag_7d_sum=sat_7d,
            pres_trend_24h=pres_trend,
            temp_trend_24h=-1.0 if pres_trend < 0 else 0.5,
            elevation=elev,
            drainage_factor=drain
        )

        # 2. XGBoost Heavy Rain Probability
        prob_matrix = self.model.predict_proba(X_vec)
        heavy_rain_prob = float(prob_matrix[0, 1])

        # 3. MCDA Flood Risk Evaluation
        risk_res = self.risk_engine.calculate_risk(
            heavy_rain_prob=heavy_rain_prob,
            rainfall_intensity_mm=sim_rain,
            elevation_m=elev,
            drainage_factor=drain,
            antecedent_rain_7d_mm=sat_7d,
            location_name=basin_name
        )

        risk_score = risk_res["risk_score"]
        risk_level = risk_res["risk_level"]
        color_hex = risk_res["color_hex"]
        components = risk_res.get("components", {})
        breakdown = {k: v.get("points_contributed", 0.0) for k, v in components.items()}

        # Compute factor percentage contributions to total score
        total_pts = sum(breakdown.values())
        if total_pts > 0:
            factor_percentages = {k: round((v / total_pts) * 100.0, 1) for k, v in breakdown.items()}
        else:
            factor_percentages = {k: 0.0 for k in breakdown}

        headline = AlertGenerator.HEADLINES.get(risk_level, "Standard seasonal monitoring.")
        recommendation = AlertGenerator.derive_recommended_action(risk_level=risk_level)

        badge_icons = {
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🟠",
            "Critical": "🔴"
        }

        return SimulationResult(
            basin_name=basin_name,
            rainfall_mm=sim_rain,
            heavy_rain_prob=heavy_rain_prob,
            heavy_rain_prob_pct=round(heavy_rain_prob * 100.0, 1),
            risk_score=risk_score,
            risk_level=risk_level,
            color_hex=color_hex,
            badge_icon=badge_icons.get(risk_level, "⚪"),
            component_breakdown=breakdown,
            factor_percentages=factor_percentages,
            explanation=risk_res["explanation"],
            headline_message=headline,
            recommended_action=recommendation,
            is_critical=(risk_level == "Critical" or risk_score >= 75.0),
            input_parameters={
                "basin_name": basin_name,
                "rainfall_mm": sim_rain,
                "elevation_m": elev,
                "drainage_factor": drain,
                "antecedent_rain_7d_mm": sat_7d,
                "humidity_pct": hum,
                "temperature_c": temp,
                "pressure_hpa": pres,
                "pressure_trend_24h": pres_trend,
                "wind_speed_kmh": wind
            }
        )

    @staticmethod
    def get_preset_scenarios() -> List[Dict[str, Any]]:
        """Returns standard demonstrator scenarios for quick one-click evaluation."""
        return [
            {
                "name": "☀️ Clear Sky / Dry Regime",
                "description": "Stable anticyclonic condition with minimal moisture and free drainage.",
                "rainfall_mm": 0.0,
                "humidity_pct": 35.0,
                "pressure_hpa": 1018.0,
                "pressure_trend_24h": 0.5,
                "antecedent_rain_7d_mm": 2.0,
                "drainage_factor": 0.65
            },
            {
                "name": "🌦️ Moderate Monsoon Shower",
                "description": "Standard southwest monsoon precipitation with moderate soil moisture.",
                "rainfall_mm": 35.0,
                "humidity_pct": 78.0,
                "pressure_hpa": 1004.0,
                "pressure_trend_24h": -2.0,
                "antecedent_rain_7d_mm": 45.0,
                "drainage_factor": 0.45
            },
            {
                "name": "🌧️ Active Monsoon Trough / Heavy Influx",
                "description": "Persistent monsoon surge saturating catchments and stressing urban storm channels.",
                "rainfall_mm": 85.0,
                "humidity_pct": 92.0,
                "pressure_hpa": 996.0,
                "pressure_trend_24h": -6.5,
                "antecedent_rain_7d_mm": 130.0,
                "drainage_factor": 0.30
            },
            {
                "name": "🚨 Cyclonic Deluge / Catastrophic Flash Flood",
                "description": "Extreme convective precipitation (>150 mm) hitting low-lying basin with clogged drainage.",
                "rainfall_mm": 165.0,
                "humidity_pct": 98.0,
                "pressure_hpa": 982.0,
                "pressure_trend_24h": -14.0,
                "antecedent_rain_7d_mm": 220.0,
                "drainage_factor": 0.15
            },
            {
                "name": "🏔️ High-Elevation Drainage Test",
                "description": "Heavy rainfall in an elevated hilly plateau with rapid runoff gradient.",
                "rainfall_mm": 90.0,
                "humidity_pct": 88.0,
                "pressure_hpa": 990.0,
                "pressure_trend_24h": -4.0,
                "antecedent_rain_7d_mm": 50.0,
                "drainage_factor": 0.70
            }
        ]

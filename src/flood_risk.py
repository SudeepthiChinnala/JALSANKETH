"""
================================================================================
JAL SANKETH - FLOOD RISK SCORING & EXPLAINABILITY ENGINE
Module Path: src/flood_risk.py
================================================================================

DISCLAIMER & PROTOTYPE STATUS:
------------------------------
The Jal Sanketh Flood Risk Engine is an EXPLAINABLE DECISION-SUPPORT PROTOTYPE
developed for Smart India Hackathon 2026. It calculates a multi-factor
vulnerability index (0–100) combining machine learning heavy-rain predictions
with terrain and drainage deficit factors.

IT IS NOT A SCIENTIFICALLY VALIDATED HYDRODYNAMIC FLOOD MODEL (such as
HEC-RAS, MIKE FLOOD, or EPA SWMM). It is designed to demonstrate automated,
transparent multi-criteria risk tiering for disaster management prototypes.
================================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import FLOOD_RISK_WEIGHTS, RISK_TIERS, BENCHMARKS


class FloodRiskEngine:
    """
    Multi-Criteria Decision Analysis (MCDA) Flood Risk Engine for Jal Sanketh.
    """

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initializes the engine with configurable weights.
        """
        self.weights = custom_weights or FLOOD_RISK_WEIGHTS.copy()
        
        # Validate that weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-5:
            raise ValueError(f"Configured weights must sum to 1.0 (current sum: {weight_sum:.4f})")

        self.benchmarks = BENCHMARKS
        self.tiers = RISK_TIERS

    def get_risk_tier(self, score: float) -> Tuple[str, Dict[str, Any]]:
        """
        Categorizes a continuous 0-100 risk score into Low, Moderate, High, or Critical.
        
        Risk Tiers:
          - 0  to 24.99 : Low
          - 25 to 49.99 : Moderate
          - 50 to 74.99 : High
          - 75 to 100.0 : Critical
        """
        score_clamped = float(np.clip(score, 0.0, 100.0))
        
        if score_clamped < 25.0:
            return "Low", self.tiers["Low"]
        elif score_clamped < 50.0:
            return "Moderate", self.tiers["Moderate"]
        elif score_clamped < 75.0:
            return "High", self.tiers["High"]
        else:
            return "Critical", self.tiers["Critical"]

    def calculate_risk(
        self,
        heavy_rain_prob: float,
        rainfall_intensity_mm: float,
        elevation_m: float,
        drainage_factor: float,
        antecedent_rain_7d_mm: float = 0.0,
        location_name: str = "Monitored Zone"
    ) -> Dict[str, Any]:
        """
        Calculates normalized, explainable flood-risk score (0 to 100) and risk level.

        Parameters:
        -----------
        heavy_rain_prob : float
            XGBoost predicted probability of heavy rain >= 64.5 mm (range: 0.0 to 1.0).
        rainfall_intensity_mm : float
            Predicted or observed 24-hr rainfall volume in mm.
        elevation_m : float
            Station elevation above mean sea level in meters.
        drainage_factor : float
            Drainage quality rating (0.0 to 1.0, where 1.0 = free flow, 0.1 = severe blockage).
        antecedent_rain_7d_mm : float
            7-day cumulative antecedent rainfall in mm (soil moisture / catchment saturation proxy).
        location_name : str
            Identifier for human-readable reporting.

        Returns:
        --------
        Dict[str, Any]: Risk score, level, sub-component points, and explanation.
        """
        # ----------------------------------------------------------------------
        # 1. Normalize Sub-Component Vulnerability Factors (All on 0 to 100 scale)
        # ----------------------------------------------------------------------

        # Component 1: Heavy Rain Probability Score (0 to 100)
        c_prob = float(np.clip(heavy_rain_prob, 0.0, 1.0) * 100.0)

        # Component 2: Rainfall Intensity Hazard Score (0 to 100)
        # Scaled against severe 150mm deluge benchmark
        c_intensity = float(np.clip(rainfall_intensity_mm / self.benchmarks["severe_rainfall_benchmark_mm"], 0.0, 1.0) * 100.0)

        # Component 3: Urban Drainage Deficit Score (0 to 100)
        # High deficit (e.g. 1 - 0.25 = 75%) indicates severe bottleneck
        c_drainage = float(np.clip(1.0 - drainage_factor, 0.0, 1.0) * 100.0)

        # Component 4: Low-Lying Elevation Vulnerability Score (0 to 100)
        # Coastal & plain areas (< 10m) score ~97-100%; high plateaus (> 300m) score baseline minimum
        c_elevation = float(np.clip(1.0 - (elevation_m / self.benchmarks["elevation_vulnerability_ceiling_m"]), 0.05, 1.0) * 100.0)

        # Component 5: Antecedent Catchment Moisture / 7-Day Saturation Score (0 to 100)
        c_saturation = float(np.clip(antecedent_rain_7d_mm / self.benchmarks["antecedent_saturation_benchmark_mm"], 0.0, 1.0) * 100.0)

        # ----------------------------------------------------------------------
        # 2. Compute Weighted Composite Risk Score (0 to 100)
        # ----------------------------------------------------------------------
        # Weighted linear combination
        p_prob = self.weights["heavy_rain_prob"] * c_prob
        p_intensity = self.weights["rainfall_intensity"] * c_intensity
        p_drainage = self.weights["drainage_deficit"] * c_drainage
        p_elevation = self.weights["elevation_vulnerability"] * c_elevation
        p_saturation = self.weights["antecedent_saturation"] * c_saturation

        raw_score = p_prob + p_intensity + p_drainage + p_elevation + p_saturation
        risk_score = float(np.clip(raw_score, 0.0, 100.0))

        # ----------------------------------------------------------------------
        # 3. Categorize Risk Level
        # ----------------------------------------------------------------------
        risk_level, tier_info = self.get_risk_tier(risk_score)

        # Component contribution breakdown
        components = {
            "heavy_rain_prob": {
                "raw_value": round(heavy_rain_prob, 4),
                "normalized_score": round(c_prob, 2),
                "weight": self.weights["heavy_rain_prob"],
                "points_contributed": round(p_prob, 2)
            },
            "rainfall_intensity": {
                "raw_value": f"{rainfall_intensity_mm:.1f} mm",
                "normalized_score": round(c_intensity, 2),
                "weight": self.weights["rainfall_intensity"],
                "points_contributed": round(p_intensity, 2)
            },
            "drainage_deficit": {
                "raw_value": f"drainage_factor={drainage_factor:.2f}",
                "normalized_score": round(c_drainage, 2),
                "weight": self.weights["drainage_deficit"],
                "points_contributed": round(p_drainage, 2)
            },
            "elevation_vulnerability": {
                "raw_value": f"{elevation_m:.1f} m",
                "normalized_score": round(c_elevation, 2),
                "weight": self.weights["elevation_vulnerability"],
                "points_contributed": round(p_elevation, 2)
            },
            "antecedent_saturation": {
                "raw_value": f"{antecedent_rain_7d_mm:.1f} mm",
                "normalized_score": round(c_saturation, 2),
                "weight": self.weights["antecedent_saturation"],
                "points_contributed": round(p_saturation, 2)
            }
        }

        # ----------------------------------------------------------------------
        # 4. Generate Human-Readable Explanation
        # ----------------------------------------------------------------------
        explanation = self.explain_risk_score(
            location_name=location_name,
            risk_score=round(risk_score, 1),
            risk_level=risk_level,
            components=components
        )

        return {
            "location_name": location_name,
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "color_hex": tier_info["color_hex"],
            "badge": tier_info["badge"],
            "advisory_summary": tier_info["description"],
            "components": components,
            "explanation": explanation
        }

    def explain_risk_score(
        self,
        location_name: str,
        risk_score: float,
        risk_level: str,
        components: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generates an intuitive, stakeholder-friendly human explanation detailing
        the precise physical and atmospheric drivers behind the assigned score.
        """
        # Sort components by points contributed
        sorted_factors = sorted(
            components.items(),
            key=lambda x: x[1]["points_contributed"],
            reverse=True
        )

        top_factor_name, top_factor_data = sorted_factors[0]
        second_factor_name, second_factor_data = sorted_factors[1]

        def format_factor_desc(name: str, data: Dict[str, Any]) -> Tuple[str, str]:
            raw = data["raw_value"]
            pts = data["points_contributed"]
            if name == "heavy_rain_prob":
                prob_pct = float(raw) * 100 if isinstance(raw, (int, float)) else float(str(raw).replace("%", ""))
                return "ML Heavy Rain Probability", f"elevated heavy rain probability ({prob_pct:.1f}%) contributing {pts:.1f} pts"
            elif name == "rainfall_intensity":
                return "Rainfall Hazard Volume", f"rainfall volume ({raw}) contributing {pts:.1f} pts"
            elif name == "drainage_deficit":
                return "Urban Drainage Deficit", f"drainage bottleneck ({raw}) contributing {pts:.1f} pts"
            elif name == "elevation_vulnerability":
                return "Low-Lying Exposure", f"low elevation ({raw}) prone to pooling contributing {pts:.1f} pts"
            elif name == "antecedent_saturation":
                return "Antecedent Catchment Moisture", f"antecedent 7-day accumulation ({raw}) contributing {pts:.1f} pts"
            return name, f"{raw} contributing {pts:.1f} pts"

        primary_title, primary_desc = format_factor_desc(top_factor_name, top_factor_data)
        _, second_desc = format_factor_desc(second_factor_name, second_factor_data)

        # Construct explanation
        lines = [
            f"Location: '{location_name}' has been assigned a Flood Risk Score of {risk_score}/100 [{risk_level.upper()} RISK].",
            f"  * Primary Driver   : {primary_title} ({primary_desc}).",
            f"  * Secondary Driver : {second_desc}.",
            f"  * Component Points Breakdown :",
            f"      - Heavy Rain Prob (w={self.weights['heavy_rain_prob']}) : {components['heavy_rain_prob']['points_contributed']:>5.1f} pts (Raw: {components['heavy_rain_prob']['raw_value']*100:.1f}%)",
            f"      - Rain Volume     (w={self.weights['rainfall_intensity']}) : {components['rainfall_intensity']['points_contributed']:>5.1f} pts (Raw: {components['rainfall_intensity']['raw_value']})",
            f"      - Drainage Deficit(w={self.weights['drainage_deficit']}) : {components['drainage_deficit']['points_contributed']:>5.1f} pts (Raw: {components['drainage_deficit']['raw_value']})",
            f"      - Elevation Vuln  (w={self.weights['elevation_vulnerability']}) : {components['elevation_vulnerability']['points_contributed']:>5.1f} pts (Raw: {components['elevation_vulnerability']['raw_value']})",
            f"      - Catchment Sat   (w={self.weights['antecedent_saturation']}) : {components['antecedent_saturation']['points_contributed']:>5.1f} pts (Raw: {components['antecedent_saturation']['raw_value']})"
        ]

        if risk_level == "Critical":
            lines.append("  * Operational Recommendation : EMERGENCY ALERT - High probability of inundation. Deploy pumps and issue travel warnings.")
        elif risk_level == "High":
            lines.append("  * Operational Recommendation : SEVERE WARNING - Waterlogging expected in low-lying subways. Mobilize quick response crews.")
        elif risk_level == "Moderate":
            lines.append("  * Operational Recommendation : ADVISORY WATCH - Clear trash screens and monitor retention basins.")
        else:
            lines.append("  * Operational Recommendation : NORMAL - Routine monitoring; no active flood warning.")

        return "\n".join(lines)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        prob_col: str = "heavy_rain_prob",
        rain_col: str = "rainfall",
        elev_col: str = "elevation",
        drain_col: str = "drainage_factor",
        sat_col: str = "rainfall_lag_7d_sum",
        name_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Batch evaluates risk scores across an entire pandas DataFrame.
        """
        res_df = df.copy()

        scores = []
        levels = []
        colors = []
        badges = []
        explanations = []

        for _, row in res_df.iterrows():
            prob = float(row.get(prob_col, 0.0))
            rain = float(row.get(rain_col, 0.0))
            elev = float(row.get(elev_col, 50.0))
            drain = float(row.get(drain_col, 0.40))
            sat = float(row.get(sat_col, 0.0))
            loc_name = str(row.get(name_col, "Station")) if name_col else "Station"

            eval_res = self.calculate_risk(
                heavy_rain_prob=prob,
                rainfall_intensity_mm=rain,
                elevation_m=elev,
                drainage_factor=drain,
                antecedent_rain_7d_mm=sat,
                location_name=loc_name
            )

            scores.append(eval_res["risk_score"])
            levels.append(eval_res["risk_level"])
            colors.append(eval_res["color_hex"])
            badges.append(eval_res["badge"])
            explanations.append(eval_res["explanation"])

        res_df["risk_score"] = scores
        res_df["risk_level"] = levels
        res_df["risk_color"] = colors
        res_df["risk_badge"] = badges
        res_df["risk_explanation"] = explanations

        return res_df


if __name__ == "__main__":
    print("=" * 80)
    print("        JAL SANKETH - FLOOD RISK ENGINE TEST & EXPLANABILITY SUITE")
    print("=" * 80)

    engine = FloodRiskEngine()

    test_scenarios = [
        {
            "name": "Delhi Plain (Dry Season)",
            "prob": 0.01,
            "rain": 0.0,
            "elev": 216.0,
            "drain": 0.50,
            "sat": 0.0
        },
        {
            "name": "Bengaluru Outer Ring Road (Moderate Shower)",
            "prob": 0.18,
            "rain": 24.0,
            "elev": 920.0,
            "drain": 0.45,
            "sat": 35.0
        },
        {
            "name": "Patna Lowland / Ganga Floodplain (Monsoon Trough)",
            "prob": 0.65,
            "rain": 72.0,
            "elev": 53.0,
            "drain": 0.28,
            "sat": 110.0
        },
        {
            "name": "Chennai Velachery Flood Basin (Cyclonic Deluge)",
            "prob": 0.95,
            "rain": 145.0,
            "elev": 7.0,
            "drain": 0.30,
            "sat": 240.0
        }
    ]

    for sc in test_scenarios:
        result = engine.calculate_risk(
            heavy_rain_prob=sc["prob"],
            rainfall_intensity_mm=sc["rain"],
            elevation_m=sc["elev"],
            drainage_factor=sc["drain"],
            antecedent_rain_7d_mm=sc["sat"],
            location_name=sc["name"]
        )

        print(f"\n>> {result['location_name']}")
        print(f"   Score : {result['risk_score']} / 100")
        print(f"   Level : {result['risk_level']} ({result['badge']})")
        print(f"   Explanation :\n{result['explanation']}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("  [OK] Flood Risk Engine executed successfully with transparent explainability.")
    print("=" * 80 + "\n")

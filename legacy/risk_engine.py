"""
Hydrological Inundation & Flood Risk Engine for Jal Sanketh
Combines predicted meteorological rainfall with terrain and drainage factors
to compute a composite, normalized Flood Risk Score (0 - 100).
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


class FloodRiskEngine:
    """
    Multi-Criteria Decision Analysis (MCDA) flood risk calculation engine.
    """

    # Normalized factor weighting configuration
    WEIGHT_RAINFALL = 0.40
    WEIGHT_ELEVATION = 0.18
    WEIGHT_DRAINAGE = 0.20
    WEIGHT_SOIL_SATURATION = 0.12
    WEIGHT_SLOPE = 0.10

    # Risk thresholds
    THRESH_LOW = 35.0
    THRESH_MODERATE = 60.0
    THRESH_HIGH = 80.0

    @classmethod
    def calculate_location_risk(
        cls,
        predicted_rainfall_mm: float,
        elevation_m: float,
        drainage_capacity_index: float,
        slope_deg: float,
        soil_saturation: float,
        proximity_to_waterbody_km: float = 2.0
    ) -> Dict[str, Any]:
        """
        Calculates normalized risk score and categorizes inundation hazard level.
        
        Args:
            predicted_rainfall_mm: Predicted 24-hr rainfall in mm.
            elevation_m: Elevation in meters above sea level.
            drainage_capacity_index: Urban drainage capacity (0.0 to 1.0, higher is better).
            slope_deg: Surface terrain slope in degrees.
            soil_saturation: Antecedent soil moisture index (0.0 to 1.0).
            proximity_to_waterbody_km: Distance to nearest major river/coast in km.
            
        Returns:
            Dictionary containing overall score, risk tier, color codes, and component breakdowns.
        """
        # 1. Rainfall Hazard Factor (0.0 to 1.0)
        # Scaled against a high intensity threshold (150 mm/day represents severe deluge)
        rainfall_hazard = float(np.clip(predicted_rainfall_mm / 150.0, 0.0, 1.0))
        
        # 2. Elevation Vulnerability Factor (0.0 to 1.0)
        # Low coastal/delta plains (< 10m) have high vulnerability; high elevations (> 300m) have low pooling risk
        elevation_vuln = float(np.clip(1.0 - (elevation_m / 250.0), 0.05, 1.0))
        
        # 3. Drainage Deficit Factor (0.0 to 1.0)
        # Silted, clogged, or insufficient drains yield high deficit
        drainage_deficit = float(np.clip(1.0 - drainage_capacity_index, 0.0, 1.0))
        
        # 4. Terrain Slope / Water Retention Factor (0.0 to 1.0)
        # Flat terrains (< 1 deg) suffer water stagnation
        slope_factor = float(np.clip(1.0 - (slope_deg / 6.0), 0.05, 1.0))
        
        # 5. Soil Saturation Factor (0.0 to 1.0)
        soil_factor = float(np.clip(soil_saturation, 0.0, 1.0))
        
        # Proximity adjustment factor
        waterbody_factor = float(np.clip(1.0 - (proximity_to_waterbody_km / 5.0), 0.0, 0.5))

        # 6. Base Raw Weighted Composite Score (0.0 to 1.0)
        terrain_vulnerability = (
            (cls.WEIGHT_ELEVATION * elevation_vuln) +
            (cls.WEIGHT_DRAINAGE * drainage_deficit) +
            (cls.WEIGHT_SLOPE * slope_factor) +
            (cls.WEIGHT_SOIL_SATURATION * soil_factor)
        ) # sum of non-rain weights is 0.60
        
        # Hydro-hazard interaction: terrain vulnerability creates inundation when driven by rainfall
        # Low rainfall (< 10mm) dampens static terrain vulnerability
        rainfall_multiplier = float(np.clip(predicted_rainfall_mm / 25.0, 0.05, 1.0)) if predicted_rainfall_mm < 25.0 else 1.0
        
        composite_score_raw = (cls.WEIGHT_RAINFALL * rainfall_hazard) + (terrain_vulnerability * rainfall_multiplier) + (waterbody_factor * 0.1 * rainfall_multiplier)
        
        # Scale to 0 - 100
        risk_score = float(np.clip(composite_score_raw * 100.0, 0.0, 100.0))
        
        # 7. Classification and Metadata
        if risk_score <= cls.THRESH_LOW:
            category = "Low"
            color_hex = "#28a745"  # Green
            badge_icon = "🟢"
            action_summary = "Normal conditions. Routine maintenance of roadside drains."
        elif risk_score <= cls.THRESH_MODERATE:
            category = "Moderate"
            color_hex = "#ffc107"  # Yellow
            badge_icon = "🟡"
            action_summary = "Advisory watch. Clear stormwater culverts in low-lying pockets."
        elif risk_score <= cls.THRESH_HIGH:
            category = "High"
            color_hex = "#fd7e14"  # Orange
            badge_icon = "🟠"
            action_summary = "Severe warning. Localized street waterlogging likely. Mobilize high-capacity dewatering pumps."
        else:
            category = "Critical"
            color_hex = "#dc3545"  # Red
            badge_icon = "🔴"
            action_summary = "EMERGENCY ALERT! High probability of widespread inundation. Issue public travel advisory and trigger flood SOPs."

        # Derived Inundation depth estimate (cm)
        if risk_score > cls.THRESH_LOW and predicted_rainfall_mm > 15.0:
            est_depth_cm = round((risk_score / 100.0) * (predicted_rainfall_mm * 0.35) * (1.0 + drainage_deficit), 1)
        else:
            est_depth_cm = 0.0

        return {
            "risk_score": round(risk_score, 1),
            "risk_category": category,
            "color_hex": color_hex,
            "badge_icon": badge_icon,
            "action_summary": action_summary,
            "estimated_inundation_depth_cm": est_depth_cm,
            "sub_factors": {
                "rainfall_hazard_pct": round(rainfall_hazard * 100, 1),
                "elevation_vuln_pct": round(elevation_vuln * 100, 1),
                "drainage_deficit_pct": round(drainage_deficit * 100, 1),
                "slope_flatness_pct": round(slope_factor * 100, 1),
                "soil_saturation_pct": round(soil_factor * 100, 1)
            }
        }

    @classmethod
    def evaluate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates risk scores across an entire DataFrame containing rainfall and geo features.
        """
        res_df = df.copy()
        
        scores = []
        categories = []
        colors = []
        actions = []
        depths = []
        
        for _, row in res_df.iterrows():
            rain = row.get("predicted_rainfall_mm", row.get("rainfall_amount_mm", 0.0))
            elev = row.get("elevation_m", 50.0)
            drain = row.get("drainage_capacity_index", 0.5)
            slope = row.get("slope_deg", 1.0)
            soil = row.get("soil_saturation_baseline", 0.6)
            water = row.get("proximity_to_waterbody_km", 2.0)
            
            res = cls.calculate_location_risk(
                predicted_rainfall_mm=rain,
                elevation_m=elev,
                drainage_capacity_index=drain,
                slope_deg=slope,
                soil_saturation=soil,
                proximity_to_waterbody_km=water
            )
            
            scores.append(res["risk_score"])
            categories.append(res["risk_category"])
            colors.append(res["color_hex"])
            actions.append(res["action_summary"])
            depths.append(res["estimated_inundation_depth_cm"])
            
        res_df["risk_score"] = scores
        res_df["risk_category"] = categories
        res_df["risk_color"] = colors
        res_df["action_advisory"] = actions
        res_df["inundation_depth_cm"] = depths
        
        return res_df


if __name__ == "__main__":
    # Test calculation
    sample_risk = FloodRiskEngine.calculate_location_risk(
        predicted_rainfall_mm=95.0,
        elevation_m=6.5,
        drainage_capacity_index=0.35,
        slope_deg=0.8,
        soil_saturation=0.75,
        proximity_to_waterbody_km=1.2
    )
    print("Sample Risk Assessment:", sample_risk)

"""
================================================================================
JAL SANKETH - EARLY-WARNING & ADVISORY GENERATION MODULE
Module Path: src/alerts.py
================================================================================

This module generates structured, explainable early-warning bulletins and
dashboard alert cards based on hydro-meteorological risk assessments.

Alert Tiers & Core Messages:
----------------------------
  • Low      : "No immediate flood warning."
  • Moderate : "Moderate rainfall/flood risk. Continue monitoring."
  • High     : "High flood risk. Authorities and residents should prepare precautionary measures."
  • Critical : "Critical flood risk. Immediate precautionary action is recommended."

Each alert contains:
  1. Risk Level
  2. Location
  3. Main Reason (Derivation from rainfall, probability & terrain)
  4. Recommended Action (Prudent, non-alarmist operational guidance)
================================================================================
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import RISK_TIERS


class AlertGenerator:
    """
    Early-warning advisory generator for the Jal Sanketh Prototype.
    """

    # Core required alert headline messages
    HEADLINES = {
        "Low": "No immediate flood warning.",
        "Moderate": "Moderate rainfall/flood risk. Continue monitoring.",
        "High": "High flood risk. Authorities and residents should prepare precautionary measures.",
        "Critical": "Critical flood risk. Immediate precautionary action is recommended."
    }

    # Visual indicators & colors
    ALERT_STYLES = {
        "Low": {"color_hex": "#28a745", "badge_icon": "[LOW]", "severity": 1},
        "Moderate": {"color_hex": "#ffc107", "badge_icon": "[MODERATE]", "severity": 2},
        "High": {"color_hex": "#fd7e14", "badge_icon": "[HIGH]", "severity": 3},
        "Critical": {"color_hex": "#dc3545", "badge_icon": "[CRITICAL]", "severity": 4}
    }

    @staticmethod
    def derive_main_reason(
        risk_level: str,
        predicted_rainfall: float,
        heavy_rainfall_prob: float,
        flood_risk_score: float,
        drainage_factor: Optional[float] = None,
        elevation_m: Optional[float] = None
    ) -> str:
        """
        Synthesizes a transparent, evidence-based reason explaining why the alert was generated.
        """
        prob_pct = heavy_rainfall_prob * 100.0 if heavy_rainfall_prob <= 1.0 else heavy_rainfall_prob
        
        terrain_context = ""
        if drainage_factor is not None and drainage_factor < 0.35:
            terrain_context = f" compounded by constrained urban drainage (factor {drainage_factor:.2f})"
        if elevation_m is not None and elevation_m < 15.0:
            terrain_context += f" and low-lying coastal/basin topography ({elevation_m:.1f} m elevation)"

        if risk_level == "Critical":
            return (
                f"Severe precipitation hazard detected with {prob_pct:.1f}% probability of extreme heavy rainfall "
                f"(predicted {predicted_rainfall:.1f} mm/24h), generating a composite flood risk score of {flood_risk_score:.1f}/100"
                f"{terrain_context}."
            )
        elif risk_level == "High":
            return (
                f"Elevated probability ({prob_pct:.1f}%) of heavy rainfall ({predicted_rainfall:.1f} mm/24h) "
                f"resulting in a heightened flood risk score of {flood_risk_score:.1f}/100"
                f"{terrain_context}."
            )
        elif risk_level == "Moderate":
            return (
                f"Moderate precipitation ({predicted_rainfall:.1f} mm/24h) and {prob_pct:.1f}% heavy-rain probability "
                f"indicate localized water accumulation potential (risk score: {flood_risk_score:.1f}/100)"
                f"{terrain_context}."
            )
        else:  # Low
            return (
                f"Precipitation is minimal ({predicted_rainfall:.1f} mm/24h) with low heavy-rain probability ({prob_pct:.1f}%); "
                f"hydro-meteorological parameters remain within safe baseline limits (risk score: {flood_risk_score:.1f}/100)."
            )

    @staticmethod
    def derive_recommended_action(risk_level: str) -> str:
        """
        Provides prudent, non-alarmist operational actions tailored for dashboard prototypes.
        """
        if risk_level == "Critical":
            return (
                "Civic authorities should mobilize high-capacity dewatering pumps to identified waterlogging hotspots, "
                "inspect tidal/sluice gates, and place quick-response teams on standby. "
                "Residents in low-lying or chronic flood pockets are advised to avoid submerged roads and secure ground-level assets."
            )
        elif risk_level == "High":
            return (
                "Municipal teams should clear stormwater grates, monitor vulnerable underpasses, and prepare backup pump units. "
                "Commuters should check route conditions before travel and avoid flood-prone transit corridors."
            )
        elif risk_level == "Moderate":
            return (
                "Continue routine monitoring of weather radar and rain gauge streams. "
                "Inspect trash screens at major culverts to prevent sudden drainage blockages."
            )
        else:  # Low
            return (
                "Maintain standard municipal monitoring. Conduct routine scheduled maintenance of roadside drainage networks."
            )

    @classmethod
    def generate_alert(
        cls,
        location: str,
        predicted_rainfall: float,
        heavy_rainfall_prob: float,
        flood_risk_score: float,
        flood_risk_level: str,
        drainage_factor: Optional[float] = None,
        elevation_m: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a structured early-warning alert record.

        Parameters:
        -----------
        location : str
            Name of monitored station / urban zone.
        predicted_rainfall : float
            Predicted or observed 24-hr rainfall volume (mm).
        heavy_rainfall_prob : float
            Predicted probability of heavy rainfall >= 64.5 mm (0.0 to 1.0).
        flood_risk_score : float
            Composite flood risk index (0 to 100).
        flood_risk_level : str
            Risk category: 'Low', 'Moderate', 'High', or 'Critical'.
        drainage_factor : Optional[float]
            Drainage quality factor (0.0 to 1.0).
        elevation_m : Optional[float]
            Elevation in meters.
        timestamp : Optional[str]
            ISO / formatted timestamp (defaults to current system time).

        Returns:
        --------
        Dict[str, Any]: Structured alert dictionary.
        """
        # Validate risk level
        level_normalized = flood_risk_level.capitalize()
        if level_normalized not in cls.HEADLINES:
            # Fallback based on score if level name is non-standard
            if flood_risk_score >= 75.0:
                level_normalized = "Critical"
            elif flood_risk_score >= 50.0:
                level_normalized = "High"
            elif flood_risk_score >= 25.0:
                level_normalized = "Moderate"
            else:
                level_normalized = "Low"

        # Timestamp
        time_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        
        # Core components
        headline = cls.HEADLINES[level_normalized]
        style = cls.ALERT_STYLES[level_normalized]
        
        main_reason = cls.derive_main_reason(
            risk_level=level_normalized,
            predicted_rainfall=predicted_rainfall,
            heavy_rainfall_prob=heavy_rainfall_prob,
            flood_risk_score=flood_risk_score,
            drainage_factor=drainage_factor,
            elevation_m=elevation_m
        )

        recommended_action = cls.derive_recommended_action(level_normalized)

        prob_pct = heavy_rainfall_prob * 100.0 if heavy_rainfall_prob <= 1.0 else heavy_rainfall_prob

        return {
            "alert_id": f"JS-ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": time_str,
            "location": location,
            "risk_level": level_normalized,
            "risk_score": round(flood_risk_score, 1),
            "headline_message": headline,
            "main_reason": main_reason,
            "recommended_action": recommended_action,
            "color_hex": style["color_hex"],
            "badge_icon": style["badge_icon"],
            "severity_rank": style["severity"],
            "parameters": {
                "predicted_rainfall_mm": round(predicted_rainfall, 1),
                "heavy_rain_probability_pct": round(prob_pct, 1),
                "drainage_factor": drainage_factor,
                "elevation_m": elevation_m
            }
        }

    @classmethod
    def format_alert_card(cls, alert: Dict[str, Any]) -> str:
        """
        Formats an alert dictionary into a clean text card suitable for terminal/dashboard rendering.
        """
        lines = [
            "=" * 75,
            f"  JAL SANKETH EARLY WARNING ALERT : {alert['badge_icon']} {alert['risk_level'].upper()}",
            "=" * 75,
            f"  * Location           : {alert['location']}",
            f"  * Risk Score         : {alert['risk_score']} / 100  [{alert['risk_level']} Risk]",
            f"  * Status Headline    : {alert['headline_message']}",
            f"  * Predicted Rainfall : {alert['parameters']['predicted_rainfall_mm']} mm/24h (P(Heavy Rain): {alert['parameters']['heavy_rain_probability_pct']}%)",
            "-" * 75,
            f"  [MAIN REASON]",
            f"  {alert['main_reason']}",
            "",
            f"  [RECOMMENDED ACTION]",
            f"  {alert['recommended_action']}",
            "=" * 75
        ]
        return "\n".join(lines)

    @classmethod
    def generate_alerts_dataframe(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates alerts for all rows in a DataFrame.
        """
        alerts = []
        for _, row in df.iterrows():
            loc = row.get("location_name", row.get("location", "Monitored Zone"))
            rain = float(row.get("rainfall", row.get("predicted_rainfall_mm", 0.0)))
            prob = float(row.get("heavy_rain_prob", 0.0))
            score = float(row.get("risk_score", 0.0))
            level = str(row.get("risk_level", "Low"))
            drain = row.get("drainage_factor", None)
            elev = row.get("elevation", None)

            alert = cls.generate_alert(
                location=loc,
                predicted_rainfall=rain,
                heavy_rainfall_prob=prob,
                flood_risk_score=score,
                flood_risk_level=level,
                drainage_factor=float(drain) if drain is not None else None,
                elevation_m=float(elev) if elev is not None else None
            )
            alerts.append(alert)
        return alerts


# Convenience alias function
def create_early_warning_alert(
    location: str,
    predicted_rainfall: float,
    heavy_rainfall_prob: float,
    flood_risk_score: float,
    flood_risk_level: str,
    drainage_factor: Optional[float] = None,
    elevation_m: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a single Jal Sanketh early-warning alert.
    """
    return AlertGenerator.generate_alert(
        location=location,
        predicted_rainfall=predicted_rainfall,
        heavy_rainfall_prob=heavy_rainfall_prob,
        flood_risk_score=flood_risk_score,
        flood_risk_level=flood_risk_level,
        drainage_factor=drainage_factor,
        elevation_m=elevation_m
    )


if __name__ == "__main__":
    print("=" * 80)
    print("      JAL SANKETH - EARLY-WARNING GENERATION MODULE TEST SUITE")
    print("=" * 80)

    test_cases = [
        {
            "location": "Delhi NCR - Kashmere Gate Plain",
            "predicted_rainfall": 2.5,
            "heavy_rainfall_prob": 0.01,
            "flood_risk_score": 14.2,
            "flood_risk_level": "Low",
            "drainage_factor": 0.48,
            "elevation_m": 216.0
        },
        {
            "location": "Bengaluru - Bellandur Outer Ring Road",
            "predicted_rainfall": 32.0,
            "heavy_rainfall_prob": 0.28,
            "flood_risk_score": 38.5,
            "flood_risk_level": "Moderate",
            "drainage_factor": 0.42,
            "elevation_m": 920.0
        },
        {
            "location": "Patna - Rajendra Nagar Ganga Basin",
            "predicted_rainfall": 78.0,
            "heavy_rainfall_prob": 0.72,
            "flood_risk_score": 68.4,
            "flood_risk_level": "High",
            "drainage_factor": 0.28,
            "elevation_m": 53.0
        },
        {
            "location": "Mumbai - Dadar / Hindmata Lowland",
            "predicted_rainfall": 155.0,
            "heavy_rainfall_prob": 0.96,
            "flood_risk_score": 93.8,
            "flood_risk_level": "Critical",
            "drainage_factor": 0.32,
            "elevation_m": 6.5
        }
    ]

    for tc in test_cases:
        alert = create_early_warning_alert(
            location=tc["location"],
            predicted_rainfall=tc["predicted_rainfall"],
            heavy_rainfall_prob=tc["heavy_rainfall_prob"],
            flood_risk_score=tc["flood_risk_score"],
            flood_risk_level=tc["flood_risk_level"],
            drainage_factor=tc["drainage_factor"],
            elevation_m=tc["elevation_m"]
        )
        print(AlertGenerator.format_alert_card(alert))
        print()

    print("=" * 80)
    print("  [OK] Alert generation module executed successfully.")
    print("=" * 80 + "\n")

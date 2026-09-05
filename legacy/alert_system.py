"""
Early Warning Bulletin & Standard Operating Procedure (SOP) Alert Generator for Jal Sanketh
"""

from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class AlertSystem:
    """
    Automated generation of municipal and citizen emergency advisories.
    """

    @staticmethod
    def get_sop_guidelines(risk_category: str) -> Dict[str, List[str]]:
        """
        Returns actionable Standard Operating Procedures based on risk level.
        """
        if risk_category == "Critical":
            return {
                "Municipal & Civic Authorities": [
                    "Activate 24x7 Emergency Operations Center (EOC) at full readiness.",
                    "Deploy high-capacity diesel dewatering pumps to pre-identified underpasses and low-lying junctions.",
                    "Inspect and operate tidal/sluice gates in coastal and riverine channels.",
                    "Pre-position heavy earthmoving equipment to clear sudden drain blockages."
                ],
                "Traffic & Disaster Response (SDRF/NDRF)": [
                    "Cordon off submerged subways, underpasses, and arterial waterlogged corridors.",
                    "Issue immediate traffic diversions via radio, SMS, and digital signage.",
                    "Station quick-response rescue inflatable boats and medical teams in vulnerable flood pockets.",
                    "Prepare temporary relief and evacuation shelters with clean drinking water and food provisions."
                ],
                "Citizens & Public Advisory": [
                    "Avoid unnecessary travel; stay indoors unless residing in low-lying flood-prone structures.",
                    "Disconnect electrical appliances from ground outlets if water begins entering premises.",
                    "Do not drive or walk through moving floodwater or submerged roads ('Turn Around, Don't Drown').",
                    "Keep emergency helpline numbers (112, 1077) on speed dial and phones fully charged."
                ]
            }
        elif risk_category == "High":
            return {
                "Municipal & Civic Authorities": [
                    "Place rapid response teams on standby in designated flood zones.",
                    "Clear primary stormwater inlets and remove floating solid waste from trash screens.",
                    "Monitor retention basins and lake discharge levels continuously."
                ],
                "Traffic & Disaster Response (SDRF/NDRF)": [
                    "Identify choke points and place barricades near known chronic waterlogging spots.",
                    "Issue yellow/orange level travel advisories for peak commute hours."
                ],
                "Citizens & Public Advisory": [
                    "Plan travel routes carefully and avoid flood-prone underpasses.",
                    "Park vehicles on elevated ground away from open drains or low basements.",
                    "Ensure adequate backup drinking water and essential medicines."
                ]
            }
        elif risk_category == "Moderate":
            return {
                "Municipal & Civic Authorities": [
                    "Desilt roadside stormwater drains and check pump station fuel reserves.",
                    "Monitor weather radar updates and local rainfall sensor streams."
                ],
                "Traffic & Disaster Response (SDRF/NDRF)": [
                    "Maintain situational awareness and test emergency communication channels."
                ],
                "Citizens & Public Advisory": [
                    "Be aware of localized puddling during sudden intense downpours."
                ]
            }
        else:
            return {
                "Municipal & Civic Authorities": [
                    "Maintain routine monitoring and scheduled drain clearance schedules."
                ],
                "Traffic & Disaster Response (SDRF/NDRF)": [
                    "Routine operations."
                ],
                "Citizens & Public Advisory": [
                    "Normal seasonal conditions. No immediate precautions required."
                ]
            }

    @classmethod
    def generate_bulletin_text(cls, location_row: pd.Series) -> str:
        """
        Formats an official disaster advisory bulletin text.
        """
        loc_name = location_row.get("location_name", "Monitored Zone")
        city = location_row.get("city", "N/A")
        state = location_row.get("state", "N/A")
        risk_cat = location_row.get("risk_category", "Moderate")
        score = location_row.get("risk_score", 0.0)
        rain_mm = location_row.get("predicted_rainfall_mm", location_row.get("rainfall_amount_mm", 0.0))
        depth_cm = location_row.get("inundation_depth_cm", 0.0)
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        
        sops = cls.get_sop_guidelines(risk_cat)
        
        bulletin = f"""================================================================================
           JAL SANKETH - EARLY WARNING & INUNDATION BULLETIN
================================================================================
Bulletin Reference : JS-ALERT-{datetime.now().strftime('%Y%m%d%H%M')}
Generated At       : {now_str}
Location / Station : {loc_name} ({city}, {state})
Risk Category      : {risk_cat.upper()} (Risk Index: {score}/100)
Predicted 24h Rain : {rain_mm} mm
Est. Inundation    : {depth_cm} cm localized pooling potential
--------------------------------------------------------------------------------
SITUATION OVERVIEW:
{location_row.get('action_advisory', 'Precautionary monitoring in effect.')}

STANDARD OPERATING PROCEDURES (SOPs):
"""
        for stakeholder, actions in sops.items():
            bulletin += f"\n▶ [{stakeholder.upper()}]\n"
            for act in actions:
                bulletin += f"  • {act}\n"
                
        bulletin += """
--------------------------------------------------------------------------------
Official Advisory issued by Jal Sanketh Automated Hydro-Meteorological Decision Engine.
Prototype developed for Smart India Hackathon 2026.
================================================================================
"""
        return bulletin


if __name__ == "__main__":
    sample_row = pd.Series({
        "location_name": "Mumbai - Dadar / Hindmata Lowland",
        "city": "Mumbai",
        "state": "Maharashtra",
        "risk_category": "Critical",
        "risk_score": 86.4,
        "predicted_rainfall_mm": 138.5,
        "inundation_depth_cm": 42.0,
        "action_advisory": "EMERGENCY ALERT! High probability of widespread inundation."
    })
    print(AlertSystem.generate_bulletin_text(sample_row))

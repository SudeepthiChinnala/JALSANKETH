"""
JAL SANKETH - Emergency Evacuation Module
Module Path: evacuation/emergency.py

This module provides emergency evacuation functionality that integrates with the existing
flood prediction model to trigger evacuation alerts when CRITICAL risk is detected.

Key Features:
-------------
1. Detects CRITICAL flood risk from existing model output
2. Displays emergency section when risk level is critical
3. Shows flood risk information, user location, and recommended safe places
4. Provides evacuation guidance without claiming definite flood occurrence
5. Integrates with existing flood risk engine and safe places database
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from evacuation.safe_places import SafePlaceDatabase
    from src.location import get_user_location
except ImportError:
    SafePlaceDatabase = None
    get_user_location = None


def is_critical_risk(risk_level: str, risk_score: float) -> bool:
    """
    Determines if the current flood risk level is CRITICAL.
    
    Parameters:
    -----------
    risk_level : str
        Risk level from the existing model (Low, Moderate, High, Critical)
    risk_score : float
        Risk score from the existing model (0-100)
        
    Returns:
    --------
    bool : True if risk is CRITICAL (score >= 75), False otherwise
    """
    return risk_level.lower() == "critical" or risk_score >= 75.0


def render_emergency_alert(
    risk_level: str,
    risk_score: float,
    location_name: str,
    rainfall_mm: float,
    heavy_rain_prob: float
) -> bool:
    """
    Renders emergency evacuation alert when CRITICAL risk is detected.
    
    Parameters:
    -----------
    risk_level : str
        Current risk level from the flood prediction model
    risk_score : float
        Current risk score from the flood prediction model (0-100)
    location_name : str
        Name of the monitored location
    rainfall_mm : float
        Predicted rainfall in mm
    heavy_rain_prob : float
        Probability of heavy rain (0-1)
        
    Returns:
    --------
    bool : True if emergency alert was displayed, False otherwise
    """
    # Only show emergency alert for critical risk
    if not is_critical_risk(risk_level, risk_score):
        return False
    
    # Get user location
    user_lat, user_lon = get_user_location() if get_user_location else (None, None)
    
    # Find nearest safe place
    nearest_safe_place = None
    if user_lat is not None and user_lon is not None and SafePlaceDatabase:
        try:
            safe_db = SafePlaceDatabase()
            nearest_places = safe_db.find_nearest_safe_places(user_lat, user_lon, limit=1)
            if nearest_places:
                nearest_safe_place = nearest_places[0]
        except Exception as e:
            st.error(f"Error loading safe places: {e}")
    
    # Display emergency alert section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 3px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.4);
    ">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <div style="font-size: 48px;">🚨</div>
            <div>
                <div style="font-size: 28px; font-weight: 800; color: #ffffff; text-transform: uppercase; letter-spacing: 2px;">
                    EMERGENCY EVACUATION ALERT
                </div>
                <div style="font-size: 14px; color: #fca5a5; margin-top: 5px;">
                    CRITICAL FLOOD RISK DETECTED
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #1f1f1f; border: 2px solid #ef4444; border-radius: 8px; padding: 15px;">
            <div style="font-size: 12px; color: #ef4444; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                📊 FLOOD RISK STATUS
            </div>
            <div style="font-size: 24px; font-weight: 800; color: #ffffff;">
                CRITICAL
            </div>
            <div style="font-size: 14px; color: #fca5a5; margin-top: 5px;">
                Risk Score: {:.1f}/100
            </div>
        </div>
        """.format(risk_score), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1f1f1f; border: 2px solid #f97316; border-radius: 8px; padding: 15px;">
            <div style="font-size: 12px; color: #f97316; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                📍 LOCATION
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #ffffff;">
                {}
            </div>
            <div style="font-size: 14px; color: #fdba74; margin-top: 5px;">
                Rainfall: {:.1f} mm
            </div>
        </div>
        """.format(location_name, rainfall_mm), unsafe_allow_html=True)
    
    with col3:
        if nearest_safe_place:
            st.markdown("""
            <div style="background: #1f1f1f; border: 2px solid #22c55e; border-radius: 8px; padding: 15px;">
                <div style="font-size: 12px; color: #22c55e; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                    🏠 RECOMMENDED SAFE PLACE
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #ffffff;">
                    {}
                </div>
                <div style="font-size: 14px; color: #86efac; margin-top: 5px;">
                    Distance: {:.1f} km
                </div>
            </div>
            """.format(nearest_safe_place['name'], nearest_safe_place['distance_km']), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #1f1f1f; border: 2px solid #6b7280; border-radius: 8px; padding: 15px;">
                <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                    🏠 SAFE PLACE
                </div>
                <div style="font-size: 14px; color: #9ca3af;">
                    Set your location to find nearest safe place
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # User location information
    if user_lat is not None and user_lon is not None:
        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px; margin: 15px 0;">
            <div style="font-size: 12px; color: #38bdf8; text-transform: uppercase; font-weight: 700; margin-bottom: 5px;">
                📍 YOUR CURRENT LOCATION
            </div>
            <div style="font-size: 14px; color: #e2e8f0;">
                Latitude: {user_lat:.6f}°N, Longitude: {user_lon:.6f}°E
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div style="background: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 12px; margin: 15px 0;">
        <div style="font-size: 12px; color: #818cf8; text-transform: uppercase; font-weight: 700; margin-bottom: 5px;">
            ⚠️ IMPORTANT DISCLAIMER
        </div>
        <div style="font-size: 13px; color: #c7d2fe; line-height: 1.5;">
            <b>Predicted flood risk: CRITICAL</b><br>
            This alert is based on the existing flood prediction model showing critical risk conditions. 
            The model predicts elevated flood risk based on current weather patterns and hydrological factors. 
            Actual flood occurrence depends on multiple variables and local conditions. 
            Please follow official emergency guidance from local authorities.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Evacuation button
    st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("🚨 START EVACUATION", type="primary", width='stretch', key="emergency_evacuation_button"):
            st.session_state["evacuation_started"] = True
            st.success("✅ Evacuation protocol initiated. Please proceed to the recommended safe place.")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # If evacuation started, show additional guidance
    if st.session_state.get("evacuation_started", False):
        st.markdown("""
        <div style="background: #064e3b; border: 2px solid #22c55e; border-radius: 8px; padding: 15px; margin: 15px 0;">
            <div style="font-size: 14px; font-weight: 700; color: #22c55e; margin-bottom: 10px;">
                ✅ EVACUATION IN PROGRESS
            </div>
            <div style="font-size: 13px; color: #86efac; line-height: 1.6;">
                <b>Immediate Actions:</b><br>
                • Gather essential documents, medications, and emergency supplies<br>
                • Follow the recommended route to the safe place<br>
                • Inform family members about your evacuation status<br>
                • Monitor official emergency broadcasts<br>
                • Avoid flooded areas and fast-moving water
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return True


def get_emergency_status(risk_level: str, risk_score: float) -> Dict[str, Any]:
    """
    Get the current emergency status based on flood risk.
    
    Parameters:
    -----------
    risk_level : str
        Current risk level from the flood prediction model
    risk_score : float
        Current risk score from the flood prediction model (0-100)
        
    Returns:
    --------
    Dict : Emergency status information
    """
    is_emergency = is_critical_risk(risk_level, risk_score)
    
    return {
        "is_emergency": is_emergency,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "severity": "CRITICAL" if is_emergency else "NORMAL",
        "requires_evacuation": is_emergency
    }


def reset_evacuation_status():
    """
    Reset the evacuation status in session state.
    """
    if "evacuation_started" in st.session_state:
        del st.session_state["evacuation_started"]
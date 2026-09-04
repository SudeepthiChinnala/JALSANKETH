"""
JAL SANKETH - Live Location Detection Module
Module Path: src/location.py

This module provides functionality for detecting user location using browser geolocation
and manual entry fallback for the JAL SANKETH flood intelligence dashboard.

Key Features:
-------------
1. Browser geolocation detection using Streamlit components
2. Manual latitude/longitude entry fallback
3. Location validation for Telangana region
4. Session state management for location data
"""

import streamlit as st
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def is_in_telangana(lat: float, lon: float) -> bool:
    """
    Validates whether given geographic coordinates fall within Telangana.
    
    Parameters:
    -----------
    lat : float
        Latitude coordinate
    lon : float
        Longitude coordinate
        
    Returns:
    --------
    bool : True if within Telangana bounds, False otherwise
    """
    # Approximate geographical bounding box for Telangana State
    # Lat: ~15.7° N to 19.95° N, Lon: ~77.1° E to 81.9° E
    return (
        15.70 <= lat <= 19.95
        and 77.10 <= lon <= 81.90
    )


def render_location_detection():
    """
    Renders the location detection UI in the sidebar.
    Provides both automatic geolocation and manual entry options.
    """
    st.markdown("**YOUR LOCATION**")
    
    # Initialize session state for location
    if "user_lat" not in st.session_state:
        st.session_state.user_lat = None
    if "user_lon" not in st.session_state:
        st.session_state.user_lon = None
    if "location_method" not in st.session_state:
        st.session_state.location_method = None
    if "location_name" not in st.session_state:
        st.session_state.location_name = None
    
    # Location detection options
    location_method = st.radio(
        "Location Detection Method",
        ["📍 Use My Location", "📍 Enter Location Manually"],
        label_visibility="collapsed",
        horizontal=True
    )
    
    if location_method == "📍 Use My Location":
        # For this implementation, we'll use a simulated automatic detection
        # In production, this would use proper browser geolocation APIs
        st.markdown("""
        <div style="background: #0c2d48; border: 1px solid #00f0ff; padding: 8px; border-radius: 6px; font-size: 0.75rem; color: #7dd3fc;">
            <b>🌍 Browser Geolocation</b><br>
            Click below to detect your current location using your device's GPS.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📍 Detect My Location", use_container_width=True):
            # Simulate geolocation for testing (in production, use proper browser API)
            # This will use Hyderabad coordinates as default for demonstration
            st.session_state.user_lat = 17.3850
            st.session_state.user_lon = 78.4867
            st.session_state.location_method = "automatic"
            st.session_state.location_name = "Detected Location (Hyderabad)"
            st.success("Location detected: 17.3850, 78.4867 (Hyderabad - Demo Mode)")
            st.info("🔧 In production, this will use your actual device GPS location")
            st.rerun()
        
        st.caption("🔧 Currently in demo mode - will use actual GPS in production")
        
    else:
        # Manual location entry
        st.markdown("""
        <div style="background: #2e1d05; border: 1px solid #d97706; padding: 8px; border-radius: 6px; font-size: 0.75rem; color: #fde047;">
            <b>📝 Manual Entry</b><br>
            Enter your coordinates manually below.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            manual_lat = st.number_input("Latitude", value=17.3850, format="%.6f", key="manual_lat")
        with col2:
            manual_lon = st.number_input("Longitude", value=78.4867, format="%.6f", key="manual_lon")
        
        if st.button("📍 Set Manual Location", use_container_width=True):
            st.session_state.user_lat = manual_lat
            st.session_state.user_lon = manual_lon
            st.session_state.location_method = "manual"
            st.session_state.location_name = f"Manual Location ({manual_lat:.4f}, {manual_lon:.4f})"
            st.success(f"Location set: {manual_lat:.6f}, {manual_lon:.6f}")
            st.rerun()
    
    # Display current location if set
    if st.session_state.user_lat is not None and st.session_state.user_lon is not None:
        lat = st.session_state.user_lat
        lon = st.session_state.user_lon
        
        # Check if location is in Telangana
        in_telangana = is_in_telangana(lat, lon)
        
        location_status = "✅ Within Telangana" if in_telangana else "⚠️ Outside Telangana"
        status_color = "#22c55e" if in_telangana else "#f97316"
        
        st.markdown(f"""
        <div style="background: #0b1528; border: 1px solid #1e293b; padding: 8px; border-radius: 6px; font-size: 0.75rem; color: #e2e8f0;">
            <div style="font-weight: bold; color: #00f0ff; margin-bottom: 4px;">📍 Current Location</div>
            <div>Latitude: {lat:.6f}</div>
            <div>Longitude: {lon:.6f}</div>
            <div style="color: {status_color}; margin-top: 4px;">{location_status}</div>
            <div style="color: #94a3b8; font-size: 0.7rem; margin-top: 2px;">
                Method: {st.session_state.location_method}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not in_telangana:
            st.warning("⚠️ Your location is outside Telangana. Flood predictions are optimized for Telangana region.")
    else:
        st.info("📍 Set your location to get personalized flood risk information")
    
    return st.session_state.user_lat, st.session_state.user_lon


def get_user_location():
    """
    Get the current user location from session state.
    
    Returns:
    --------
    tuple : (latitude, longitude) or (None, None) if not set
    """
    return st.session_state.get("user_lat"), st.session_state.get("user_lon")


def set_user_location(lat: float, lon: float, method: str = "manual", name: str = None):
    """
    Set the user location in session state.
    
    Parameters:
    -----------
    lat : float
        Latitude coordinate
    lon : float
        Longitude coordinate
    method : str
        Detection method ("automatic" or "manual")
    name : str, optional
        Location name/description
    """
    st.session_state.user_lat = lat
    st.session_state.user_lon = lon
    st.session_state.location_method = method
    st.session_state.location_name = name or f"Location ({lat:.4f}, {lon:.4f})"


def clear_user_location():
    """
    Clear the user location from session state.
    """
    st.session_state.user_lat = None
    st.session_state.user_lon = None
    st.session_state.location_method = None
    st.session_state.location_name = None

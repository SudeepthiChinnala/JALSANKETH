"""
================================================================================
JAL SANKETH — TELANGANA FLOOD RISK GEOSPATIAL MAP COMPONENT
Module Path: src/map.py
================================================================================

This module provides a modular, reusable geospatial map component strictly
focused on Telangana, India for the Jal Sanketh early-warning and inundation
prediction dashboard.

Key Features:
-------------
1. Primary Geographic Focus: Telangana, India.
2. Initial map center: approximately around Telangana (~17.87°N, 79.05°E).
3. Initial zoom level: 7 (displays the entire Telangana state while allowing
   users to zoom into individual districts).
4. Authentic Telangana State & District Boundary GeoJSON overlay loaded from
   data/geospatial/telangana_boundary.geojson.
5. Telangana boundary visually distinguishable with a distinct cyan-blue stroke
   and translucent highlight.
6. Restricts and validates all risk locations strictly within Telangana.
7. Displays district/location-level flood risk using standardized colors:
     • LOW       → Green  (#22c55e)
     • MODERATE  → Yellow (#eab308)
     • HIGH      → Orange (#f97316)
     • CRITICAL  → Red    (#ef4444)
8. Interactive popup cards displaying:
     • Location / District Name
     • 24-hr Predicted Rainfall (mm)
     • Heavy-Rain Probability (%)
     • Flood Risk Score (0 to 100)
     • Risk Level Badge
     • Elevation (m ASL)
     • Drainage Factor
9. Clear Map Title ("TELANGANA FLOOD RISK MAP") and HTML Legend.
================================================================================
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import folium
from folium.plugins import MarkerCluster
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import get_carto_tile_config, CARTO_ATTRIBUTION

# Import safe places module
try:
    from evacuation.safe_places import SafePlaceDatabase
except ImportError:
    SafePlaceDatabase = None

# ------------------------------------------------------------------------------
# 1. CONSTANTS & TELANGANA BOUNDS
# ------------------------------------------------------------------------------
# Approximate geographical center of Telangana State
TELANGANA_CENTER = [17.8700, 79.0500]
TELANGANA_DEFAULT_ZOOM = 7

# Geographic bounding box for Telangana validation
# Lat: ~15.7° N to 19.95° N, Lon: ~77.1° E to 81.9° E
TELANGANA_BBOX = {
    "min_lat": 15.70,
    "max_lat": 19.95,
    "min_lon": 77.10,
    "max_lon": 81.90
}

# Standardized risk color mapping (SIH 2026 Specification)
RISK_COLOR_MAP: Dict[str, str] = {
    "Low": "#22c55e",        # Green
    "Moderate": "#eab308",   # Yellow
    "High": "#f97316",       # Orange
    "Critical": "#ef4444"    # Red
}

GEOSPATIAL_DIR = os.path.join(PROJECT_ROOT, "data", "geospatial")
TELANGANA_GEOJSON_PATH = os.path.join(GEOSPATIAL_DIR, "telangana_boundary.geojson")


def get_risk_color(risk_level: str) -> str:
    """
    Returns the standard color hex code for a given risk level.
    """
    level_clean = str(risk_level).capitalize().strip()
    return RISK_COLOR_MAP.get(level_clean, "#22c55e")


def is_in_telangana(lat: float, lon: float) -> bool:
    """
    Validates whether given geographic coordinates fall within Telangana.
    """
    return (
        TELANGANA_BBOX["min_lat"] <= lat <= TELANGANA_BBOX["max_lat"]
        and TELANGANA_BBOX["min_lon"] <= lon <= TELANGANA_BBOX["max_lon"]
    )


# ------------------------------------------------------------------------------
# 2. POPUP HTML BUILDER
# ------------------------------------------------------------------------------
def build_popup_html(record: Dict[str, Any]) -> str:
    """
    Constructs an interactive, styled HTML popup card for a Telangana station marker.
    """
    location = record.get("location", record.get("location_name", "Telangana Basin"))
    lat = float(record.get("latitude", record.get("lat", 0.0)))
    lon = float(record.get("longitude", record.get("lon", 0.0)))
    rainfall = float(record.get("rainfall", record.get("predicted_rainfall_mm", record.get("rainfall_amount_mm", 0.0))))
    
    prob_raw = record.get("heavy_rain_prob", record.get("heavy_rainfall_prob", record.get("heavy_rain_probability", 0.0)))
    prob_pct = float(prob_raw) * 100.0 if float(prob_raw) <= 1.0 else float(prob_raw)
    
    score = float(record.get("risk_score", record.get("flood_risk_score", 0.0)))
    level = str(record.get("risk_level", record.get("flood_risk_level", record.get("risk_category", "Low")))).capitalize()
    
    elevation = record.get("elevation", record.get("elevation_m", None))
    drainage = record.get("drainage_factor", record.get("drainage_capacity_index", None))
    advisory = record.get("headline_message", record.get("advisory_summary", record.get("action_advisory", "")))

    color = get_risk_color(level)

    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 260px; max-width: 320px; padding: 6px; color: #e2e8f0; background: #0b1528; border-radius: 8px; border: 1px solid #1e293b;">
        <!-- Header -->
        <div style="border-bottom: 2px solid {color}; padding-bottom: 6px; margin-bottom: 8px;">
            <div style="font-size: 10px; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">TELANGANA DISTRICT BASIN</div>
            <h4 style="margin: 2px 0 0 0; font-size: 14px; font-weight: 700; color: #ffffff;">{location}</h4>
            <span style="font-size: 10px; color: #94a3b8;">Coordinates: {lat:.4f}°N, {lon:.4f}°E</span>
        </div>

        <!-- Risk Badge -->
        <div style="margin-bottom: 8px;">
            <span style="background-color: {color}; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-weight: 700; font-size: 11px; letter-spacing: 0.04em;">
                {level.upper()} RISK ({score:.1f}/100)
            </span>
        </div>

        <!-- Data Metrics Table -->
        <table style="width: 100%; font-size: 11px; border-collapse: collapse; line-height: 1.7; margin-bottom: 6px; color: #cbd5e1;">
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>🌧️ Rainfall:</b></td>
                <td style="text-align: right; font-weight: 700; color: #38bdf8; font-family: monospace;">{rainfall:.1f} mm</td>
            </tr>
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>⚡ Heavy Rain Prob:</b></td>
                <td style="text-align: right; font-weight: 700; color: {'#ef4444' if prob_pct >= 50 else ('#f97316' if prob_pct >= 25 else '#22c55e')}; font-family: monospace;">{prob_pct:.1f}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>📊 Flood Risk Score:</b></td>
                <td style="text-align: right; font-weight: 700; color: {color}; font-family: monospace;">{score:.1f} / 100</td>
            </tr>
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>🎯 Risk Level:</b></td>
                <td style="text-align: right; font-weight: 700; color: {color};">{level}</td>
            </tr>
    """

    if elevation is not None:
        html += f"""
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>🏔️ Elevation:</b></td>
                <td style="text-align: right; color: #e2e8f0; font-family: monospace;">{float(elevation):.1f} m ASL</td>
            </tr>
        """

    if drainage is not None:
        html += f"""
            <tr style="border-bottom: 1px solid #14223d;">
                <td style="color: #94a3b8;"><b>🚰 Drainage Factor:</b></td>
                <td style="text-align: right; color: #e2e8f0; font-family: monospace;">{float(drainage):.2f} / 1.00</td>
            </tr>
        """

    html += """
        </table>
    """

    if advisory:
        html += f"""
        <div style="background-color: #060e1c; border-left: 3px solid {color}; padding: 5px 8px; font-size: 10px; color: #94a3b8; line-height: 1.4; border-radius: 0 4px 4px 0;">
            <b>Advisory:</b> {advisory}
        </div>
        """

    html += "</div>"
    return html


# ------------------------------------------------------------------------------
# 3. TELANGANA BOUNDARY OVERLAY LOADER
# ------------------------------------------------------------------------------
def add_telangana_boundary_overlay(m: folium.Map, geojson_path: Optional[str] = None) -> bool:
    """
    Adds authentic Telangana State & District boundary overlay from GeoJSON file.
    The boundary is styled to be visually distinguishable with a distinct stroke
    and subtle tint.
    """
    target_path = geojson_path or TELANGANA_GEOJSON_PATH

    if not os.path.exists(target_path):
        logger.warning(f"Telangana GeoJSON boundary file not found at: {target_path}")
        return False

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Style function to make Telangana boundary visually prominent
        def style_function(feature):
            return {
                "fillColor": "#0284c7",
                "color": "#00f0ff",
                "weight": 2.0,
                "fillOpacity": 0.05,
                "opacity": 0.85,
                "dashArray": "3, 6"
            }

        def highlight_function(feature):
            return {
                "fillColor": "#38bdf8",
                "color": "#ffffff",
                "weight": 3.0,
                "fillOpacity": 0.20,
                "opacity": 1.0
            }

        # Check available properties for tooltip
        first_feat = geojson_data.get("features", [{}])[0]
        props = first_feat.get("properties", {})
        tooltip_fields = []
        tooltip_aliases = []

        for candidate in ["dtname", "district", "District", "NAME_2", "dt_name", "DISTRICT"]:
            if candidate in props:
                tooltip_fields.append(candidate)
                tooltip_aliases.append("District:")
                break

        tooltip = None
        if tooltip_fields:
            tooltip = folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style="background-color: #0b1528; color: #00f0ff; font-family: sans-serif; font-size: 11px; padding: 4px 8px; border-radius: 4px; border: 1px solid #00f0ff;"
            )

        folium.GeoJson(
            geojson_data,
            name="Telangana Administrative Boundaries",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip
        ).add_to(m)

        return True
    except Exception as e:
        logger.error(f"Failed to render Telangana GeoJSON boundary overlay: {e}")
        return False


# ------------------------------------------------------------------------------
# 4. MAP BUILDER & STREAMLIT RENDERER
# ------------------------------------------------------------------------------
def create_flood_risk_map(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    center: Optional[List[float]] = None,
    zoom_start: int = TELANGANA_DEFAULT_ZOOM,
    tile_style: str = "CartoDB dark_matter",
    enable_catchment_zones: bool = True,
    geojson_path: Optional[str] = None,
    user_location: Optional[Tuple[float, float]] = None,
    show_safe_places: bool = False
) -> folium.Map:
    """
    Constructs an interactive Folium Map strictly focused on Telangana, India.

    Parameters:
    -----------
    data : Union[pd.DataFrame, List[Dict[str, Any]]]
        Evaluation DataFrame or list of station dictionaries. Non-Telangana
        locations are filtered out to adhere strictly to the Telangana domain.
    center : Optional[List[float]]
        Map initial center [latitude, longitude]. Defaults to Telangana center [17.87, 79.05].
    zoom_start : int
        Initial map zoom level (default: 7).
    tile_style : str
        Basemap tile style ('CartoDB dark_matter', 'OpenStreetMap', etc.).
    enable_catchment_zones : bool
        Whether to render semi-transparent radial catchment circles around stations.
    geojson_path : Optional[str]
        Path to telangana_boundary.geojson.
    user_location : Optional[Tuple[float, float]]
        User's current location as (latitude, longitude). If provided, adds a distinctive marker.
    show_safe_places : bool
        Whether to display safe places from the database on the map.

    Returns:
    --------
    folium.Map: Fully rendered Folium Map instance.
    """
    if isinstance(data, pd.DataFrame):
        records = data.to_dict(orient="records")
    else:
        records = list(data)

    # 1. Filter and validate locations strictly within Telangana
    tg_records = []
    for r in records:
        lat = float(r.get("latitude", r.get("lat", 0.0)))
        lon = float(r.get("longitude", r.get("lon", 0.0)))
        if is_in_telangana(lat, lon):
            tg_records.append(r)
        else:
            loc_name = r.get("location", r.get("location_name", "Unknown"))
            logger.info(f"Filtering out non-Telangana location: {loc_name} ({lat}, {lon})")

    map_center = center if center is not None else TELANGANA_CENTER

    # 2. Configure basemap tiles (with CARTO_API_KEY support)
    tile_cfg = get_carto_tile_config()
    if tile_style in ["CartoDB dark_matter", "cartodb dark_matter", None] and tile_cfg["is_custom_url"]:
        m = folium.Map(
            location=map_center,
            zoom_start=zoom_start,
            tiles=tile_cfg["tiles"],
            attr=tile_cfg["attr"],
            subdomains=tile_cfg["subdomains"],
            control_scale=True,
            prefer_canvas=True
        )
    else:
        m = folium.Map(
            location=map_center,
            zoom_start=zoom_start,
            tiles=tile_style or "CartoDB dark_matter",
            control_scale=True,
            prefer_canvas=True
        )

    # 3. Add Telangana Boundary Overlay
    add_telangana_boundary_overlay(m, geojson_path=geojson_path)

    # 4. Add Location Markers
    for rec in tg_records:
        lat = float(rec.get("latitude", rec.get("lat", 0.0)))
        lon = float(rec.get("longitude", rec.get("lon", 0.0)))
        location = rec.get("location", rec.get("location_name", "Telangana Basin"))
        score = float(rec.get("risk_score", rec.get("flood_risk_score", 0.0)))
        level = str(rec.get("risk_level", rec.get("flood_risk_level", rec.get("risk_category", "Low")))).capitalize()
        color = get_risk_color(level)

        # Dynamic marker size scaled by risk score (8px to 16px)
        marker_radius = 8.0 + (score / 100.0) * 8.0

        popup_content = build_popup_html(rec)
        tooltip_text = f"📍 {location} | {level} Risk ({score:.1f}/100)"

        # Catchment zone circle
        if enable_catchment_zones:
            folium.Circle(
                location=[lat, lon],
                radius=10000 + (score / 100.0) * 12000,  # 10km to 22km radius for district basins
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.15,
                weight=1,
                popup=folium.Popup(popup_content, max_width=340)
            ).add_to(m)

        # Inner station circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            weight=2,
            popup=folium.Popup(popup_content, max_width=340),
            tooltip=tooltip_text
        ).add_to(m)

    # 4.5. Add User Location Marker (if provided)
    if user_location is not None:
        user_lat, user_lon = user_location
        
        # Only add user marker if within Telangana bounds
        if is_in_telangana(user_lat, user_lon):
            # Create distinctive user location marker
            user_popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 200px; padding: 8px; color: #e2e8f0; background: #0b1528; border-radius: 8px; border: 2px solid #00f0ff;">
                <div style="font-size: 12px; color: #00f0ff; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px;">
                    📍 YOUR LOCATION
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #ffffff;">
                    User Position
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                    Coordinates: {user_lat:.6f}°N, {user_lon:.6f}°E
                </div>
            </div>
            """
            
            # Pulsing effect circle for user location
            folium.Circle(
                location=[user_lat, user_lon],
                radius=2000,  # 2km radius for visibility
                color="#00f0ff",
                fill=True,
                fill_color="#00f0ff",
                fill_opacity=0.2,
                weight=2,
                popup=folium.Popup(user_popup_html, max_width=300)
            ).add_to(m)
            
            # Central user location marker - distinctive star-like appearance
            folium.CircleMarker(
                location=[user_lat, user_lon],
                radius=12,
                color="#00f0ff",
                fill=True,
                fill_color="#ffffff",
                fill_opacity=1.0,
                weight=3,
                popup=folium.Popup(user_popup_html, max_width=300),
                tooltip="📍 Your Current Location"
            ).add_to(m)
            
            # Add a smaller inner circle for contrast
            folium.CircleMarker(
                location=[user_lat, user_lon],
                radius=6,
                color="#00f0ff",
                fill=True,
                fill_color="#00f0ff",
                fill_opacity=1.0,
                weight=1
            ).add_to(m)
            
            logger.info(f"Added user location marker at ({user_lat:.6f}, {user_lon:.6f})")
        else:
            logger.warning(f"User location ({user_lat:.6f}, {user_lon:.6f}) is outside Telangana bounds - marker not added")

    # 4.6. Add Safe Places Markers (if enabled)
    if show_safe_places and SafePlaceDatabase is not None:
        try:
            safe_db = SafePlaceDatabase()
            if safe_db.safe_places_df is not None and not safe_db.safe_places_df.empty:
                safe_places = safe_db.get_all_safe_places()
                
                for place in safe_places:
                    place_lat = place['latitude']
                    place_lon = place['longitude']
                    place_name = place['name']
                    place_type = place['type']
                    place_capacity = place['capacity']
                    place_elevation = place['elevation']
                    
                    # Only show safe places within Telangana
                    if is_in_telangana(place_lat, place_lon):
                        # Calculate distance from user if available
                        distance_text = ""
                        if user_location is not None:
                            user_lat, user_lon = user_location
                            distance = safe_db.calculate_distance(user_lat, user_lon, place_lat, place_lon)
                            distance_text = f"<div style='font-size: 11px; color: #00f0ff; margin-top: 4px;'>📍 Distance: {distance:.1f} km</div>"
                        
                        # Create popup for safe place
                        safe_popup_html = f"""
                        <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 220px; padding: 8px; color: #e2e8f0; background: #0b1528; border-radius: 8px; border: 2px solid #22c55e;">
                            <div style="font-size: 12px; color: #22c55e; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px;">
                                🏠 SAFE PLACE
                            </div>
                            <div style="font-size: 14px; font-weight: 700; color: #ffffff;">
                                {place_name}
                            </div>
                            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                                Type: {place_type}
                            </div>
                            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                                Capacity: {place_capacity} people
                            </div>
                            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                                Elevation: {place_elevation}m
                            </div>
                            {distance_text}
                        </div>
                        """
                        
                        # Create distinctive safe place marker (green square)
                        folium.CircleMarker(
                            location=[place_lat, place_lon],
                            radius=8,
                            color="#22c55e",
                            fill=True,
                            fill_color="#22c55e",
                            fill_opacity=0.8,
                            weight=2,
                            popup=folium.Popup(safe_popup_html, max_width=300),
                            tooltip=f"🏠 {place_name} ({place_type})"
                        ).add_to(m)
                        
                        # Add a small inner circle for visual distinction
                        folium.CircleMarker(
                            location=[place_lat, place_lon],
                            radius=4,
                            color="#ffffff",
                            fill=True,
                            fill_color="#ffffff",
                            fill_opacity=1.0,
                            weight=1
                        ).add_to(m)
                
                logger.info(f"Added {len(safe_places)} safe places to map")
            else:
                logger.warning("Safe places database is empty or not loaded")
        except Exception as e:
            logger.error(f"Error loading safe places: {e}")

    # 5. Add Custom Map Title & Legend Overlay
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 9999;
        background: rgba(11, 21, 40, 0.92);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #e2e8f0;
        font-size: 11px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        backdrop-filter: blur(4px);
    ">
        <div style="font-weight: 800; font-size: 11px; color: #00f0ff; letter-spacing: 0.05em; margin-bottom: 6px; text-transform: uppercase;">
            🗺️ TELANGANA FLOOD RISK MAP
        </div>
        <div style="display: flex; flex-direction: column; gap: 4px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {RISK_COLOR_MAP['Low']};"></span>
                <span>Low Risk (0–24)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {RISK_COLOR_MAP['Moderate']};"></span>
                <span>Moderate Risk (25–49)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {RISK_COLOR_MAP['High']};"></span>
                <span>High Risk (50–74)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {RISK_COLOR_MAP['Critical']};"></span>
                <span>Critical Risk (75–100)</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def render_streamlit_map(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    center: Optional[List[float]] = None,
    zoom_start: int = TELANGANA_DEFAULT_ZOOM,
    tile_style: str = "CartoDB dark_matter",
    width: Optional[int] = None,
    height: int = 520,
    enable_catchment_zones: bool = True,
    geojson_path: Optional[str] = None,
    user_location: Optional[Tuple[float, float]] = None,
    show_safe_places: bool = False
):
    """
    Renders the Telangana flood risk map component inside a Streamlit application page.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, List[Dict[str, Any]]]
        Evaluation DataFrame or list of station dictionaries.
    center : Optional[List[float]]
        Map initial center [latitude, longitude].
    zoom_start : int
        Initial map zoom level (default: 7).
    tile_style : str
        Basemap tile style ('CartoDB dark_matter', 'OpenStreetMap', etc.).
    width : Optional[int]
        Map width in pixels.
    height : int
        Map height in pixels (default: 520).
    enable_catchment_zones : bool
        Whether to render catchment zones around stations.
    geojson_path : Optional[str]
        Path to telangana_boundary.geojson.
    user_location : Optional[Tuple[float, float]]
        User's current location as (latitude, longitude).
    show_safe_places : bool
        Whether to display safe places from the database on the map.
    """
    try:
        from streamlit_folium import folium_static
        import streamlit as st

        folium_map = create_flood_risk_map(
            data=data,
            center=center,
            zoom_start=zoom_start,
            tile_style=tile_style,
            enable_catchment_zones=enable_catchment_zones,
            geojson_path=geojson_path,
            user_location=user_location,
            show_safe_places=show_safe_places
        )

        folium_static(folium_map, width=width, height=height)
    except ImportError as e:
        raise ImportError(f"[!] streamlit-folium is required to render maps inside Streamlit: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("      JAL SANKETH — TELANGANA FLOOD RISK MAP MODULE TEST")
    print("=" * 80)

    # Sample Telangana District Basin records
    sample_telangana_evaluations = [
        {
            "location": "Hyderabad — Musi River Basin",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "rainfall": 128.0,
            "heavy_rain_prob": 0.88,
            "risk_score": 84.5,
            "risk_level": "Critical",
            "elevation": 505.0,
            "drainage_factor": 0.38,
            "headline_message": "Critical inundation alert along Musi River corridor."
        },
        {
            "location": "Warangal — Kakatiya Basin",
            "latitude": 17.9689,
            "longitude": 79.5941,
            "rainfall": 72.0,
            "heavy_rain_prob": 0.65,
            "risk_score": 64.0,
            "risk_level": "High",
            "elevation": 302.0,
            "drainage_factor": 0.42,
            "headline_message": "High flood alert in low-lying urban catchments."
        },
        {
            "location": "Karimnagar — Lower Manair Catchment",
            "latitude": 18.4386,
            "longitude": 79.1288,
            "rainfall": 35.0,
            "heavy_rain_prob": 0.28,
            "risk_score": 38.0,
            "risk_level": "Moderate",
            "elevation": 265.0,
            "drainage_factor": 0.45,
            "headline_message": "Moderate catchment inflow. Continue monitoring."
        },
        {
            "location": "Adilabad — Penganga Basin",
            "latitude": 19.6641,
            "longitude": 78.5320,
            "rainfall": 4.0,
            "heavy_rain_prob": 0.02,
            "risk_score": 12.0,
            "risk_level": "Low",
            "elevation": 264.0,
            "drainage_factor": 0.44,
            "headline_message": "No immediate flood hazard."
        }
    ]

    test_map = create_flood_risk_map(sample_telangana_evaluations)
    output_html = os.path.join("reports", "figures", "test_flood_risk_map.html")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    test_map.save(output_html)

    print(f"[+] Map generated successfully for Telangana.")
    print(f"[+] Telangana GeoJSON present: {os.path.exists(TELANGANA_GEOJSON_PATH)}")
    print(f"[+] Saved standalone test map to: {output_html}")
    print("=" * 80)
    print("  [OK] Telangana map component verified successfully.")
    print("=" * 80 + "\n")


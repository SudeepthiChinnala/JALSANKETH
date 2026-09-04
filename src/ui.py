"""
================================================================================
JAL SANKETH — HIGH-TECH DISASTER MANAGEMENT COMMAND CENTER UI THEME
Module Path: src/ui.py
================================================================================

This module encapsulates all CSS styling, Plotly dark theme templates, and
reusable command-center UI card components for the Jal Sanketh weather intelligence
and flood early-warning dashboard.
================================================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional


def load_command_center_css():
    """
    Injects professional dark-mode command-center CSS styling.
    """
    st.markdown("""
    <style>
        /* Base typography and dark command center palette */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global Background & Text */
        .stApp {
            background-color: #080e1a !important;
            color: #e2e8f0 !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }

        /* Top padding container */
        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 2.0rem !important;
            max-width: 1550px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #060b14 !important;
            border-right: 1px solid #172544 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox, 
        section[data-testid="stSidebar"] .stRadio {
            color: #e2e8f0 !important;
        }

        /* Top Operational Header */
        .cc-header {
            background: linear-gradient(180deg, #0d1a33 0%, #091224 100%);
            border: 1px solid #1a2f5a;
            border-radius: 8px;
            padding: 0.8rem 1.4rem;
            margin-bottom: 1.0rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        }
        .cc-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .cc-logo-icon {
            font-size: 1.6rem;
            color: #00f0ff;
            text-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
        }
        .cc-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
            margin: 0;
        }
        .cc-subtitle {
            font-size: 0.76rem;
            color: #38bdf8;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 0;
            font-weight: 500;
        }
        .cc-meta-status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #0d2847;
            border: 1px solid #0284c7;
            color: #38bdf8;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        .demo-pill {
            background: #2e1d05;
            border: 1px solid #d97706;
            color: #fbbf24;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        /* High-Tech Card Container */
        .cc-card {
            background: #0b1528;
            border: 1px solid #172a4d;
            border-radius: 8px;
            padding: 1.0rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            position: relative;
        }
        .cc-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #14223d;
            padding-bottom: 0.5rem;
            margin-bottom: 0.8rem;
        }
        .cc-card-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #00f0ff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: 'JetBrains Mono', monospace;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .cc-card-subtitle {
            font-size: 0.72rem;
            color: #64748b;
            margin: 0;
        }

        /* Metric Typography */
        .cc-metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
        }
        .cc-metric-unit {
            font-size: 0.95rem;
            color: #94a3b8;
            font-weight: 500;
        }

        /* Weather Attribute Rows */
        .cc-weather-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.38rem 0;
            border-bottom: 1px dashed #14223d;
            font-size: 0.82rem;
        }
        .cc-weather-label {
            color: #94a3b8;
            font-weight: 500;
        }
        .cc-weather-val {
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }

        /* Alert Feed Items */
        .cc-alert-item {
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            border-left: 4px solid;
            font-size: 0.82rem;
            background: #091222;
        }
        .cc-alert-critical {
            border-color: #ef4444;
            background: #1a0a0e;
        }
        .cc-alert-high {
            border-color: #f97316;
            background: #190e06;
        }
        .cc-alert-moderate {
            border-color: #eab308;
            background: #181305;
        }
        .cc-alert-low {
            border-color: #22c55e;
            background: #06160d;
        }

        .alert-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 4px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge-critical { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }
        .badge-high { background: #431407; color: #fb923c; border: 1px solid #f97316; }
        .badge-moderate { background: #422006; color: #fde047; border: 1px solid #eab308; }
        .badge-low { background: #052e16; color: #4ade80; border: 1px solid #22c55e; }

        /* Navigation Bar Tabs */
        .stButton button {
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.04em !important;
            transition: all 0.15s ease !important;
        }
        
        /* Map border framing */
        .folium-map-wrapper {
            border: 1px solid #1a2f5a;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Custom Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #080e1a;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
    </style>
    """, unsafe_allow_html=True)


def get_plotly_dark_theme() -> Dict[str, Any]:
    """
    Returns a unified Plotly layout dictionary matching the dark command-center aesthetic.
    """
    return dict(
        paper_bgcolor="#0b1528",
        plot_bgcolor="#0b1528",
        font=dict(family="JetBrains Mono, Inter, sans-serif", color="#e2e8f0", size=11),
        margin=dict(l=25, r=20, t=30, b=25),
        xaxis=dict(
            gridcolor="#172a4d",
            zerolinecolor="#1e3a6a",
            linecolor="#172a4d",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="#172a4d",
            zerolinecolor="#1e3a6a",
            linecolor="#172a4d",
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            bgcolor="rgba(11, 21, 40, 0.8)",
            bordercolor="#172a4d",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=10)
        )
    )


def render_command_center_header(
    active_tab: str,
    selected_location: str,
    selected_date: str,
    temp_c: float,
    condition: str = "Overcast / Precipitation"
):
    """
    Renders the fixed-looking operational top navigation header.
    """
    st.markdown(f"""
    <div class="cc-header">
        <div class="cc-brand">
            <div class="cc-logo-icon">💧</div>
            <div>
                <div class="cc-title">JAL SANKETH</div>
                <div class="cc-subtitle">AI-Powered Rainfall & Flood Risk Intelligence</div>
            </div>
        </div>
        <div class="cc-meta-status">
            <span class="status-pill">📍 {selected_location}</span>
            <span class="status-pill">📅 {selected_date}</span>
            <span class="status-pill">🌡️ {temp_c:.1f}°C &nbsp;|&nbsp; {condition}</span>
            <span class="demo-pill">● DATA MODE: DEMO</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_weather_card_html(
    temp_c: float,
    condition_text: str,
    humidity_pct: float,
    wind_kmh: float,
    pressure_hpa: float,
    dew_point_c: Optional[float] = None
) -> str:
    """
    Renders the HTML for the CURRENT WEATHER card.
    """
    dew_str = f"{dew_point_c:.1f} °C" if dew_point_c is not None else "N/A"
    return f"""
    <div class="cc-card">
        <div class="cc-card-header">
            <div class="cc-card-title">🌦️ CURRENT WEATHER</div>
            <div class="cc-card-subtitle">Real-Time Ingestion</div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.6rem;">
            <div class="cc-metric-value">{temp_c:.1f}<span class="cc-metric-unit">°C</span></div>
            <div style="font-weight:700; color:#38bdf8; font-size:0.92rem; text-align:right;">{condition_text}</div>
        </div>
        <div style="margin-top: 0.6rem;">
            <div class="cc-weather-row">
                <span class="cc-weather-label">💧 Relative Humidity</span>
                <span class="cc-weather-val">{humidity_pct:.0f} %</span>
            </div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">💨 Wind Speed</span>
                <span class="cc-weather-val">{wind_kmh:.1f} km/h</span>
            </div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">⏲️ Atmospheric Pressure</span>
                <span class="cc-weather-val">{pressure_hpa:.0f} hPa</span>
            </div>
            <div class="cc-weather-row" style="border-bottom:none;">
                <span class="cc-weather-label">🧊 Dew Point Saturation</span>
                <span class="cc-weather-val">{dew_str}</span>
            </div>
        </div>
    </div>
    """


def render_system_status_card_html(
    test_accuracy_pct: float = 95.4,
    last_updated_time: str = "06:00 IST",
    total_stations: int = 14
) -> str:
    """
    Renders the HTML for the SYSTEM STATUS card.
    """
    return f"""
    <div class="cc-card">
        <div class="cc-card-header">
            <div class="cc-card-title">⚙️ SYSTEM STATUS</div>
            <div class="cc-card-subtitle">Core Telemetry</div>
        </div>
        <div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">Data Connection</span>
                <span class="cc-weather-val" style="color: #4ade80;">● Online (Local Demo)</span>
            </div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">ML Model Status</span>
                <span class="cc-weather-val" style="color: #00f0ff;">Active (XGBoost)</span>
            </div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">Monitored Basins</span>
                <span class="cc-weather-val">{total_stations} Locations</span>
            </div>
            <div class="cc-weather-row">
                <span class="cc-weather-label">Model Test Accuracy</span>
                <span class="cc-weather-val">{test_accuracy_pct:.1f}% (Held-Out Test)</span>
            </div>
            <div class="cc-weather-row" style="border-bottom:none;">
                <span class="cc-weather-label">Operating Mode</span>
                <span class="cc-weather-val" style="color: #fbbf24;">DEMO DATASET</span>
            </div>
        </div>
    </div>
    """

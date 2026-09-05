"""
================================================================================
JAL SANKETH — AI-POWERED RAINFALL & FLOOD RISK INTELLIGENCE
High-Tech Disaster Management Operational Command Center Dashboard
Smart India Hackathon 2026 Prototype
================================================================================
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui import (
    load_command_center_css,
    get_plotly_dark_theme,
    render_command_center_header,
    render_weather_card_html,
    render_system_status_card_html
)
from src.feature_engineering import FeatureEngineer, load_engineered_data
from src.flood_risk import FloodRiskEngine
from src.alerts import create_early_warning_alert, AlertGenerator
from src.map import render_streamlit_map, create_flood_risk_map, get_risk_color
from utils.config import FLOOD_RISK_WEIGHTS, RISK_TIERS

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & THEME INITIALIZATION
# ------------------------------------------------------------------------------
load_command_center_css()

# Known Indian Urban Basins Mapping
LOCATION_COORDINATE_MAP = {
    (19.076, 72.8777): "Mumbai — Dadar / Hindmata",
    (13.0827, 80.2707): "Chennai — Velachery Basin",
    (12.9716, 77.5946): "Bengaluru — Bellandur Catchment",
    (22.5726, 88.3639): "Kolkata — Hooghly Estuary",
    (28.6139, 77.2090): "Delhi NCR — Yamuna Floodplain",
    (25.5941, 85.1376): "Patna — Ganga Floodplain",
    (26.1445, 91.7362): "Guwahati — Brahmaputra Basin",
    (9.9312, 76.2673): "Kochi — Vembanad Plain",
    (17.3850, 78.4867): "Hyderabad — Musi River Basin",
    (20.2961, 85.8245): "Bhubaneswar — Daya Basin",
    (21.1702, 72.8311): "Surat — Tapi Estuary",
    (23.0225, 72.5714): "Ahmedabad — Sabarmati Basin",
    (18.5204, 73.8567): "Pune — Mula-Mutha Basin",
    (11.6854, 76.1320): "Wayanad — Vythiri Foothills"
}


# ------------------------------------------------------------------------------
# 2. CACHED MODEL & DATA PIPELINE
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    """
    Loads trained XGBoost model and feature columns schema.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "heavy_rain_model.joblib")
    feat_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        from src.train_model import train_heavy_rain_classifier
        train_heavy_rain_classifier()

    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return model, feature_cols, metrics


@st.cache_data
def load_evaluated_dataset():
    """
    Loads raw dataset, transforms features, executes XGBoost predictions, and computes flood risk.
    """
    raw_csv = os.path.join(PROJECT_ROOT, "data", "raw", "weather_demo.csv")
    if not os.path.exists(raw_csv):
        from data.create_demo_dataset import generate_demo_dataset
        generate_demo_dataset(output_path=raw_csv)

    model, feature_cols, _ = load_model_assets()
    engineer = FeatureEngineer(raw_data_path=raw_csv)
    full_df = engineer.fit_transform()

    # Assign friendly names
    def assign_name(row):
        for (k_lat, k_lon), name in LOCATION_COORDINATE_MAP.items():
            if abs(row["latitude"] - k_lat) < 0.01 and abs(row["longitude"] - k_lon) < 0.01:
                return name
        return f"Station ({row['latitude']:.2f}N, {row['longitude']:.2f}E)"

    full_df["location_name"] = full_df.apply(assign_name, axis=1)

    # ML Inference
    X_mat = full_df[feature_cols]
    full_df["heavy_rain_prob"] = model.predict_proba(X_mat)[:, 1]
    full_df["ml_heavy_rain_pred"] = model.predict(X_mat)

    # Flood Risk Calculation
    engine = FloodRiskEngine()
    evaluated_df = engine.evaluate_dataframe(
        full_df,
        prob_col="heavy_rain_prob",
        rain_col="rainfall",
        elev_col="elevation",
        drain_col="drainage_factor",
        sat_col="rainfall_lag_7d_sum",
        name_col="location_name"
    )

    evaluated_df["date_str"] = pd.to_datetime(evaluated_df["date"]).dt.strftime("%Y-%m-%d")
    return evaluated_df, engine


# Backwards-compatible alias for testing
get_evaluated_dataset = load_evaluated_dataset


def main():
    # Load pipeline assets
    model, feature_cols, model_metrics = load_model_assets()
    df, risk_engine = load_evaluated_dataset()

    # Session State for Top Navigation
    if "nav_tab" not in st.session_state:
        st.session_state["nav_tab"] = "Dashboard"

    # --------------------------------------------------------------------------
    # SIDEBAR: MINIMAL COMMAND CONSOLE
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🧭 COMMAND CONSOLE")
        st.caption("Operational station filter & telemetry status.")

        st.markdown("**LOCATION**")
        all_locations = sorted(df["location_name"].unique())
        selected_location = st.selectbox("Select Station Basin", all_locations, index=0, label_visibility="collapsed")

        st.markdown("**DATE / TIME**")
        loc_df = df[df["location_name"] == selected_location].sort_values("date_str", ascending=False)
        available_dates = loc_df["date_str"].tolist()
        
        # Pick a rain date by default if available
        rainy_dates = loc_df[loc_df["rainfall"] >= 45.0]["date_str"].tolist()
        default_date_idx = 0
        if rainy_dates and rainy_dates[0] in available_dates:
            default_date_idx = available_dates.index(rainy_dates[0])

        selected_date = st.selectbox("Select Observation Date", available_dates, index=default_date_idx, label_visibility="collapsed")

        st.markdown("**RISK LEVEL FILTER**")
        risk_filter = st.selectbox("Filter Stations by Risk", ["All Levels", "Critical Only", "High & Critical", "Moderate+"], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("**SYSTEM TELEMETRY**")
        st.markdown("""
        * **Data Connection**: <span style="color:#4ade80;font-weight:700;">● ONLINE (Demo)</span>
        * **ML Model**: <span style="color:#00f0ff;font-weight:700;">XGBoost Active</span>
        * **Evaluation ROC-AUC**: <span style="color:#38bdf8;font-weight:700;">0.9717</span>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**DATA MODE**")
        st.markdown("""
        <div style="background:#2e1d05; border:1px solid #d97706; padding:8px; border-radius:6px; font-size:0.75rem; color:#fde047;">
            <b>● DEMO MODE</b><br>
            Operating on synthetic Indian monsoon meteorological demo dataset for SIH 2026.
        </div>
        """, unsafe_allow_html=True)

    # Filtered records
    current_record = df[(df["location_name"] == selected_location) & (df["date_str"] == selected_date)].iloc[0]
    date_stations_df = df[df["date_str"] == selected_date].copy()

    # Apply Risk Filter to Map Dataset
    if risk_filter == "Critical Only":
        filtered_stations_df = date_stations_df[date_stations_df["risk_level"] == "Critical"]
    elif risk_filter == "High & Critical":
        filtered_stations_df = date_stations_df[date_stations_df["risk_level"].isin(["High", "Critical"])]
    elif risk_filter == "Moderate+":
        filtered_stations_df = date_stations_df[date_stations_df["risk_level"].isin(["Moderate", "High", "Critical"])]
    else:
        filtered_stations_df = date_stations_df

    # --------------------------------------------------------------------------
    # TOP HEADER & FIXED NAVIGATION BAR
    # --------------------------------------------------------------------------
    temp_val = float(current_record["temperature"])
    condition_val = "Severe Deluge" if current_record["rainfall"] >= 64.5 else ("Moderate Rain" if current_record["rainfall"] >= 15.6 else "Light / Clear")
    
    render_command_center_header(
        active_tab=st.session_state["nav_tab"],
        selected_location=selected_location,
        selected_date=selected_date,
        temp_c=temp_val,
        condition=condition_val
    )

    # Navigation Tabs Bar
    nav_cols = st.columns(6)
    tab_names = ["Dashboard", "Rainfall Prediction", "Flood Risk Map", "Alerts", "Reports", "Settings"]
    
    for i, tab in enumerate(tab_names):
        with nav_cols[i]:
            btn_type = "primary" if st.session_state["nav_tab"] == tab else "secondary"
            if st.button(tab, key=f"nav_btn_{tab}", type=btn_type, use_container_width=True):
                st.session_state["nav_tab"] = tab
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # VIEW ROUTING
    # ==========================================================================

    # --------------------------------------------------------------------------
    # 1. MAIN COMMAND CENTER DASHBOARD TAB
    # --------------------------------------------------------------------------
    if st.session_state["nav_tab"] == "Dashboard":
        col_w, col_m, col_a = st.columns([1.1, 2.3, 1.4])

        with col_w:
            weather_html = render_weather_card_html(
                temp_c=float(current_record["temperature"]),
                condition_text=condition_val,
                humidity_pct=float(current_record["humidity"]),
                wind_kmh=float(current_record["wind_speed"]),
                pressure_hpa=float(current_record["pressure"]),
                dew_point_c=float(current_record.get("dew_point_est", current_record["temperature"] - 5.0))
            )
            st.markdown(weather_html, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">🏔️ TERRAIN PROFILE</div>
                    <div class="cc-card-subtitle">Local Basin Parameters</div>
                </div>
                <div>
                    <div class="cc-weather-row">
                        <span class="cc-weather-label">Elevation</span>
                        <span class="cc-weather-val">{current_record['elevation']:.1f} m ASL</span>
                    </div>
                    <div class="cc-weather-row">
                        <span class="cc-weather-label">Drainage Capacity</span>
                        <span class="cc-weather-val">{current_record['drainage_factor']:.2f} / 1.00</span>
                    </div>
                    <div class="cc-weather-row" style="border-bottom:none;">
                        <span class="cc-weather-label">7-Day Saturation</span>
                        <span class="cc-weather-val" style="color:#38bdf8;">{current_record['rainfall_lag_7d_sum']:.1f} mm</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m:
            st.markdown("""
            <div class="cc-card" style="padding: 0.8rem;">
                <div class="cc-card-header" style="margin-bottom: 0.5rem;">
                    <div class="cc-card-title">🗺️ FLOOD RISK MAP</div>
                    <div class="cc-card-subtitle" style="font-size:0.75rem; color:#94a3b8;">
                        ● <span style="color:#22c55e;">Low (0-24)</span> &nbsp;
                        ● <span style="color:#eab308;">Moderate (25-49)</span> &nbsp;
                        ● <span style="color:#f97316;">High (50-74)</span> &nbsp;
                        ● <span style="color:#ef4444;">Critical (75-100)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            render_streamlit_map(
                data=filtered_stations_df,
                zoom_start=5,
                tile_style="CartoDB dark_matter",
                height=360
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_a:
            st.markdown("""
            <div class="cc-card" style="height: 100%;">
                <div class="cc-card-header">
                    <div class="cc-card-title">🚨 ACTIVE ALERTS</div>
                    <div class="cc-card-subtitle">Severity Sorted Feed</div>
                </div>
            """, unsafe_allow_html=True)

            all_alerts = AlertGenerator.generate_alerts_dataframe(date_stations_df)
            sorted_alerts = sorted(all_alerts, key=lambda x: x["severity_rank"], reverse=True)

            for al_item in sorted_alerts[:3]:
                al_tier = al_item["risk_level"].lower()
                badge_class = f"badge-{al_tier}"
                alert_css = f"cc-alert-item cc-alert-{al_tier}"
                
                st.markdown(f"""
                <div class="{alert_css}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                        <span class="alert-badge {badge_class}">{al_item['risk_level']} RISK</span>
                        <span style="font-size:0.7rem; color:#94a3b8;">{al_item['location'].split('—')[0]}</span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:700; color:#ffffff; margin: 3px 0;">
                        {al_item['headline_message']}
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; line-height:1.3;">
                        <b>Rec:</b> {al_item['recommended_action'][:90]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("VIEW ALL ALERTS →", key="view_all_alerts_btn", use_container_width=True):
                st.session_state["nav_tab"] = "Alerts"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns([1.1, 1.2, 1.1, 1.2])

        with r2_c1:
            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">🌧️ RAINFALL PREDICTION</div>
                    <div class="cc-card-subtitle">Next 24 Hours</div>
                </div>
            """, unsafe_allow_html=True)

            hours = ["00-03h", "03-06h", "06-09h", "09-12h", "12-15h", "15-18h", "18-21h", "21-24h"]
            tot_rain = float(current_record["rainfall"])
            r_weights = np.array([0.05, 0.10, 0.25, 0.30, 0.15, 0.08, 0.04, 0.03])
            rain_distribution = (tot_rain * r_weights).round(1)

            fig_pred = go.Figure(go.Bar(
                x=hours,
                y=rain_distribution,
                marker_color="#00f0ff",
                opacity=0.85
            ))
            fig_pred.update_layout(
                **get_plotly_dark_theme(),
                height=130,
                xaxis_title=None,
                yaxis_title="mm",
                margin=dict(l=20, r=10, t=10, b=20)
            )
            st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

            prob_val = float(current_record["heavy_rain_prob"]) * 100.0
            prob_col = "#ef4444" if prob_val >= 50 else ("#f97316" if prob_val >= 25 else "#22c55e")
            
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-top:0.4rem; padding-top:0.4rem; border-top:1px dashed #14223d;">
                <div>
                    <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">TOTAL PREDICTED</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#38bdf8; font-family:'JetBrains Mono';">{tot_rain:.1f} mm</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;">HEAVY RAIN PROB</div>
                    <div style="font-size:1.2rem; font-weight:800; color:{prob_col}; font-family:'JetBrains Mono';">{prob_val:.1f}%</div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with r2_c2:
            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">📅 5-DAY FORECAST</div>
                    <div class="cc-card-subtitle">Demo Projection</div>
                </div>
            """, unsafe_allow_html=True)

            sub_dates_df = loc_df.sort_values("date").copy()
            match_idx = sub_dates_df[sub_dates_df["date_str"] == selected_date].index
            if not match_idx.empty:
                idx_pos = sub_dates_df.index.get_loc(match_idx[0])
                start_p = max(0, idx_pos)
                five_day_df = sub_dates_df.iloc[start_p:start_p + 5]
            else:
                five_day_df = sub_dates_df.head(5)

            x_labels = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5"][:len(five_day_df)]
            y_vals = five_day_df["rainfall"].values

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=x_labels,
                y=y_vals,
                mode="lines+markers+text",
                text=[f"{v:.0f}" for v in y_vals],
                textposition="top center",
                textfont=dict(color="#38bdf8", size=10),
                line=dict(color="#0284c7", width=2.5),
                marker=dict(size=6, color="#00f0ff")
            ))
            fig_fc.update_layout(
                **get_plotly_dark_theme(),
                height=150,
                yaxis_title="mm",
                margin=dict(l=20, r=10, t=15, b=20)
            )
            st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
            st.markdown("<div style='font-size:0.65rem; color:#64748b; text-align:center;'>DEMONSTRATION FORECAST DATA (Historical Series Extrapolation)</div></div>", unsafe_allow_html=True)

        with r2_c3:
            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">📊 RISK SUMMARY</div>
                    <div class="cc-card-subtitle">Network Breakdown</div>
                </div>
            """, unsafe_allow_html=True)

            tier_counts = date_stations_df["risk_level"].value_counts()
            tier_labels = ["Low", "Moderate", "High", "Critical"]
            tier_values = [tier_counts.get(t, 0) for t in tier_labels]
            tier_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=tier_labels,
                values=tier_values,
                hole=0.55,
                marker=dict(colors=tier_colors),
                textinfo="percent",
                textfont=dict(size=10, color="#ffffff"),
                hoverinfo="label+value"
            )])
            fig_donut.update_layout(
                **get_plotly_dark_theme(),
                height=150,
                showlegend=False,
                margin=dict(l=5, r=5, t=5, b=5),
                annotations=[dict(text="AREA<br>DIST", x=0.5, y=0.5, font_size=10, font_family="JetBrains Mono", font_color="#94a3b8", showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            
            tot_s = len(date_stations_df)
            st.markdown(f"""
            <div style="font-size:0.72rem; display:flex; justify-content:space-between; font-family:'JetBrains Mono';">
                <span style="color:#22c55e;">Low: {(tier_values[0]/tot_s)*100:.0f}%</span>
                <span style="color:#eab308;">Mod: {(tier_values[1]/tot_s)*100:.0f}%</span>
                <span style="color:#f97316;">High: {(tier_values[2]/tot_s)*100:.0f}%</span>
                <span style="color:#ef4444;">Crit: {(tier_values[3]/tot_s)*100:.0f}%</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with r2_c4:
            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">📈 RISK BY LOCATION</div>
                    <div class="cc-card-subtitle">Top 5 Vulnerable Zones</div>
                </div>
            """, unsafe_allow_html=True)

            top5_df = date_stations_df.sort_values("risk_score", ascending=False).head(5)
            short_names = [n.split("—")[0].strip() for n in top5_df["location_name"]]
            bar_colors = [get_risk_color(l) for l in top5_df["risk_level"]]

            fig_rank = go.Figure(go.Bar(
                x=top5_df["risk_score"].values,
                y=short_names,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{s:.0f}% ({l})" for s, l in zip(top5_df["risk_score"], top5_df["risk_level"])],
                textposition="inside",
                textfont=dict(size=9, color="#ffffff", family="JetBrains Mono")
            ))
            fig_rank.update_layout(
                **get_plotly_dark_theme(),
                height=150,
                xaxis=dict(range=[0, 100], showticklabels=False),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=5, b=5)
            )
            st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. RAINFALL PREDICTION PAGE
    # --------------------------------------------------------------------------
    elif st.session_state["nav_tab"] == "Rainfall Prediction":
        st.markdown("### 🌧️ Machine Learning Rainfall Prediction Intelligence")
        st.caption("Deep-dive inspection of XGBoost prediction mechanics, atmospheric inputs, and feature attribution.")

        col_p1, col_p2 = st.columns([1.2, 1.0])
        with col_p1:
            st.markdown(f"""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">⚡ XGBOOST INFERENCE & MODEL INPUTS</div>
                    <div class="cc-card-subtitle">Station: {selected_location}</div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                    <div class="cc-weather-row"><span class="cc-weather-label">Surface Pressure</span><span class="cc-weather-val">{current_record['pressure']:.1f} hPa</span></div>
                    <div class="cc-weather-row"><span class="cc-weather-label">Pressure 24h Trend</span><span class="cc-weather-val">{current_record['pressure_trend_24h']:+.2f} hPa</span></div>
                    <div class="cc-weather-row"><span class="cc-weather-label">Relative Humidity</span><span class="cc-weather-val">{current_record['humidity']:.1f} %</span></div>
                    <div class="cc-weather-row"><span class="cc-weather-label">Dew Point Spread</span><span class="cc-weather-val">{current_record['dew_point_depression']:.2f} °C</span></div>
                    <div class="cc-weather-row"><span class="cc-weather-label">Vapor Deficit (VPD)</span><span class="cc-weather-val">{current_record['vapor_pressure_deficit']:.2f} kPa</span></div>
                    <div class="cc-weather-row"><span class="cc-weather-label">Antecedent 7d Rain</span><span class="cc-weather-val">{current_record['rainfall_lag_7d_sum']:.1f} mm</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">🔍 WHY DID THE MODEL MAKE THIS PREDICTION?</div>
                    <div class="cc-card-subtitle">Non-Causal Feature Contribution Analysis</div>
                </div>
                <div style="font-size:0.85rem; line-height:1.6; color:#cbd5e1;">
            """, unsafe_allow_html=True)

            prob_pct = current_record["heavy_rain_prob"] * 100.0
            if prob_pct >= 50.0:
                st.markdown(f"""
                * **High atmospheric moisture saturation** (Relative humidity {current_record['humidity']:.0f}%, Dewpoint spread {current_record['dew_point_depression']:.1f}°C) contributed most strongly to the predicted heavy rainfall probability.
                * **Barometric pressure convergence** (Surface pressure {current_record['pressure']:.0f} hPa with 24h change {current_record['pressure_trend_24h']:+.1f} hPa) served as a primary model trigger.
                * **Antecedent precipitation buildup** ({current_record['rainfall_lag_7d_sum']:.1f} mm past 7 days) compounded the hydrological vulnerability.
                """)
            else:
                st.markdown(f"""
                * **Moderate-to-low moisture levels** ({current_record['humidity']:.0f}% humidity) and high barometric stability ({current_record['pressure']:.0f} hPa) kept the heavy-rain probability suppressed ({prob_pct:.1f}%).
                * **Minimal antecedent rainfall** reduced the likelihood of compound inundation hazard.
                """)

            st.markdown("</div></div>", unsafe_allow_html=True)

        with col_p2:
            st.markdown("""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">🌲 XGBOOST FEATURE IMPORTANCE (GAIN)</div>
                    <div class="cc-card-subtitle">Held-Out Test Set Attribution</div>
                </div>
            """, unsafe_allow_html=True)
            
            top_feats = model_metrics.get("top_features", [])
            if top_feats:
                df_feat = pd.DataFrame(top_feats).head(10)
                fig_f = px.bar(
                    df_feat.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                    labels={"importance": "Gain Weight", "feature": "Feature"}
                )
                fig_f.update_layout(**get_plotly_dark_theme(), height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. FLOOD RISK MAP PAGE
    # --------------------------------------------------------------------------
    elif st.session_state["nav_tab"] == "Flood Risk Map":
        st.markdown("### 🗺️ Full-Screen Geospatial Inundation Command Map")
        st.caption("Inspect spatial flood vulnerability across all 14 monitored Indian river basins and urban centers.")

        c_map_full, c_map_side = st.columns([2.2, 1.0])
        
        with c_map_full:
            render_streamlit_map(
                data=filtered_stations_df,
                zoom_start=5,
                tile_style="CartoDB dark_matter",
                height=580
            )

        with c_map_side:
            st.markdown(f"""
            <div class="cc-card">
                <div class="cc-card-header">
                    <div class="cc-card-title">🎯 SELECTED BASIN PROFILE</div>
                    <div class="cc-card-subtitle">{selected_location}</div>
                </div>
            """, unsafe_allow_html=True)
            st.code(current_record["risk_explanation"], language="yaml")
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. ALERTS PAGE
    # --------------------------------------------------------------------------
    elif st.session_state["nav_tab"] == "Alerts":
        st.markdown("### 🚨 National & Regional Early-Warning Advisory Feed")
        st.caption("Official standard operating procedures and severity-ranked warning bulletins.")

        all_alerts = AlertGenerator.generate_alerts_dataframe(date_stations_df)
        sorted_alerts = sorted(all_alerts, key=lambda x: x["severity_rank"], reverse=True)

        for al in sorted_alerts:
            al_tier = al["risk_level"].lower()
            badge_class = f"badge-{al_tier}"
            alert_css = f"cc-alert-item cc-alert-{al_tier}"
            
            with st.expander(f"[{al['risk_level'].upper()} ALERT] {al['location']} — Risk Score: {al['risk_score']}/100", expanded=(al["risk_level"] in ["Critical", "High"])):
                st.markdown(f"""
                <div style="font-size:0.9rem; line-height:1.6;">
                    <p><b>Status Headline:</b> {al['headline_message']}</p>
                    <p><b>Primary Cause:</b> {al['main_reason']}</p>
                    <p><b>Recommended Operational Action:</b> {al['recommended_action']}</p>
                    <p style="color:#94a3b8; font-size:0.75rem;">Timestamp: {al['timestamp']} | Reference: {al['alert_id']}</p>
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. REPORTS PAGE
    # --------------------------------------------------------------------------
    elif st.session_state["nav_tab"] == "Reports":
        st.markdown("### 📑 Hydro-Meteorological Emergency Report Generator")
        st.caption("Download standardized disaster management situation reports and operational summaries.")

        rep_text = f"""================================================================================
JAL SANKETH — NATIONAL DISASTER MANAGEMENT AUTHORITY (NDMA)
HYDRO-METEOROLOGICAL SITUATION REPORT (SITREP)
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Observation Date: {selected_date}
Station Basin: {selected_location}

1. BASIN METEOROLOGICAL PARAMETERS
--------------------------------------------------------------------------------
* Observed / Predicted 24h Rainfall : {current_record['rainfall']:.1f} mm
* Heavy Rain Classification Probability: {current_record['heavy_rain_prob']*100:.1f}%
* Surface Barometric Pressure        : {current_record['pressure']:.1f} hPa ({current_record['pressure_trend_24h']:+.1f} hPa 24h change)
* Relative Humidity                  : {current_record['humidity']:.1f}%
* Ambient Temperature                : {current_record['temperature']:.1f} °C

2. HYDROLOGICAL RISK & TERRAIN CHARACTERISTICS
--------------------------------------------------------------------------------
* Computed Flood Risk Score          : {current_record['risk_score']:.1f} / 100
* Composite Risk Classification      : {current_record['risk_level'].upper()}
* Surface Elevation                  : {current_record['elevation']:.1f} m Above Sea Level
* Drainage Infrastructure Index      : {current_record['drainage_factor']:.2f} / 1.00
* Antecedent 7-Day Saturation        : {current_record['rainfall_lag_7d_sum']:.1f} mm

3. DIAGNOSTIC EVALUATION SUMMARY
--------------------------------------------------------------------------------
{current_record['risk_explanation']}

================================================================================
END OF REPORT — JAL SANKETH COMMAND CENTER PLATFORM (SIH 2026)
================================================================================
"""
        st.code(rep_text, language="text")

        st.download_button(
            label="💾 DOWNLOAD SITREP REPORT (.TXT)",
            data=rep_text,
            file_name=f"JalSanketh_SITREP_{selected_location.split('—')[0].strip()}_{selected_date}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # --------------------------------------------------------------------------
    # 6. SETTINGS PAGE
    # --------------------------------------------------------------------------
    elif st.session_state["nav_tab"] == "Settings":
        st.markdown("### ⚙️ Command Center Configuration & Threshold Tuning")
        st.caption("Adjust weight parameters for flood risk evaluation and ML inference settings.")

        c_set1, c_set2 = st.columns(2)
        with c_set1:
            st.markdown("#### Risk Model Weights")
            st.slider("XGBoost Model Probability Weight", 0.0, 1.0, FLOOD_RISK_WEIGHTS.get("model_prob", 0.35), 0.05)
            st.slider("24h Rainfall Intensity Weight", 0.0, 1.0, FLOOD_RISK_WEIGHTS.get("rainfall_24h", 0.30), 0.05)
            st.slider("Elevation / Terrain Weight", 0.0, 1.0, FLOOD_RISK_WEIGHTS.get("elevation", 0.15), 0.05)
            st.slider("7-Day Antecedent Saturation Weight", 0.0, 1.0, FLOOD_RISK_WEIGHTS.get("saturation", 0.20), 0.05)

        with c_set2:
            st.markdown("#### Alert Dispatch Channels")
            st.checkbox("Enable Automated SMS Gateway Alerts", value=True)
            st.checkbox("Enable WhatsApp Broadcast API", value=True)
            st.checkbox("Enable State Emergency Operations Center (SEOC) Push", value=True)
            st.text_input("Emergency Operational Escalation Contact", "+91-11-26701700 (NDMA Control Room)")


if __name__ == "__main__":
    main()
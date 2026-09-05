
import os
import sys
import json
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FeatureEngineer
from src.flood_risk import FloodRiskEngine
from src.map import is_in_telangana

TELANGANA_STATIONS = [
    {'name': 'Hyderabad — Musi River Basin', 'lat': 17.3850, 'lon': 78.4867, 'elev': 505.0, 'drain': 0.38},
    {'name': 'Warangal — Kakatiya Basin', 'lat': 17.9689, 'lon': 79.5941, 'elev': 302.0, 'drain': 0.42},
    {'name': 'Karimnagar — Lower Manair Catchment', 'lat': 18.4386, 'lon': 79.1288, 'elev': 265.0, 'drain': 0.45},
    {'name': 'Nizamabad — Godavari Basin / Alisagar', 'lat': 18.6725, 'lon': 78.0941, 'elev': 395.0, 'drain': 0.40},
    {'name': 'Khammam — Munneru River Basin', 'lat': 17.2473, 'lon': 80.1514, 'elev': 112.0, 'drain': 0.35},
    {'name': 'Bhadrachalam — Godavari Lowland', 'lat': 17.6689, 'lon': 80.8933, 'elev': 55.0, 'drain': 0.28},
    {'name': 'Nalgonda — Dindi / Krishna Catchment', 'lat': 17.0575, 'lon': 79.2684, 'elev': 240.0, 'drain': 0.46},
    {'name': 'Mahabubnagar — Palamuru / Krishna Basin', 'lat': 16.7488, 'lon': 77.9840, 'elev': 498.0, 'drain': 0.48},
    {'name': 'Adilabad — Penganga Basin', 'lat': 19.6641, 'lon': 78.5320, 'elev': 264.0, 'drain': 0.44},
    {'name': 'Nirmal — Kadem Reservoir Catchment', 'lat': 19.0964, 'lon': 78.3429, 'elev': 340.0, 'drain': 0.36},
    {'name': 'Mancherial — Pranahita-Godavari Valley', 'lat': 18.8679, 'lon': 79.4639, 'elev': 150.0, 'drain': 0.34},
    {'name': 'Peddapalli — Ramagundam Godavari Basin', 'lat': 18.7557, 'lon': 79.5130, 'elev': 165.0, 'drain': 0.39},
    {'name': 'Jagtial — SRSP Godavari Downstream', 'lat': 18.7946, 'lon': 78.9126, 'elev': 293.0, 'drain': 0.43},
    {'name': 'Siddipet — Komati Cheruvu Catchment', 'lat': 18.1018, 'lon': 78.8520, 'elev': 675.0, 'drain': 0.50},
    {'name': 'Medak — Manjeera River Basin', 'lat': 18.0450, 'lon': 78.2630, 'elev': 488.0, 'drain': 0.41},
    {'name': 'Sangareddy — Singur Dam Catchment', 'lat': 17.6190, 'lon': 78.0810, 'elev': 496.0, 'drain': 0.44},
    {'name': 'Suryapet — Musi-Krishna Confluence', 'lat': 17.1439, 'lon': 79.6239, 'elev': 218.0, 'drain': 0.42},
    {'name': 'Vikarabad — Ananthagiri Hills Catchment', 'lat': 17.3366, 'lon': 77.9048, 'elev': 652.0, 'drain': 0.55},
    {'name': 'Nagarkurnool — Dindi River Catchment', 'lat': 16.4854, 'lon': 78.3338, 'elev': 460.0, 'drain': 0.47},
    {'name': 'Wanaparthy — Sarala Sagar Catchment', 'lat': 16.3624, 'lon': 78.0628, 'elev': 380.0, 'drain': 0.46}
]

def resolve_forecast_csv_path(path: Optional[str] = None) -> str:
    if path and os.path.exists(path):
        return path
    candidates = [
        os.path.join(PROJECT_ROOT, 'data', 'weather_forecast.csv'),
        os.path.join(PROJECT_ROOT, 'weather_forecast.csv'),
        os.path.join(PROJECT_ROOT, 'data', 'raw', 'weather_forecast.csv')
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('weather_forecast.csv not found')

def load_raw_forecast_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    resolved_path = resolve_forecast_csv_path(csv_path)
    df = pd.read_csv(resolved_path)
    col_map = {
        'time': 'datetime',
        'temperature_2m': 'temperature',
        'relative_humidity_2m': 'humidity',
        'precipitation': 'rainfall',
        'pressure_msl': 'pressure',
        'wind_speed_10m': 'wind_speed',
        'soil_moisture_0_to_7cm': 'soil_moisture'
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    num_cols = ['temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median() if not df[c].dropna().empty else 0.0)
    return df

def aggregate_forecast_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        'temperature': 'mean',
        'humidity': 'mean',
        'rainfall': 'sum',
        'pressure': 'mean',
        'wind_speed': 'mean'
    }
    if 'soil_moisture' in hourly_df.columns:
        agg_dict['soil_moisture'] = 'mean'
    daily_df = hourly_df.groupby('date').agg(agg_dict).reset_index()
    return daily_df

import requests

def fetch_live_station_forecast(lat: float, lon: float, timeout: int = 8) -> Optional[pd.DataFrame]:
    """
    Fetches real-time 7-day hourly forecast from Open-Meteo REST API.
    Returns cleaned hourly DataFrame or None if offline / timed out.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "soil_moisture_0_to_7cm"
        ],
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if "hourly" in data and data["hourly"]:
                df = pd.DataFrame(data["hourly"])
                col_map = {
                    'time': 'datetime',
                    'temperature_2m': 'temperature',
                    'relative_humidity_2m': 'humidity',
                    'precipitation': 'rainfall',
                    'pressure_msl': 'pressure',
                    'wind_speed_10m': 'wind_speed',
                    'soil_moisture_0_to_7cm': 'soil_moisture'
                }
                df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
                num_cols = ['temperature', 'humidity', 'rainfall', 'pressure', 'wind_speed']
                for c in num_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
                return df
    except Exception as e:
        logger.warning(f"Live Open-Meteo fetch failed ({lat}, {lon}): {e}")
    return None


def process_forecast_for_telangana(
    csv_path: Optional[str] = None,
    stations: Optional[List[Dict[str, Any]]] = None,
    hourly_override_df: Optional[pd.DataFrame] = None,
    custom_weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_path = os.path.join(PROJECT_ROOT, 'models', 'heavy_rain_model.joblib')
    feat_path = os.path.join(PROJECT_ROOT, 'models', 'feature_columns.json')
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        from src.train_model import train_heavy_rain_classifier
        train_heavy_rain_classifier()
    model = joblib.load(model_path)
    with open(feat_path, 'r') as f:
        feature_cols = json.load(f)

    if hourly_override_df is not None and not hourly_override_df.empty:
        hourly_df = hourly_override_df.copy()
    else:
        hourly_df = load_raw_forecast_csv(csv_path)

    base_daily = aggregate_forecast_daily(hourly_df)
    target_stations = stations or TELANGANA_STATIONS
    all_station_records = []

    for stn in target_stations:
        lat = stn['lat']
        lon = stn['lon']
        if not is_in_telangana(lat, lon):
            continue
        stn_daily = base_daily.copy()
        stn_daily['latitude'] = round(lat, 4)
        stn_daily['longitude'] = round(lon, 4)
        stn_daily['elevation'] = round(stn['elev'], 1)
        stn_daily['drainage_factor'] = round(stn['drain'], 2)
        stn_daily['location_name'] = stn['name']
        stn_daily['heavy_rain'] = (stn_daily['rainfall'] >= 64.5).astype(int)

        elev_diff = stn['elev'] - 505.0
        stn_daily['temperature'] = (stn_daily['temperature'] - (elev_diff / 1000.0) * 6.5).round(2)
        stn_daily['pressure'] = (stn_daily['pressure'] - (elev_diff / 8.5)).round(2)
        all_station_records.append(stn_daily)

    full_forecast_df = pd.concat(all_station_records, ignore_index=True)
    engineer = FeatureEngineer()
    engineered_df = engineer.fit_transform(full_forecast_df)
    X_mat = engineered_df[feature_cols]
    engineered_df['heavy_rain_prob'] = model.predict_proba(X_mat)[:, 1]
    engineered_df['ml_heavy_rain_pred'] = model.predict(X_mat)

    risk_engine = FloodRiskEngine(custom_weights=custom_weights)
    evaluated_df = risk_engine.evaluate_dataframe(
        engineered_df,
        prob_col='heavy_rain_prob',
        rain_col='rainfall',
        elev_col='elevation',
        drain_col='drainage_factor',
        sat_col='rainfall_lag_7d_sum',
        name_col='location_name'
    )
    evaluated_df['date_str'] = pd.to_datetime(evaluated_df['date']).dt.strftime('%Y-%m-%d')
    return evaluated_df, hourly_df

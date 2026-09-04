"""
Preprocessing and Feature Engineering Module for Jal Sanketh
Cleans data, builds temporal lag features, atmospheric indices, and prepares ML datasets.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List

# Core ML feature set for XGBoost
FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "wind_direction_deg",
    "dew_point_c",
    "cloud_cover_pct",
    "rain_lag_1d",
    "rain_lag_3d",
    "pressure_diff_24h",
    "dew_point_spread",
    "month_sin",
    "month_cos",
    "elevation_m"
]

TARGET_COLUMN = "rainfall_amount_mm"
CLASSIFICATION_TARGET = "heavy_rain_flag"


def load_raw_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw weather time-series and location profiles.
    """
    weather_path = os.path.join(data_dir, "weather_historical.csv")
    geo_path = os.path.join(data_dir, "location_geo_profiles.csv")
    
    if not os.path.exists(weather_path) or not os.path.exists(geo_path):
        raise FileNotFoundError(f"Missing raw CSV files in {data_dir}. Please run data_generator.py first.")
        
    weather_df = pd.read_csv(weather_path)
    geo_df = pd.read_csv(geo_path)
    return weather_df, geo_df


def engineer_features(weather_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, temporal lag calculations, and atmospheric feature engineering.
    """
    # 1. Ensure datetime formatting and sorting
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["location_id", "date"]).reset_index(drop=True)
    
    # 2. Merge geographical attributes
    df = df.merge(geo_df, on="location_id", how="left")
    
    # 3. Lag and Rolling features per location group
    lagged_dfs = []
    for loc_id, group in df.groupby("location_id"):
        g = group.copy().sort_values("date")
        
        # 1-day and 3-day rainfall lags
        g["rain_lag_1d"] = g["rainfall_amount_mm"].shift(1)
        g["rain_lag_3d"] = g["rainfall_amount_mm"].shift(1).rolling(window=3, min_periods=1).mean()
        
        # 24-hour pressure tendency (cyclonic depression indicator)
        g["pressure_diff_24h"] = g["pressure_hpa"] - g["pressure_hpa"].shift(1)
        
        # Temperature lag
        g["temp_lag_1d"] = g["temperature_c"].shift(1)
        
        lagged_dfs.append(g)
        
    df = pd.concat(lagged_dfs, ignore_index=True)
    
    # 4. Atmospheric Saturation / Dew Point Spread
    df["dew_point_spread"] = np.maximum(df["temperature_c"] - df["dew_point_c"], 0.0)
    
    # 5. Cyclical Seasonality Encodings (Month)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    # 6. Fill or drop initial lag boundary NaNs
    df["rain_lag_1d"] = df["rain_lag_1d"].fillna(0.0)
    df["rain_lag_3d"] = df["rain_lag_3d"].fillna(0.0)
    df["pressure_diff_24h"] = df["pressure_diff_24h"].fillna(0.0)
    df["temp_lag_1d"] = df["temp_lag_1d"].fillna(df["temperature_c"])
    
    return df


def prepare_and_save_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline that saves the cleaned merged dataset.
    """
    os.makedirs(processed_dir, exist_ok=True)
    weather_df, geo_df = load_raw_data(raw_dir)
    processed_df = engineer_features(weather_df, geo_df)
    
    output_path = os.path.join(processed_dir, "merged_training_data.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"[+] Processed training dataset saved to {output_path} ({len(processed_df)} rows, {len(processed_df.columns)} columns)")
    return processed_df


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts feature matrix X, regression target y_reg, and classification target y_clf.
    """
    X = df[FEATURE_COLUMNS].copy()
    y_reg = df[TARGET_COLUMN].copy()
    y_clf = df[CLASSIFICATION_TARGET].copy()
    return X, y_reg, y_clf


if __name__ == "__main__":
    prepare_and_save_data()

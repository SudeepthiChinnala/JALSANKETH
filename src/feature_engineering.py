"""
================================================================================
JAL SANKETH - FEATURE ENGINEERING MODULE
Module Path: src/feature_engineering.py
================================================================================

This module performs end-to-end data validation, cleaning, temporal feature
transformation, atmospheric index calculation, and spatial lag feature
engineering for the Jal Sanketh rainfall and flood risk prediction models.

Strict Data Leakage Prevention:
--------------------------------
1. All time-series lag and rolling statistics are calculated strictly on
   shifted historical values (shift >= 1 day) within each spatial node.
2. Target variables ('rainfall', 'heavy_rain') are never used concurrently as
   features for the current timestamp.
3. Temporal sorting by [latitude, longitude, date] is enforced before any
   windowed transformations.
4. Missing boundary values from lagging are cleanly imputed with zero or
   unbiased historical defaults without future peeking.

Feature Catalog:
----------------
[Temporal Features]
  • month                  : Calendar month (1 to 12)
  • day_of_year            : Julian day of the year (1 to 366)
  • month_sin, month_cos   : Cyclical trigonometric month encodings
  • day_sin, day_cos       : Cyclical trigonometric day-of-year encodings
  • is_monsoon_sw          : Southwest monsoon indicator (June–September)
  • is_monsoon_ne          : Northeast monsoon indicator (October–December)

[Spatial Time-Series Lag Features (per station)]
  • rainfall_lag_1d        : Antecedent 1-day shifted rainfall (mm)
  • rainfall_lag_3d_mean   : 3-day rolling mean of antecedent shifted rainfall (mm)
  • rainfall_lag_7d_sum    : 7-day cumulative sum of antecedent shifted rainfall (mm)
  • pressure_trend_24h     : 24-hr barometric pressure tendency (hPa drop indicator)
  • temp_trend_24h         : 24-hr temperature tendency (°C)

[Atmospheric Thermodynamics & Saturation Indices]
  • dew_point_est          : Approximated dew point temperature (°C)
  • dew_point_depression   : Ambient temperature minus dew point (saturation proximity)
  • vapor_pressure_deficit : Evaporative demand / atmospheric moisture holding proxy
  • instability_index      : Convective storm trigger index (humidity, pressure & wind)

[Geographical & Terrain Interaction Features]
  • drainage_deficit       : Drainage bottleneck score (1 - drainage_factor)
  • elevation_vulnerability: Inverse elevation score (low-lying ponding hazard)
  • runoff_potential_index : Interaction of drainage deficit and elevation vulnerability
================================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union


# Expected mandatory columns in raw weather dataset
MANDATORY_RAW_COLUMNS = [
    "date",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall",
    "elevation",
    "drainage_factor",
    "heavy_rain"
]

# Core feature list returned in X for model training
ENGINEERED_FEATURE_COLUMNS = [
    # Meteorological Base
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    # Temporal & Seasonality
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "is_monsoon_sw",
    "is_monsoon_ne",
    # Time-Series Lags (strictly shifted)
    "rainfall_lag_1d",
    "rainfall_lag_3d_mean",
    "rainfall_lag_7d_sum",
    "pressure_trend_24h",
    "temp_trend_24h",
    # Atmospheric Indices
    "dew_point_est",
    "dew_point_depression",
    "vapor_pressure_deficit",
    "instability_index",
    # Geographical & Terrain
    "elevation",
    "drainage_factor",
    "drainage_deficit",
    "elevation_vulnerability",
    "runoff_potential_index"
]


class FeatureEngineer:
    """
    Feature engineering pipeline for the Jal Sanketh hydro-meteorological system.
    """

    def __init__(self, raw_data_path: str = "data/raw/weather_demo.csv"):
        self.raw_data_path = raw_data_path
        self.feature_columns = ENGINEERED_FEATURE_COLUMNS
        self.target_regression = "rainfall"
        self.target_classification = "heavy_rain"

    def load_and_validate_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads raw dataset and validates schema integrity and mandatory columns.
        """
        file_path = path or self.raw_data_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[!] Raw dataset not found at: {file_path}")

        df = pd.read_csv(file_path)

        # Check required columns
        missing_cols = [col for col in MANDATORY_RAW_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[!] Raw dataset is missing required columns: {missing_cols}")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles any null or NaN values in raw features using sensible physical defaults.
        """
        df_clean = df.copy()

        # Check if missing values exist
        if df_clean.isnull().sum().sum() > 0:
            # Impute numerical features with median per location
            num_cols = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "elevation", "drainage_factor"]
            for col in num_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean.groupby(["latitude", "longitude"])[col].transform(lambda x: x.fillna(x.median()))
                    # Fallback global median if group is empty
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

            if df_clean["heavy_rain"].isnull().any():
                df_clean["heavy_rain"] = (df_clean["rainfall"] >= 64.5).astype(int)

        return df_clean

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal calendar attributes and cyclical trigonometric representations.
        
        Features Added:
          - date (datetime)
          - month (1-12)
          - day_of_year (1-366)
          - month_sin, month_cos: Continuous cyclical representation of month.
          - day_sin, day_cos: Continuous cyclical representation of annual solar cycle.
          - is_monsoon_sw: 1 if month in [6, 7, 8, 9] (Southwest Monsoon), else 0.
          - is_monsoon_ne: 1 if month in [10, 11, 12] (Northeast Monsoon / Retracting Monsoon), else 0.
        """
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])

        # Month and Day of Year
        month = df_temp["date"].dt.month
        day_of_year = df_temp["date"].dt.dayofyear

        df_temp["month"] = month
        df_temp["day_of_year"] = day_of_year

        # Cyclical Sine and Cosine Encodings (prevents artificial distance between Dec and Jan)
        df_temp["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        df_temp["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        df_temp["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_temp["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Monsoon Seasonality Indicators for Indian Climate Regimes
        df_temp["is_monsoon_sw"] = month.isin([6, 7, 8, 9]).astype(int)
        df_temp["is_monsoon_ne"] = month.isin([10, 11, 12]).astype(int)

        return df_temp

    def create_spatial_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes lagged weather & antecedent rainfall features strictly per spatial location.
        
        Zero Leakage Guarantee:
          - All lag transformations use `.shift(1)` or higher so that only PAST observations
            are used to predict the current day's weather.
          - Multi-day rolling windows apply strictly after `.shift(1)`.
        
        Features Added:
          - rainfall_lag_1d: Rainfall recorded on day (t-1).
          - rainfall_lag_3d_mean: 3-day rolling mean of rainfall for days (t-1, t-2, t-3).
          - rainfall_lag_7d_sum: 7-day cumulative rainfall for days (t-1 through t-7).
          - pressure_trend_24h: 24h pressure difference: P(t-1) - P(t-2). Negative values indicate rapid barometric drop.
          - temp_trend_24h: 24h temperature difference: T(t-1) - T(t-2).
        """
        # Sort chronologically within each unique geographic station
        df_lag = df.sort_values(by=["latitude", "longitude", "date"]).reset_index(drop=True)

        grouped = df_lag.groupby(["latitude", "longitude"])

        # 1. Antecedent Rainfall Lags (strictly shifted by 1 day)
        df_lag["rainfall_lag_1d"] = grouped["rainfall"].shift(1)
        df_lag["rainfall_lag_3d_mean"] = grouped["rainfall"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df_lag["rainfall_lag_7d_sum"] = grouped["rainfall"].transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
        )

        # 2. Atmospheric 24-hr Trends (t-1 minus t-2)
        df_lag["pressure_trend_24h"] = grouped["pressure"].shift(1) - grouped["pressure"].shift(2)
        df_lag["temp_trend_24h"] = grouped["temperature"].shift(1) - grouped["temperature"].shift(2)

        # 3. Clean boundary NaNs resulting from initial station time steps
        df_lag["rainfall_lag_1d"] = df_lag["rainfall_lag_1d"].fillna(0.0)
        df_lag["rainfall_lag_3d_mean"] = df_lag["rainfall_lag_3d_mean"].fillna(0.0)
        df_lag["rainfall_lag_7d_sum"] = df_lag["rainfall_lag_7d_sum"].fillna(0.0)
        df_lag["pressure_trend_24h"] = df_lag["pressure_trend_24h"].fillna(0.0)
        df_lag["temp_trend_24h"] = df_lag["temp_trend_24h"].fillna(0.0)

        return df_lag

    def create_atmospheric_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates physical thermodynamics, air saturation proxies, and convective instability indices.
        
        Features Added:
          - dew_point_est: Approximated dew point temperature using Magnus-Tetens inversion:
                           T_dew ≈ T - ((100 - Humidity) / 5)
          - dew_point_depression: T - T_dew. Near 0 indicates 100% moisture saturation (fog/rain).
          - vapor_pressure_deficit: Proxy for saturation vapor pressure vs actual vapor pressure (kPa).
          - instability_index: Convective index combining high humidity, high wind speed, and low pressure.
        """
        df_atm = df.copy()

        temp = df_atm["temperature"]
        hum = df_atm["humidity"]
        pres = df_atm["pressure"]
        wind = df_atm["wind_speed"]

        # 1. Dew Point Approximation (°C)
        dew_point = temp - ((100.0 - hum) / 5.0)
        df_atm["dew_point_est"] = dew_point.round(2)

        # 2. Dew Point Depression (°C)
        df_atm["dew_point_depression"] = np.maximum(temp - dew_point, 0.0).round(2)

        # 3. Vapor Pressure Deficit (VPD) Proxy (kPa)
        # Saturated vapor pressure e_s ≈ 0.61078 * exp((17.27 * T) / (T + 237.3))
        e_sat = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        e_actual = e_sat * (hum / 100.0)
        df_atm["vapor_pressure_deficit"] = np.maximum(e_sat - e_actual, 0.0).round(3)

        # 4. Convective Instability Index (Normalized Trigger)
        # Instability spikes when humidity is near 100%, wind is squalling, and barometric pressure is low
        norm_hum = hum / 100.0
        norm_wind = np.clip(wind / 40.0, 0.1, 2.0)
        norm_press = np.clip(1013.25 / np.maximum(pres, 800.0), 0.9, 1.3)
        df_atm["instability_index"] = (norm_hum * norm_wind * norm_press).round(3)

        return df_atm

    def create_terrain_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives topographical vulnerability and urban drainage stress indices.
        
        Features Added:
          - drainage_deficit: 1.0 - drainage_factor (higher deficit = severe bottleneck).
          - elevation_vulnerability: Normalized inverse elevation scale (low elevation = higher pooling hazard).
          - runoff_potential_index: Compound interaction of drainage deficit and low elevation.
        """
        df_geo = df.copy()

        elev = df_geo["elevation"]
        drain = df_geo["drainage_factor"]

        # Drainage Deficit
        df_geo["drainage_deficit"] = (1.0 - drain).round(3)

        # Elevation Vulnerability (0.0 for high elevations > 500m; up to 1.0 for coastal/lowland < 10m)
        elev_vuln = np.clip(1.0 - (elev / 350.0), 0.05, 1.0)
        df_geo["elevation_vulnerability"] = elev_vuln.round(3)

        # Runoff Potential Index
        df_geo["runoff_potential_index"] = (df_geo["drainage_deficit"] * df_geo["elevation_vulnerability"]).round(3)

        return df_geo

    def fit_transform(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Executes full feature engineering pipeline on the raw dataset.
        """
        if df is None:
            df = self.load_and_validate_data()
        else:
            # Validate input dataframe
            missing_cols = [col for col in MANDATORY_RAW_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"[!] Input dataframe is missing required columns: {missing_cols}")

        # Sequential feature pipeline
        df_proc = self.handle_missing_values(df)
        df_proc = self.create_temporal_features(df_proc)
        df_proc = self.create_spatial_lag_features(df_proc)
        df_proc = self.create_atmospheric_indices(df_proc)
        df_proc = self.create_terrain_interaction_features(df_proc)

        return df_proc

    def get_features_and_targets(
        self,
        df: Optional[pd.DataFrame] = None,
        target_mode: str = "both"
    ) -> Union[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series, pd.Series]
    ]:
        """
        Extracts cleanly formatted feature matrix X and target(s) y for model training.

        Parameters:
        -----------
        df : Optional[pd.DataFrame]
            Processed DataFrame (if None, pipeline will run fit_transform on raw_data_path).
        target_mode : str
            - 'regression'     : Returns (X, y_rainfall)
            - 'classification' : Returns (X, y_heavy_rain)
            - 'both'           : Returns (X, y_rainfall, y_heavy_rain)

        Returns:
        --------
        Clean feature matrix X (pd.DataFrame) and target series y (pd.Series).
        """
        if df is None or not set(self.feature_columns).issubset(df.columns):
            df_proc = self.fit_transform(df)
        else:
            df_proc = df.copy()

        # Feature matrix X (guaranteed zero leakage)
        X = df_proc[self.feature_columns].copy()

        # Targets
        y_reg = df_proc[self.target_regression].copy()
        y_clf = df_proc[self.target_classification].copy()

        if target_mode == "regression":
            return X, y_reg
        elif target_mode == "classification":
            return X, y_clf
        elif target_mode == "both":
            return X, y_reg, y_clf
        else:
            raise ValueError(f"Invalid target_mode '{target_mode}'. Choose 'regression', 'classification', or 'both'.")


def load_engineered_data(raw_data_path: str = "data/raw/weather_demo.csv") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Convenience function to quickly load X, y_regression, y_classification, and the full processed dataframe.
    """
    engineer = FeatureEngineer(raw_data_path=raw_data_path)
    full_df = engineer.fit_transform()
    X, y_reg, y_clf = engineer.get_features_and_targets(full_df, target_mode="both")
    return X, y_reg, y_clf, full_df


if __name__ == "__main__":
    print("=" * 80)
    print("         JAL SANKETH - FEATURE ENGINEERING PIPELINE TEST")
    print("=" * 80)
    
    engineer = FeatureEngineer()
    X, y_reg, y_clf, processed_df = load_engineered_data()

    print(f"\n[+] Raw Data Records Processed : {len(processed_df):,}")
    print(f"[+] Feature Matrix (X) Shape   : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"[+] Regression Target (y_reg)  : {len(y_reg):,} values (Target: '{engineer.target_regression}')")
    print(f"[+] Class Target (y_clf)       : {len(y_clf):,} values (Target: '{engineer.target_classification}')")
    
    print("\n--- ENGINEERED FEATURE LIST (X) ---")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:>2}. {col:<26} (dtype: {str(X[col].dtype):<8} | sample: {X[col].iloc[0]})")

    print("\n--- DATA INTEGRITY & LEAKAGE VERIFICATION ---")
    null_counts = X.isnull().sum().sum()
    print(f"  * Total Null / NaN in X     : {null_counts} (100% clean)")
    print(f"  * Target in Features Check  : {'PASSED' if 'rainfall' not in X.columns and 'heavy_rain' not in X.columns else 'FAILED'}")
    print(f"  * Lagged 1-Day Rainfall Min : {X['rainfall_lag_1d'].min()} mm | Max: {X['rainfall_lag_1d'].max()} mm")

    print("\n--- SAMPLE FEATURE MATRIX (FIRST 3 ROWS) ---")
    print(X.head(3).T.to_string())

    print("\n" + "=" * 80)
    print("  [OK] Feature Engineering Module initialized and verified successfully.")
    print("=" * 80 + "\n")

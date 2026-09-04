"""
================================================================================
JAL SANKETH PROTOTYPE - SYNTHETIC WEATHER & TERRAIN DEMO DATASET GENERATOR
================================================================================

DISCLAIMER & NOTICE:
--------------------
THIS IS A SYNTHETIC DEMO DATASET GENERATED SOLELY FOR DEMONSTRATING THE
"JAL SANKETH" PROTOTYPE IN THE CONTEXT OF SMART INDIA HACKATHON 2026.
IT MUST NOT BE PRESENTED OR INTERPRETED AS OFFICIAL OBSERVATIONAL DATA FROM THE
INDIA METEOROLOGICAL DEPARTMENT (IMD) OR ANY OTHER STATUTORY AGENCY.

The generation logic models realistic physical and statistical correlations
typical of Indian meteorological regimes (e.g., monsoon seasonality, inverse
pressure-rainfall relationships, lapse rates, and elevation profiles), but all
individual records are algorithmically synthesized.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def generate_demo_dataset(
    output_path: str = "data/raw/weather_demo.csv",
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-31",
    seed: int = 42,
    heavy_rain_threshold_mm: float = 64.5
) -> pd.DataFrame:
    """
    Generates a realistic, correlated synthetic meteorological dataset across
    multiple Indian urban and river-basin locations.

    Parameters:
    -----------
    output_path : str
        Target CSV file path.
    start_date : str
        Start date for daily time series (YYYY-MM-DD).
    end_date : str
        End date for daily time series (YYYY-MM-DD).
    seed : int
        Random seed for reproducibility.
    heavy_rain_threshold_mm : float
        Rainfall threshold (mm/24h) for classifying heavy rain (IMD standard = 64.5 mm).

    Returns:
    --------
    pd.DataFrame: Generated dataset containing at least 10,000 records.
    """
    np.random.seed(seed)

    # 1. Location definitions representing Telangana district hydrological & urban basins
    locations = [
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elev_base": 505.0, "drain_base": 0.38, "climate": "plateau_convective"},
        {"name": "Warangal", "lat": 17.9689, "lon": 79.5941, "elev_base": 302.0, "drain_base": 0.42, "climate": "plateau_convective"},
        {"name": "Karimnagar", "lat": 18.4386, "lon": 79.1288, "elev_base": 265.0, "drain_base": 0.45, "climate": "plateau_convective"},
        {"name": "Nizamabad", "lat": 18.6725, "lon": 78.0941, "elev_base": 395.0, "drain_base": 0.40, "climate": "plateau_convective"},
        {"name": "Khammam", "lat": 17.2473, "lon": 80.1514, "elev_base": 112.0, "drain_base": 0.35, "climate": "plateau_convective"},
        {"name": "Bhadrachalam", "lat": 17.6689, "lon": 80.8933, "elev_base": 55.0, "drain_base": 0.28, "climate": "gangetic_floodplain"},
        {"name": "Nalgonda", "lat": 17.0575, "lon": 79.2684, "elev_base": 240.0, "drain_base": 0.46, "climate": "plateau_convective"},
        {"name": "Mahabubnagar", "lat": 16.7488, "lon": 77.9840, "elev_base": 498.0, "drain_base": 0.48, "climate": "plateau_convective"},
        {"name": "Adilabad", "lat": 19.6641, "lon": 78.5320, "elev_base": 264.0, "drain_base": 0.44, "climate": "plateau_convective"},
        {"name": "Nirmal", "lat": 19.0964, "lon": 78.3429, "elev_base": 340.0, "drain_base": 0.36, "climate": "plateau_convective"},
        {"name": "Mancherial", "lat": 18.8679, "lon": 79.4639, "elev_base": 150.0, "drain_base": 0.34, "climate": "gangetic_floodplain"},
        {"name": "Peddapalli", "lat": 18.7557, "lon": 79.5130, "elev_base": 165.0, "drain_base": 0.39, "climate": "plateau_convective"},
        {"name": "Jagtial", "lat": 18.7946, "lon": 78.9126, "elev_base": 293.0, "drain_base": 0.43, "climate": "plateau_convective"},
        {"name": "Siddipet", "lat": 18.1018, "lon": 78.8520, "elev_base": 675.0, "drain_base": 0.50, "climate": "plateau_convective"},
        {"name": "Medak", "lat": 18.0450, "lon": 78.2630, "elev_base": 488.0, "drain_base": 0.41, "climate": "plateau_convective"},
        {"name": "Sangareddy", "lat": 17.6190, "lon": 78.0810, "elev_base": 496.0, "drain_base": 0.44, "climate": "plateau_convective"},
        {"name": "Suryapet", "lat": 17.1439, "lon": 79.6239, "elev_base": 218.0, "drain_base": 0.42, "climate": "plateau_convective"},
        {"name": "Vikarabad", "lat": 17.3366, "lon": 77.9048, "elev_base": 652.0, "drain_base": 0.55, "climate": "plateau_convective"},
        {"name": "Nagarkurnool", "lat": 16.4854, "lon": 78.3338, "elev_base": 460.0, "drain_base": 0.47, "climate": "plateau_convective"},
        {"name": "Wanaparthy", "lat": 16.3624, "lon": 78.0628, "elev_base": 380.0, "drain_base": 0.46, "climate": "plateau_convective"}
    ]

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for loc in locations:
        czone = loc["climate"]
        base_elev = loc["elev_base"]
        base_drain = loc["drain_base"]
        lat = loc["lat"]
        lon = loc["lon"]

        for dt in dates:
            month = dt.month
            day_of_year = dt.dayofyear

            # ------------------------------------------------------------------
            # 1. Seasonality & Monsoon Factor
            # ------------------------------------------------------------------
            if czone in ["coastal_sw_monsoon", "western_ghats_orographic"]:
                # Peak Southwest Monsoon (June - September)
                is_monsoon = 1 if 6 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 7.5) ** 2) / 2.2)
            elif czone == "coastal_ne_monsoon":
                # Peak Northeast Monsoon for Tamil Nadu (October - December)
                is_monsoon = 1 if 10 <= month <= 12 else 0
                monsoon_factor = np.exp(-((month - 11.0) ** 2) / 1.6)
            elif czone == "ne_heavy_basin":
                # Pre-monsoon & SW monsoon (May - September)
                is_monsoon = 1 if 5 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 6.8) ** 2) / 3.0)
            elif czone == "gangetic_floodplain":
                # July - August peak
                is_monsoon = 1 if 7 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 8.0) ** 2) / 2.0)
            else:
                # General Indian summer monsoon
                is_monsoon = 1 if 6 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 7.5) ** 2) / 2.5)

            # ------------------------------------------------------------------
            # 2. Temperature (°C)
            # ------------------------------------------------------------------
            # Annual solar cycle with peak in May (day ~140)
            annual_temp_cycle = 32.0 - 5.5 * np.cos(2 * np.pi * (day_of_year - 140) / 365.25)
            # Environmental lapse rate reduction for higher elevation (~6.5°C per 1000m)
            lapse_rate_drop = (base_elev / 1000.0) * 6.5
            # Cooling during heavy monsoon rains
            evaporative_cooling = monsoon_factor * 3.5
            
            temp_noise = np.random.normal(0, 1.8)
            temperature = float(np.clip(annual_temp_cycle - lapse_rate_drop - evaporative_cooling + temp_noise, 12.0, 45.0))

            # ------------------------------------------------------------------
            # 3. Relative Humidity (%)
            # ------------------------------------------------------------------
            # Higher in coastal zones and during monsoon; inversely related to dry temperature
            base_humidity = 52.0 + 38.0 * monsoon_factor
            if "coastal" in czone or "western_ghats" in czone:
                base_humidity += 10.0
            
            hum_noise = np.random.normal(0, 5.0)
            humidity = float(np.clip(base_humidity + hum_noise, 22.0, 99.0))

            # ------------------------------------------------------------------
            # 4. Barometric Pressure (hPa)
            # ------------------------------------------------------------------
            # Pressure drops significantly during monsoon troughs & depressions
            # Elevation decreases barometric pressure roughly ~1 hPa per 8.5 meters
            elev_pressure_offset = (base_elev / 8.5)
            monsoon_pressure_dip = monsoon_factor * 10.5
            
            # Random low-pressure depression events during monsoon
            is_depression = 1 if (is_monsoon and np.random.rand() > 0.88) else 0
            depression_dip = 8.0 if is_depression else 0.0

            press_noise = np.random.normal(0, 2.0)
            pressure = float(np.clip(1013.25 - elev_pressure_offset - monsoon_pressure_dip - depression_dip + press_noise, 880.0, 1025.0))

            # ------------------------------------------------------------------
            # 5. Wind Speed (km/h)
            # ------------------------------------------------------------------
            # Stronger winds during monsoon squalls and low pressure events
            wind_base = 8.0 + 14.0 * monsoon_factor + (18.0 if is_depression else 0.0)
            wind_speed = float(np.clip(wind_base + np.random.exponential(4.5), 1.5, 95.0))

            # ------------------------------------------------------------------
            # 6. Physical Rainfall Formation & Correlations
            # ------------------------------------------------------------------
            # High humidity + lower pressure + high monsoon factor => high rain probability
            # Normalized atmospheric instability score (0 to 1)
            norm_hum = humidity / 100.0
            norm_pres_deficit = np.clip((1015.0 - pressure - elev_pressure_offset) / 30.0, 0.0, 1.0)
            instability_score = (0.50 * norm_hum) + (0.30 * norm_pres_deficit) + (0.20 * monsoon_factor)

            rain_probability = float(np.clip(instability_score * 0.92, 0.04, 0.95))

            if np.random.rand() < rain_probability:
                # Extreme heavy downpour (Cloudburst / Cyclonic deluge)
                # Meaningful minority tail (> 64.5 mm)
                if (humidity > 82.0 and is_monsoon) and (np.random.rand() < 0.16 or is_depression):
                    rainfall = float(np.clip(np.random.gamma(shape=4.0, scale=26.0), 65.0, 260.0))
                # Moderate rainfall (15.6 mm - 64.4 mm)
                elif np.random.rand() < 0.38:
                    rainfall = float(np.clip(np.random.gamma(shape=2.5, scale=12.0), 15.6, 64.4))
                # Light rainfall (0.1 mm - 15.5 mm)
                else:
                    rainfall = float(np.clip(np.random.exponential(scale=4.0), 0.1, 15.5))
            else:
                rainfall = 0.0

            # ------------------------------------------------------------------
            # 7. Heavy Rain Binary Target
            # ------------------------------------------------------------------
            # Standard IMD definition: Heavy Rainfall is >= 64.5 mm in 24 hours
            heavy_rain = 1 if rainfall >= heavy_rain_threshold_mm else 0

            # ------------------------------------------------------------------
            # 8. Elevation & Drainage Quality (0.0 to 1.0)
            # ------------------------------------------------------------------
            # Small local variation representing different sensor points within city
            elevation = float(np.clip(base_elev + np.random.normal(0, 1.2), 1.0, 1500.0))
            drainage_factor = float(np.clip(base_drain + np.random.normal(0, 0.03), 0.05, 0.98))

            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 2),
                "pressure": round(pressure, 2),
                "wind_speed": round(wind_speed, 2),
                "rainfall": round(rainfall, 2),
                "elevation": round(elevation, 1),
                "drainage_factor": round(drainage_factor, 2),
                "heavy_rain": heavy_rain
            })

    df = pd.DataFrame(records)

    # --------------------------------------------------------------------------
    # Save output to CSV
    # --------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def print_dataset_summary(df: pd.DataFrame, output_path: str, heavy_rain_threshold_mm: float = 64.5) -> None:
    """
    Prints required dataset metrics, columns, missing values, statistics, and class distribution.
    """
    print("\n" + "=" * 80)
    print("      JAL SANKETH - SYNTHETIC DEMO DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"[+] Output File Saved At : {output_path}")
    print(f"[+] Dataset Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("-" * 80)

    print("\n--- 1. COLUMN NAMES & DATA TYPES ---")
    for col in df.columns:
        print(f"  * {col:<18} : {str(df[col].dtype):<10} (Sample: {df[col].iloc[0]})")

    print("\n--- 2. MISSING VALUES CHECK ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  [OK] No missing values found across all columns (0 nulls).")
    else:
        print(missing[missing > 0])

    print("\n--- 3. BASIC NUMERICAL STATISTICS ---")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    print("\n--- 4. HEAVY RAIN CLASS DISTRIBUTION (Threshold >= 64.5 mm) ---")
    counts = df["heavy_rain"].value_counts()
    percs = df["heavy_rain"].value_counts(normalize=True) * 100
    for val in [0, 1]:
        label = "Normal / Light / Moderate Rain (0)" if val == 0 else "Heavy Rain Event (1)"
        cnt = counts.get(val, 0)
        pct = percs.get(val, 0.0)
        print(f"  * Class {val} [{label:<35}] : {cnt:>6,} rows ({pct:>5.2f}%)")

    print("\n" + "=" * 80)
    print("NOTICE: Synthetic demo data generated successfully for Jal Sanketh.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    target_csv = os.path.join("data", "raw", "weather_demo.csv")
    dataset_df = generate_demo_dataset(output_path=target_csv)
    print_dataset_summary(dataset_df, target_csv)

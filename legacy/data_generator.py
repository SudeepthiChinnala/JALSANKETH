"""
Data Generator for Jal Sanketh Prototype
Generates realistic historical meteorological and topographical dataset for Indian stations.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_location_profiles() -> pd.DataFrame:
    """
    Returns pre-configured topographical profiles for major flood-prone monitoring locations across India.
    """
    locations = [
        {
            "location_id": "LOC001",
            "location_name": "Mumbai - Dadar / Hindmata Lowland",
            "city": "Mumbai",
            "state": "Maharashtra",
            "latitude": 19.0178,
            "longitude": 72.8478,
            "elevation_m": 6.5,
            "slope_deg": 0.8,
            "drainage_capacity_index": 0.35,
            "soil_saturation_baseline": 0.75,
            "proximity_to_waterbody_km": 1.2,
            "climate_zone": "coastal_sw_monsoon"
        },
        {
            "location_id": "LOC002",
            "location_name": "Chennai - Velachery Flood Basin",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "latitude": 12.9815,
            "longitude": 80.2180,
            "elevation_m": 8.0,
            "slope_deg": 0.5,
            "drainage_capacity_index": 0.30,
            "soil_saturation_baseline": 0.80,
            "proximity_to_waterbody_km": 0.8,
            "climate_zone": "coastal_ne_monsoon"
        },
        {
            "location_id": "LOC003",
            "location_name": "Bengaluru - Bellandur Catchment",
            "city": "Bengaluru",
            "state": "Karnataka",
            "latitude": 12.9304,
            "longitude": 77.6784,
            "elevation_m": 870.0,
            "slope_deg": 2.2,
            "drainage_capacity_index": 0.40,
            "soil_saturation_baseline": 0.55,
            "proximity_to_waterbody_km": 0.5,
            "climate_zone": "plateau_thunderstorm"
        },
        {
            "location_id": "LOC004",
            "location_name": "Patna - Ganga Floodplain (Rajendra Nagar)",
            "city": "Patna",
            "state": "Bihar",
            "latitude": 25.6022,
            "longitude": 85.1589,
            "elevation_m": 53.0,
            "slope_deg": 0.4,
            "drainage_capacity_index": 0.25,
            "soil_saturation_baseline": 0.85,
            "proximity_to_waterbody_km": 1.5,
            "climate_zone": "gangetic_plain"
        },
        {
            "location_id": "LOC005",
            "location_name": "Guwahati - Brahmaputra Basin (Zoo Road)",
            "city": "Guwahati",
            "state": "Assam",
            "latitude": 26.1754,
            "longitude": 91.7766,
            "elevation_m": 55.0,
            "slope_deg": 1.5,
            "drainage_capacity_index": 0.38,
            "soil_saturation_baseline": 0.88,
            "proximity_to_waterbody_km": 2.0,
            "climate_zone": "ne_heavy_rain"
        },
        {
            "location_id": "LOC006",
            "location_name": "Kochi - Vembanad Coastal Plain",
            "city": "Kochi",
            "state": "Kerala",
            "latitude": 9.9312,
            "longitude": 76.2673,
            "elevation_m": 4.0,
            "slope_deg": 0.3,
            "drainage_capacity_index": 0.42,
            "soil_saturation_baseline": 0.82,
            "proximity_to_waterbody_km": 0.4,
            "climate_zone": "coastal_sw_monsoon"
        },
        {
            "location_id": "LOC007",
            "location_name": "Delhi - Yamuna Floodplain (Kashmere Gate)",
            "city": "Delhi",
            "state": "Delhi",
            "latitude": 28.6672,
            "longitude": 77.2289,
            "elevation_m": 216.0,
            "slope_deg": 0.9,
            "drainage_capacity_index": 0.48,
            "soil_saturation_baseline": 0.60,
            "proximity_to_waterbody_km": 0.9,
            "climate_zone": "northern_plains"
        },
        {
            "location_id": "LOC008",
            "location_name": "Kolkata - Hooghly Estuary (Central Avenue)",
            "city": "Kolkata",
            "state": "West Bengal",
            "latitude": 22.5726,
            "longitude": 88.3639,
            "elevation_m": 9.0,
            "slope_deg": 0.4,
            "drainage_capacity_index": 0.32,
            "soil_saturation_baseline": 0.78,
            "proximity_to_waterbody_km": 1.1,
            "climate_zone": "delta_cyclonic"
        },
        {
            "location_id": "LOC009",
            "location_name": "Hyderabad - Musi River Lowland (Amberpet)",
            "city": "Hyderabad",
            "state": "Telangana",
            "latitude": 17.3871,
            "longitude": 78.5130,
            "elevation_m": 505.0,
            "slope_deg": 1.8,
            "drainage_capacity_index": 0.45,
            "soil_saturation_baseline": 0.50,
            "proximity_to_waterbody_km": 0.6,
            "climate_zone": "plateau_thunderstorm"
        },
        {
            "location_id": "LOC010",
            "location_name": "Bhubaneswar - Daya River Basin (Old Town)",
            "city": "Bhubaneswar",
            "state": "Odisha",
            "latitude": 20.2444,
            "longitude": 85.8346,
            "elevation_m": 45.0,
            "slope_deg": 1.0,
            "drainage_capacity_index": 0.50,
            "soil_saturation_baseline": 0.70,
            "proximity_to_waterbody_km": 1.8,
            "climate_zone": "delta_cyclonic"
        },
        {
            "location_id": "LOC011",
            "location_name": "Surat - Tapi Basin (Athwa Lines)",
            "city": "Surat",
            "state": "Gujarat",
            "latitude": 21.1702,
            "longitude": 72.8311,
            "elevation_m": 13.0,
            "slope_deg": 0.6,
            "drainage_capacity_index": 0.40,
            "soil_saturation_baseline": 0.68,
            "proximity_to_waterbody_km": 1.0,
            "climate_zone": "coastal_sw_monsoon"
        },
        {
            "location_id": "LOC012",
            "location_name": "Wayanad - Vythiri Foothills",
            "city": "Wayanad",
            "state": "Kerala",
            "latitude": 11.5527,
            "longitude": 76.0407,
            "elevation_m": 720.0,
            "slope_deg": 8.5,
            "drainage_capacity_index": 0.65,
            "soil_saturation_baseline": 0.85,
            "proximity_to_waterbody_km": 0.3,
            "climate_zone": "ghats_high_precipitation"
        }
    ]
    return pd.DataFrame(locations)


def generate_historical_weather(locations_df: pd.DataFrame, start_date: str = "2024-01-01", end_date: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic daily meteorological time-series records for each station based on climatic region patterns.
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    records = []
    
    for _, loc in locations_df.iterrows():
        loc_id = loc['location_id']
        czone = loc['climate_zone']
        
        # Base climate parameters per zone
        for dt in dates:
            month = dt.month
            day_of_year = dt.dayofyear
            
            # Seasonal Monsoon intensity indicator
            if czone in ["coastal_sw_monsoon", "ghats_high_precipitation"]:
                # Strong SW Monsoon peak in June-September
                is_monsoon = 1 if 6 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 7.5) ** 2) / 2.0)
            elif czone == "coastal_ne_monsoon":
                # Chennai NE Monsoon peak in October-December
                is_monsoon = 1 if 10 <= month <= 12 else 0
                monsoon_factor = np.exp(-((month - 11.0) ** 2) / 1.5)
            elif czone == "ne_heavy_rain":
                # Assam: Pre-monsoon & SW monsoon May-Sept
                is_monsoon = 1 if 5 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 6.8) ** 2) / 3.0)
            elif czone == "gangetic_plain":
                # North/East India: July-August
                is_monsoon = 1 if 7 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 8.0) ** 2) / 1.8)
            else:
                # General Indian summer monsoon
                is_monsoon = 1 if 6 <= month <= 9 else 0
                monsoon_factor = np.exp(-((month - 7.5) ** 2) / 2.5)

            # Temperature calculation (°C)
            base_temp = 32.0 - 5.0 * np.cos(2 * np.pi * (day_of_year - 140) / 365)
            if loc['elevation_m'] > 500:
                base_temp -= (loc['elevation_m'] - 500) * 0.0065  # Lapse rate
            temp_c = float(np.clip(base_temp + np.random.normal(0, 2.0) - (monsoon_factor * 3.5), 12.0, 44.0))

            # Humidity calculation (%)
            base_humidity = 55.0 + 35.0 * monsoon_factor + np.random.normal(0, 6.0)
            if "coastal" in czone:
                base_humidity += 10.0
            humidity_pct = float(np.clip(base_humidity, 25.0, 99.0))

            # Barometric Pressure (hPa)
            # Monsoon lows and cyclonic depressions
            pressure_drop = monsoon_factor * 12.0 + (5.0 if np.random.rand() > 0.85 and is_monsoon else 0.0)
            base_pressure = 1012.0 - pressure_drop + np.random.normal(0, 2.5)
            # Elevation adjustment
            pressure_hpa = float(np.clip(base_pressure - (loc['elevation_m'] * 0.09), 920.0, 1022.0))

            # Wind speed (km/h) and direction (degrees)
            wind_speed_kmh = float(np.clip(10.0 + 15.0 * monsoon_factor + np.random.exponential(4.0), 2.0, 85.0))
            wind_dir_deg = float(np.clip(np.random.normal(230 if "sw" in czone else 60, 40) % 360, 0, 360))

            # Dew point temperature (°C)
            # Approximated via Magnus formula inversion
            dew_point_c = float(np.clip(temp_c - ((100 - humidity_pct) / 5.0) + np.random.normal(0, 0.8), 5.0, 32.0))

            # Cloud cover (%)
            cloud_cover_pct = float(np.clip(15.0 + 75.0 * monsoon_factor + np.random.normal(0, 10.0), 0.0, 100.0))

            # Rainfall generation logic (Zero-inflated with extreme tails for cloudbursts)
            rain_prob = 0.08 + 0.65 * monsoon_factor
            if humidity_pct > 85 and pressure_hpa < 1008:
                rain_prob += 0.20
            
            rain_prob = min(rain_prob, 0.95)

            if np.random.rand() < rain_prob:
                # Rain event
                if np.random.rand() < 0.12 and is_monsoon:
                    # Extreme / Heavy rainfall event (Cloudburst / Cyclonic downpour)
                    rainfall_mm = float(np.clip(np.random.gamma(shape=3.5, scale=30.0), 65.0, 240.0))
                elif np.random.rand() < 0.35:
                    # Moderate rainfall (15.6 - 64.4 mm)
                    rainfall_mm = float(np.clip(np.random.gamma(shape=2.5, scale=12.0), 16.0, 64.0))
                else:
                    # Light rain (0.1 - 15.5 mm)
                    rainfall_mm = float(np.clip(np.random.exponential(scale=4.5), 0.5, 15.5))
            else:
                rainfall_mm = 0.0

            # IMD Heavy Rain Category Flag
            if rainfall_mm >= 115.6:
                rain_category = "Very/Extremely Heavy"
                heavy_flag = 1
            elif rainfall_mm >= 64.5:
                rain_category = "Heavy"
                heavy_flag = 1
            elif rainfall_mm >= 15.6:
                rain_category = "Moderate"
                heavy_flag = 0
            elif rainfall_mm > 0:
                rain_category = "Light"
                heavy_flag = 0
            else:
                rain_category = "No Rain"
                heavy_flag = 0

            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "location_id": loc_id,
                "temperature_c": round(temp_c, 2),
                "humidity_pct": round(humidity_pct, 2),
                "pressure_hpa": round(pressure_hpa, 2),
                "wind_speed_kmh": round(wind_speed_kmh, 2),
                "wind_direction_deg": round(wind_dir_deg, 1),
                "dew_point_c": round(dew_point_c, 2),
                "cloud_cover_pct": round(cloud_cover_pct, 1),
                "rainfall_amount_mm": round(rainfall_mm, 2),
                "heavy_rain_flag": heavy_flag,
                "imd_rain_category": rain_category
            })
            
    df = pd.DataFrame(records)
    return df


def generate_all_datasets(output_dir: str = "data/raw") -> None:
    """
    Main driver to generate both raw CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Location Geo Profiles
    loc_df = get_location_profiles()
    loc_file = os.path.join(output_dir, "location_geo_profiles.csv")
    loc_df.to_csv(loc_file, index=False)
    print(f"[+] Saved location profiles to {loc_file} ({len(loc_df)} locations)")
    
    # 2. Historical Weather
    weather_df = generate_historical_weather(loc_df)
    weather_file = os.path.join(output_dir, "weather_historical.csv")
    weather_df.to_csv(weather_file, index=False)
    print(f"[+] Saved historical weather data to {weather_file} ({len(weather_df)} records)")


if __name__ == "__main__":
    generate_all_datasets()

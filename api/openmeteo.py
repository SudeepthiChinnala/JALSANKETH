import os
import json
import requests
import pandas as pd
from datetime import datetime

def collect_openmeteo(latitude, longitude, start_date, end_date, save_dir=None):
    """Fetch historical hourly data from Open-Meteo API.

    Parameters
    ----------
    latitude: float
    longitude: float
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    save_dir: str or None
        Directory where raw JSON and CSV will be saved.
        If None, defaults to "data/raw/openmeteo".

    Returns
    -------
    dict with keys:
        status_code, records, variables, json_path, csv_path, error (None if ok)
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relativehumidity_2m",
            "precipitation",
            "surface_pressure",
            "windspeed_10m",
            "winddirection_10m",
            "cloudcover",
            "soil_moisture_0_to_7cm",
        ],
        "timezone": "auto",
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
    except requests.Timeout:
        return {"error": "Request timed out", "status_code": None}
    except requests.HTTPError as e:
        return {"error": f"HTTP error: {e}", "status_code": response.status_code if response else None}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "status_code": None}

    data = response.json()
    if "hourly" not in data or not data["hourly"]:
        return {"error": "Missing hourly data in response", "status_code": response.status_code}

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    # Convert time list to timestamp column
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop(columns=["time"])  # remove original list column
    df["latitude"] = latitude
    df["longitude"] = longitude

    if save_dir is None:
        save_dir = os.path.join("data", "raw", "openmeteo")
    os.makedirs(save_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_path = os.path.join(save_dir, f"openmeteo_{latitude}_{longitude}_{ts}.json")
    csv_path = os.path.join(save_dir, f"openmeteo_{latitude}_{longitude}_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    df.to_csv(csv_path, index=False)

    return {
        "status_code": response.status_code,
        "records": len(df),
        "variables": list(df.columns),
        "json_path": json_path,
        "csv_path": csv_path,
        "error": None,
    }

if __name__ == "__main__":
    # simple manual test
    result = collect_openmeteo(17.3850, 78.4867, "2024-09-01", "2024-09-07")
    print(result)

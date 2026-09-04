import requests
import pandas as pd

# --------------------------------
# 1. Open-Meteo API address
# --------------------------------
url = "https://api.open-meteo.com/v1/forecast"


# --------------------------------
# 2. Define request parameters
# --------------------------------
params = {
    "latitude": 17.3850,
    "longitude": 78.4867,
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


# --------------------------------
# 3. Send request & convert JSON
# --------------------------------
response = requests.get(url, params=params)
data = response.json()


# --------------------------------
# 4. Load into Pandas DataFrame
# --------------------------------
hourly_data = data["hourly"]
df = pd.DataFrame(hourly_data)


# --------------------------------
# 5. Clean up & display
# --------------------------------
# Convert 'time' column to proper datetime format
df["time"] = pd.to_datetime(df["time"])

print("------ FIRST 5 ROWS ------")
print(df.head())

print("\n------ DATAFRAME INFO ------")
print(df.info())
# --------------------------------
# 6. Save to CSV
# --------------------------------
df.to_csv("weather_forecast.csv", index=False)
print("\nCSV file saved successfully!")
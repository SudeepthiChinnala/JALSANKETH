# 🌊 Jal Sanketh (जल संकेत)
### AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
**Smart India Hackathon 2026 Prototype**

---

## 📌 Problem Statement Overview
Urban flooding and heavy precipitation events pose major threats to life, civic infrastructure, and urban mobility. Early warning systems must bridge the gap between atmospheric forecasts and localized inundation vulnerability.

**Jal Sanketh** is a prototype that integrates:
1. **Machine Learning Weather Prediction**: Uses an **XGBoost** regression model trained on historical meteorological parameters to predict 24-hour rainfall intensity and flag extreme precipitation events (> 64.5 mm/day per IMD standards).
2. **Hydrological & Terrain Vulnerability Analysis**: Ingests key topographical factors (elevation, urban drainage capacity index, slope, antecedent soil moisture, and proximity to water bodies).
3. **Compound Flood Risk Index (0–100)**: Computes a multi-criteria risk score classified into `Low`, `Moderate`, `High`, and `Critical` tiers.
4. **Interactive Early Warning Dashboard**: A modular **Streamlit** dashboard featuring Folium geospatial maps, an interactive "What-If" scenario sandbox, Plotly weather analytics, and automated NDMA/IMD-style emergency bulletins.

---

## 🛠️ Technology Stack
- **Core Language**: Python 3.10+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Geospatial & Visualizations**: Folium, Streamlit-Folium, Plotly
- **Application Framework**: Streamlit

---

## 📁 Repository Structure
```
JAL-SANKETH/
├── data/
│   ├── raw/
│   │   ├── weather_historical.csv       # Multi-station historical weather records (8,700+ rows)
│   │   └── location_geo_profiles.csv    # 12 Indian urban flood basin profiles
│   └── processed/
│       └── merged_training_data.csv     # Preprocessed lag-engineered dataset
├── models/
│   ├── xgboost_rainfall_model.pkl       # Trained XGBoost regressor
│   ├── preprocessor.pkl                 # Fitted StandardScaler pipeline
│   └── model_metrics.json               # Evaluation metrics (RMSE, MAE, R², F1)
├── src/
│   ├── __init__.py
│   ├── data_generator.py                # Synthetic dataset generator for Indian climatic zones
│   ├── preprocessing.py                 # Cleaning, atmospheric index calculation & lag features
│   ├── train_model.py                   # XGBoost training and evaluation pipeline
│   ├── risk_engine.py                   # Multi-criteria flood risk scoring formula
│   └── alert_system.py                  # Standard Operating Procedure (SOP) bulletin generator
├── app/
│   ├── __init__.py
│   ├── app.py                           # Main Streamlit dashboard entrypoint
│   ├── utils.py                         # Model caching, inference helpers, and custom CSS
│   └── views/
│       ├── __init__.py
│       ├── live_map.py                  # Interactive Folium map with color-coded risk markers
│       ├── simulator.py                 # "What-If" real-time scenario simulation engine
│       ├── analytics.py                 # Plotly rainfall trends, correlations & distributions
│       ├── alerts.py                    # Actionable advisories & downloadable bulletins
│       └── model_insights.py            # Feature importance & SIH prototype roadmap
├── requirements.txt                     # Dependency specifications
└── README.md                            # Documentation
```

---

## ⚙️ Installation & Running the Application

### 1. Clone or Open the Repository
```bash
cd d:\JAL-SANKETH
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train the ML Model (Optional - App auto-generates if missing)
```bash
python -m src.data_generator
python -m src.preprocessing
python src/train_model.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 Machine Learning & Risk Formulation

### 1. Input Features for XGBoost
- Ambient Temperature (°C)
- Relative Atmospheric Humidity (%)
- Surface Barometric Pressure (hPa) & 24h Pressure Drop
- Wind Speed (km/h) & Wind Direction (°)
- Dew Point Temperature (°C) & Dew Point Spread
- Cloud Cover (%)
- Antecedent 1-Day & 3-Day Lagged Rainfall (mm)
- Seasonality Cyclical Encodings ($\sin(\text{month}), \cos(\text{month})$)
- Station Elevation (m)

### 2. Flood Risk Formula ($0 - 100$)
$$\text{Flood Risk Score} = (0.40 \cdot R_{\text{norm}}) + (0.18 \cdot E_{\text{norm}}) + (0.20 \cdot D_{\text{norm}}) + (0.10 \cdot S_{\text{norm}}) + (0.12 \cdot M_{\text{norm}})$$

- **🟢 Low (0–35)**: Normal seasonal conditions.
- **🟡 Moderate (36–60)**: Advisory alert; inspect stormwater drains.
- **🟠 High (61–80)**: Severe alert; localized inundation expected, deploy pump units.
- **🔴 Critical (81–100)**: Emergency warning; high inundation risk, trigger SDRF/NDRF evacuation SOPs.

---

## 🔍 Scope Transparency for SIH 2026 Jury
To maintain full academic and technical integrity:

| System Component | Hackathon Prototype (Implemented) | Production Roadmap (Future Integrations) |
| :--- | :--- | :--- |
| **Weather Feed** | Curated historical multi-station time-series dataset. | Real-time IMD AWS API & Open-Meteo REST ingestion. |
| **Radar Data** | Derived atmospheric reflectivity & rain rate parameters in tabular format. | Direct ingestion of IMD Doppler Weather Radar (DWR) NetCDF/HDF5 feeds. |
| **Satellite Feeds** | Simulated cloud-cover & water vapor proxies. | Automated GeoTIFF ingestion from INSAT-3D/3DR via ISRO MOSDAC. |
| **NWP Model** | Feature-engineered spatial grids with lagged dynamics. | High-resolution WRF / NCMRWF GFS gridded model outputs. |
| **Geospatial Terrain** | Location profiles with synthetic & derived DEM values. | 30m CartoDEM / LiDAR raster integration with urban drainage shapefiles. |

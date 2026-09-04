# 🌊 Jal Sanketh (जल संकेत)
### AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
**Smart India Hackathon 2026 Prototype — Telangana State Flood Command Center**

---

## 📌 Problem Statement Overview
Urban flooding and extreme convective precipitation events pose severe threats to human life, public infrastructure, and economic productivity. Conventional weather advisories often provide generalized regional forecasts that fail to account for hyper-local topographical bottlenecks, soil saturation, and urban drainage constraints.

**Jal Sanketh** bridges the gap between atmospheric meteorology and ground-level hydrological vulnerability. Specifically tailored for **Telangana State**, the platform continuously monitors **20 district river basins and urban catchments** (e.g., Hyderabad Musi Basin, Warangal Kakatiya Basin, Bhadrachalam Godavari Lowland, Nizamabad, Karimnagar):

1. **Machine Learning Weather Intelligence**: An **XGBoost Classifier** (`heavy_rain_model.joblib`) trained on 15,240 records across 20 Telangana stations predicting extreme precipitation events ($\ge 64.5$ mm/day, IMD standard) with **0.9710 ROC-AUC** and 95.41% accuracy.
2. **Zero-Leakage Atmospheric Feature Engineering**: 24 engineered parameters including strictly shifted antecedent lags (1d, 3d rolling, 7d cumulative sum), 24h barometric tendencies, Magnus dew-point estimations, vapor pressure deficit (VPD), and cyclical seasonality encodings.
3. **Compound Flood Risk Index (0–100)**: Multi-Criteria Decision Analysis (MCDA) combining ML heavy-rain probability, rainfall volume, urban drainage deficit, low elevation vulnerability, and antecedent saturation.
4. **Interactive Command Center Dashboard**: A 7-tab **Streamlit** dashboard featuring Folium geospatial maps with authentic Telangana GeoJSON boundaries, an interactive **"What-If" scenario simulation sandbox**, Plotly weather analytics, automated NDMA/IMD-style emergency bulletins, and instant `.txt` SITREP reports.
5. **Emergency Evacuation System**: Database of 53 safe shelters across Telangana with Haversine distance nearest-shelter calculations and automated emergency dispatch banners for Critical risk zones.
6. **Live REST API Integration**: Real-time 7-day hourly forecast synchronization via the Open-Meteo REST API for any selected district basin.

---

## 🛠️ Technology Stack
- **Core Language**: Python 3.10+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Geospatial & Visualization**: Folium, Streamlit-Folium, Plotly Express/Graph Objects
- **Application Framework**: Streamlit
- **Quality Assurance**: Pytest (24 Automated Unit & E2E Integration Tests)

---

## 📁 Repository Structure
```
JAL-SANKETH/
├── app.py                               # Main Streamlit Command Center entrypoint
├── requirements.txt                     # Dependency specifications
├── README.md                            # Comprehensive documentation
├── data/
│   ├── geospatial/
│   │   └── telangana_boundary.geojson   # Authentic Telangana state & district vector boundaries
│   ├── raw/
│   │   └── weather_demo.csv             # 15,240 correlated records across 20 Telangana basins
│   ├── safe_places.csv                  # 53 designated emergency shelter profiles
│   ├── weather_forecast.csv             # Baseline 7-day hourly forecast series
│   └── create_demo_dataset.py           # Realistic meteorological & terrain dataset generator
├── evacuation/
│   ├── __init__.py
│   ├── emergency.py                     # Emergency dispatch escalation for Critical risk
│   └── safe_places.py                   # Haversine distance nearest shelter calculation engine
├── models/
│   ├── heavy_rain_model.joblib          # Trained XGBoost heavy-rainfall classifier
│   ├── feature_columns.json             # Strict 24-feature schema specification
│   └── model_metrics.json               # Evaluation metrics (ROC-AUC: 0.9710, Accuracy: 95.41%)
├── src/
│   ├── __init__.py
│   ├── ui.py                            # Dark-mode command center CSS & custom card components
│   ├── feature_engineering.py           # Zero-leakage 24-feature transformation pipeline
│   ├── flood_risk.py                    # MCDA composite risk scoring & natural language diagnostics
│   ├── simulator.py                     # "What-If" real-time scenario simulation engine
│   ├── alerts.py                        # NDMA early warning advisory generator & SOPs
│   ├── map.py                           # Folium geospatial map with boundary & shelter layers
│   ├── forecast_loader.py               # Live Open-Meteo API & forecast ingestion pipeline
│   ├── location.py                      # Geolocation detection & manual coordinate fallback
│   ├── train_model.py                   # XGBoost training pipeline
│   └── evaluate_model.py                # Meteorological scenario testing & held-out evaluation
├── tests/
│   ├── test_e2e_pipeline.py             # 15 end-to-end system integration tests
│   ├── test_flood_risk.py               # 6 flood risk calculation unit tests
│   └── test_simulator.py                # 3 scenario simulation unit tests
├── utils/
│   ├── __init__.py
│   └── config.py                        # Risk weights, tier thresholds, benchmarks
├── legacy/                              # Safely archived earlier pan-India prototype files
└── reports/
    └── figures/                         # Publication-grade model & EDA visualization charts
```

---

## ⚙️ Installation & Quickstart

### 1. Clone or Open the Repository
```bash
cd d:\jalsanketh\JALSANKETH
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```bash
pytest tests/ -v
```
*All 24 unit and end-to-end integration tests should pass with 100% success.*

### 4. Launch the Command Center Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 Machine Learning & Risk Formulation

### 1. 24-Feature XGBoost Architecture
- **Meteorological Base**: Ambient Temperature (°C), Relative Humidity (%), Surface Barometric Pressure (hPa), Wind Speed (km/h).
- **Temporal & Cyclical**: $\sin/\cos$ month encodings, $\sin/\cos$ Julian day-of-year encodings, Southwest & Northeast monsoon flags.
- **Strict Shifted Lags**: 1-Day shifted rainfall lag, 3-Day shifted rolling mean, 7-Day shifted cumulative saturation sum.
- **Tendencies & Gradients**: 24-Hour barometric pressure tendency, 24-Hour temperature tendency.
- **Atmospheric Thermodynamics**: Magnus-formula estimated dew point (°C), dew point depression (°C), vapor pressure deficit (VPD in kPa), convective atmospheric instability index.
- **Terrain Interactions**: Elevation (m ASL), drainage factor, drainage deficit, elevation vulnerability, runoff potential index.

### 2. Compound Flood Risk Formulation ($0 - 100$)
$$\text{Flood Risk Score} = (0.35 \cdot P_{\text{ML}}) + (0.25 \cdot R_{\text{norm}}) + (0.20 \cdot D_{\text{norm}}) + (0.12 \cdot E_{\text{norm}}) + (0.08 \cdot S_{\text{norm}})$$

Where:
- $P_{\text{ML}}$ = XGBoost predicted probability of extreme heavy rainfall ($\ge 64.5$ mm) $\in [0, 1]$
- $R_{\text{norm}}$ = 24-hr precipitation normalized to 150 mm deluge benchmark $\in [0, 1]$
- $D_{\text{norm}}$ = Urban drainage bottleneck deficit ($1.0 - \text{drainage factor}$) $\in [0, 1]$
- $E_{\text{norm}}$ = Inverse elevation vulnerability ($1.0 - \frac{\text{elevation}}{300\text{m}}$) $\in [0, 1]$
- $S_{\text{norm}}$ = 7-day antecedent catchment saturation normalized to 200 mm benchmark $\in [0, 1]$

### 3. Operational Classification Tiers
- **🟢 Low (0–24.99)**: Normal seasonal conditions. Negligible pooling; routine drainage maintenance.
- **🟡 Moderate (25–49.99)**: Advisory alert. Localized ponding in low-lying subways and culverts.
- **🟠 High (50–74.99)**: Severe alert. Waterlogging across transit corridors; deploy dewatering pump units.
- **🔴 Critical (75–100)**: EMERGENCY ALERT. High inundation risk; activate SDRF/NDRF evacuation SOPs.

---

## 🧪 "What-If" Scenario Simulation Sandbox
The dashboard features an interactive simulation sandbox allowing evaluators to stress-test any basin profile:
- **Preset Scenarios**: One-click loading of *Clear Sky / Dry Regime*, *Moderate Monsoon Shower*, *Active Monsoon Trough*, *Cyclonic Deluge / Flash Flood*, and *High-Elevation Drainage Test*.
- **Interactive Sliders**: Independent manipulation of simulated rainfall ($0-250$ mm), drainage capacity ($0.05-1.0$), 7-day saturation ($0-300$ mm), and barometric drop.
- **Real-Time Visualizations**: Dynamic Plotly gauge chart, factor attribution point breakdown bar chart, natural language scientific explanation, and nearest evacuation shelter calculation.

---

## 🔍 Scope Transparency for SIH 2026 Jury

| System Component | Hackathon Prototype (Implemented) | Production Roadmap (Future Integrations) |
| :--- | :--- | :--- |
| **Weather Feed** | Multi-station Telangana historical series + Live Open-Meteo REST API sync. | Real-time IMD AWS API & NCMRWF operational feeds. |
| **Radar Data** | Derived atmospheric reflectivity & instability parameters in tabular format. | Direct ingestion of IMD Doppler Weather Radar (DWR) NetCDF/HDF5 feeds. |
| **Satellite Feeds** | Convective instability and vapor deficit thermodynamic proxies. | Automated GeoTIFF ingestion from INSAT-3D/3DR via ISRO MOSDAC. |
| **Geospatial Terrain** | 20 Telangana district river basins with authentic GeoJSON boundary overlay. | 30m CartoDEM / LiDAR raster integration with municipal GIS storm pipe networks. |
| **Evacuation** | Haversine distance nearest shelter selection across 53 Telangana facilities. | Turn-by-turn road network pathfinding with dynamic flooded road avoidance (OSRM). |

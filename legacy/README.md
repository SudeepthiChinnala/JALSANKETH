# Jal Sanketh — Legacy Architecture Archive

This directory contains archived code from the initial Phase 1 conceptual prototype (12 pan-India cities) of **Jal Sanketh**. 

The operational system has been upgraded to a dedicated **20 Telangana River Basin Disaster Management Command Center** (located in `src/` and `app.py`).

### Archived Components
1. **`GEMINI.py` & `Gemini corrected code.py`**: Initial monolithic experimental Streamlit drafts.
2. **`alert_system.py`**: Early rule-based alert system for the 12 pan-India cities. (Superseded by `src/alerts.py`).
3. **`risk_engine.py`**: Early 4-factor risk scoring engine. (Superseded by the 5-factor compound MCDA engine in `src/flood_risk.py`).
4. **`preprocessing.py`**: Early preprocessing routines. (Superseded by `src/feature_engineering.py`).
5. **`data_generator.py`**: Early synthetic generator. (Superseded by `data/create_demo_dataset.py`).

These files are preserved here for archival provenance and historical reference.

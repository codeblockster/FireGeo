# Wildfire Management System 🔥

> A comprehensive system for wildfire risk assessment, detection, and spread prediction using machine learning and satellite data, integrated into a single Streamlit dashboard.

## 🌟 Features

- **🔍 Active Fire Detection** - Real-time monitoring using NASA FIRMS satellite data (VIIRS/MODIS) with **Global Earth Engine Integration**
- **⚠️ Pre-Fire Risk Assessment** - Advanced **CatBoost** model using **81 environmental features** (Weather, Terrain, Vegetation, **Active Fire Density**)
- **📊 Post-Fire Spread Prediction** - U-Net deep learning model for spatial fire spread forecasting
- **🗺️ Interactive Dashboard** - Streamlit web interface with **Global Map Capabilities**
- **🌍 Environmental Data Integration** - Automated fetching from **Google Earth Engine** and **Open-Meteo Weather API**

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Earth Engine account (for environmental data)
- NASA FIRMS API key (for fire detection)
- Weather API Key (OpenWeatherMap or similar, for live data)

### Installation

```bash
# 1. Clone the repository
cd "f:\minor project wildfire management v3"

# 2. Set up the Python 3.11 Environment
# Create a virtual environment with Python 3.11
python -m venv venv_py311

# Activate the environment
# Windows:
.\venv_py311\Scripts\activate
# Linux/Mac:
# source venv_py311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT
# - NASA_FIRMS_API_KEY
# - WEATHER_API_KEY
```

### Running the Application

Double-click `run_app_py311.bat` to start the application automatically.

Or run manually:
```bash
# Activate environment (if not already active)
.\venv_py311\Scripts\activate

# Start the Integrated Dashboard
streamlit run frontend/app.py --server.port 8501
```

**Access the Application:**
- 🌐 **Dashboard**: http://localhost:8501

## 📁 Project Structure

```
f:\v4 cleanup\
├── backend/                      # All Backend Code and Data
│   ├── src/                      # Application Source Code
│   │   ├── analysis/            # Analysis modules
│   │   │   ├── pre_fire.py     # CatBoost Risk Analyzer
│   │   │   ├── post_fire.py    # Post-fire burn severity (dNBR)
│   │   │   └── spread_prediction.py # Fire spread forecasting
│   │   ├── data_collection/     # Data collection modules
│   │   │   ├── gee_extractor.py    # GEE: Terrain, LST, SAVI
│   │   │   ├── weather_api.py      # Weather: Current, History, Rolling Stats
│   │   │   ├── nasa_firms.py       # NASA FIRMS fire data
│   │   │   └── sentinel_manager.py # Sentinel-2 imagery
│   │   ├── models/              # Model classes
│   │   │   ├── catboost_deployment_tuned.py # CatBoost Predictor Class
│   │   │   └── unet_fire.py    # U-Net architecture
│   │   ├── utils/
│   │   │   ├── feature_engineering.py # Feature Orchestrator (81 features)
│   │   │   └── calculations.py # Math helpers (VPD, etc.)
│   │   ├── preprocessing/
│   │   └── training/
│   ├── data/                    # Data Storage
│   │   └── models/              # Trained model files
│   ├── config.py                # Central Configuration
│   ├── prefire/                 # Pre-fire specific modules
│   ├── postfire/                # Post-fire specific modules
│   └── firedetect/              # Fire detection modules
├── frontend/                     # Streamlit Dashboard
│   └── app.py                   # Main dashboard application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Google Earth Engine
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT=your_service_account
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH=path/to/key.json

# NASA FIRMS
NASA_FIRMS_API_KEY=your_api_key

# Weather API
WEATHER_API_KEY=your_openweather_key
```

## 🔍 Troubleshooting

### Models Not Loading
- System runs in MOCK mode when models are not present
- To enable full ML mode, place trained models in `backend/data/models/pre_fire/` or the configured directory:
  - `catboost_s_tier_model.pkl` (or `model.cbm`)
  - Associated JSON configurations (`optimal_threshold_info.json`, etc.)

## 📚 Documentation

### Module Overview

- **`backend.src.data_collection`**: Handles all external API interactions.
    - `weather_api.py`: Fetches current and historical weather data.
    - `gee_extractor.py`: Interfaces with Google Earth Engine for satellite data.
- **`backend.src.utils`**:
    - `feature_engineering.py`: **Critical Driver**. Orchestrates data collection to assemble the 81-feature vector required by the model.
- **`backend.src.models`**:
    - `catboost_deployment_tuned.py`: Wrapper for the CatBoost model, handling loading, preprocessing, and risk probability estimation.
- **`backend.src.analysis`**:
    - `pre_fire.py`: High-level analyzer that uses `FeatureEngineer` and `CatBoostPredictor` to generate user-facing risk assessments.

### Data Flow (Pre-Fire Risk)
1. **User** clicks on map (Frontend).
2. **App** calls `FeatureEngineer.get_all_features(lat, lon)`.
3. **FeatureEngineer** fetches:
    - Weather (Open-Meteo)
    - Vegetation & Terrain (GEE)
    - Derived stats (VPD, Rolling Averages)
4. **App** calls `PreFireAnalyzer.predict_from_features()`.
5. **Analyzer** passes data to `CatBoostPredictor`.
6. **Model** returns probability and risk level.

## 🐛 Debugging and Testing

### 1. Integration Tests
We have included a full system integration test to verify the pipeline from data collection to prediction.

```bash
# Run the full system test
python tests/test_simple.py
```

**Expected Output:**
- `[1/3] Testing Feature Engineering...`: Should print fetched features.
- `[2/3] Testing Pre-Fire Analyzer...`: Should print risk probability and level.
- `✅ SYSTEM INTEGRATION TEST PASSED`

### 2. Mock Mode
If external APIs (GEE, Weather) or Model files are missing, the system automatically falls back to **Mock Mode**.
- **Check Logs**: Look for "Returning mock data" or "Model not loaded" warnings in the console.
- **Verify UI**: The frontend will still function but will show simulated data.

### 3. Common Issues
- **GEE Authentication Error**: Ensure your service account JSON key is valid and the path in `.env` is correct.
- **Weather API Error**: Verify your API key in `.env`.
- **Model Loading Failed**: Ensure `catboost_s_tier_model.pkl` is in `src/models/` or the configured directory.

## 📄 License

This project is for educational and research purposes.

---

**Built with:** Python • Streamlit • CatBoost • Google Earth Engine • NASA FIRMS

# Wildfire Management System 🔥

> A comprehensive system for wildfire risk assessment, detection, and spread prediction using machine learning and satellite data, integrated into a single Streamlit dashboard.

## 🌟 Features

- **🔍 Active Fire Detection** - Real-time monitoring using NASA FIRMS satellite data (VIIRS/MODIS)
- **⚠️ Pre-Fire Risk Assessment** - ML ensemble model (CatBoost, LightGBM, XGBoost) with 11 environmental features
- **📊 Post-Fire Spread Prediction** - U-Net deep learning model for spatial fire spread forecasting
- **🗺️ Interactive Dashboard** - Streamlit web interface with Folium maps
- **🌍 Google Earth Engine Integration** - 11 environmental features from multiple satellite sources

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Earth Engine account (for environmental data)
- NASA FIRMS API key (for fire detection)

### Installation

```bash
# 1. Clone the repository
cd "f:\minor project wildfire management v3"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Application

```bash
# Start the Integrated Dashboard
streamlit run frontend/app.py --server.port 8501
```

**Access the Application:**
- 🌐 **Dashboard**: http://localhost:8501

## 📁 Project Structure

```
f:\minor project wildfire management v3\
├── src/                          # Application Source Code
│   ├── analysis/                # Analysis modules
│   │   ├── pre_fire.py         # Pre-fire risk assessment
│   │   ├── post_fire.py        # Post-fire burn severity (dNBR)
│   │   └── spread_prediction.py # Fire spread forecasting
│   ├── data_collection/         # Data collection modules
│   │   ├── gee_extractor.py    # Google Earth Engine integration
│   │   └── nasa_firms.py       # NASA FIRMS fire data
│   ├── models/                  # Model classes
│   │   ├── pre_fire/
│   │   │   └── ensemble_model.py # Ensemble ML model
│   │   └── unet_fire.py        # U-Net architecture
│   ├── preprocessing/
│   ├── training/
│   └── utils/
├── data/                         # Data Storage
│   ├── models/                  # Trained model files
│   ├── raw/                    # Raw satellite data
│   ├── processed/              # Processed datasets
│   └── training/               # Training datasets
├── scripts/                      # Utility Scripts
├── frontend/                     # Streamlit Dashboard
│   └── app.py                  # Main dashboard application
├── tests/                        # Test Suite
├── config.py                     # Central Configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔧 Configuration

### Central Configuration (`config.py`)
All paths and settings are managed through the central `config.py` file.

### Environment Variables (`.env`)
```env
# Google Earth Engine
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT=your_service_account
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH=path/to/key.json

# NASA FIRMS
NASA_FIRMS_API_KEY=your_api_key
```

## 🧪 Testing

```bash
# Run all tests
pytest
```

## 🔍 Troubleshooting

### Models Not Loading
- System runs in MOCK mode when models are not present
- To enable full ML mode, place trained models in `data/models/pre_fire/`:
  - `catboost_s_tier_model.pkl`
  - `lightgbm_best_model.pkl`
  - `xgboost_enhanced_model.joblib`

## 📄 License

This project is for educational and research purposes.

---

**Built with:** Python • Streamlit • PyTorch • Scikit-learn • Google Earth Engine • NASA FIRMS

**Status**: ✅ Operational | ⚠️ Models in MOCK mode

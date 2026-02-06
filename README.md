# Wildfire Management System 🔥

> A comprehensive system for wildfire risk assessment, detection, and spread prediction using machine learning and satellite data

## 🌟 Features

- **🔍 Active Fire Detection** - Real-time monitoring using NASA FIRMS satellite data (VIIRS/MODIS)
- **⚠️ Pre-Fire Risk Assessment** - ML ensemble model (CatBoost, LightGBM, XGBoost) with 11 environmental features
- **📊 Post-Fire Spread Prediction** - U-Net deep learning model for spatial fire spread forecasting
- **🗺️ Interactive Dashboard** - Streamlit web interface with Folium maps
- **🔌 REST API** - FastAPI backend with comprehensive endpoints
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

**Option 1: Manual Start (Recommended for Development)**

```bash
# Terminal 1: Start Backend API
python -m uvicorn api.backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Frontend Dashboard
streamlit run frontend/app.py --server.port 8501
```

**Option 2: Docker Deployment**

```bash
docker-compose up --build
```

**Access the Application:**
- 🌐 **Dashboard**: http://localhost:8501
- 🔌 **API Docs**: http://localhost:8000/docs
- 📖 **ReDoc**: http://localhost:8000/redoc

## 📁 Project Structure

```
f:\minor project wildfire management v3\
├── api/                          # FastAPI Backend
│   ├── backend/
│   │   └── main.py              # Server entry point
│   └── routes/
│       └── predictions.py       # API endpoints
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
├── data/                         # Data Storage (NEW)
│   ├── models/                  # Trained model files
│   │   ├── pre_fire/           # Ensemble models
│   │   ├── post_fire/          # dNBR models
│   │   └── spread/             # U-Net models
│   ├── raw/                    # Raw satellite data
│   ├── processed/              # Processed datasets
│   └── training/               # Training datasets
├── scripts/                      # Utility Scripts (NEW)
│   ├── training/               # Model training scripts
│   ├── data_collection/        # Data collection scripts
│   └── analysis/               # Analysis scripts
├── frontend/                     # Streamlit Dashboard
│   └── app.py                  # Main dashboard application
├── tests/                        # Test Suite
├── config.py                     # Central Configuration (NEW)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔧 Configuration

### Central Configuration (`config.py`)
All paths and settings are managed through the central `config.py` file:
- Automatic path resolution relative to project root
- Model directory paths
- API configuration
- Environment variables

### Environment Variables (`.env`)
```env
# Google Earth Engine
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT=your_service_account
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH=path/to/key.json

# NASA FIRMS
NASA_FIRMS_API_KEY=your_api_key
```

## 📡 API Endpoints

### Health Check
- `GET /api/health` - Server health status

### Environmental Data
- `GET /predictions/environmental` - Get 11 environmental features from GEE

### Pre-Fire Risk Assessment
- `POST /predictions/pre-fire/risk-map` - Generate risk map with ensemble model
- `POST /predictions/predict` - Simple risk prediction

### Active Fire Detection
- `GET /predictions/fires/active` - Get active fires from NASA FIRMS

### Post-Fire Analysis
- `POST /predictions/post-fire/assessment` - Burn severity analysis (dNBR)

### Fire Spread Prediction
- `POST /predictions/spread/next-day` - Predict fire spread using U-Net

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific endpoints
python test_integration.py
python test_all_endpoints.py
```

## 📊 Environmental Features

The system uses 11 environmental features from multiple satellite sources:

1. **Temperature** (Max/Min) - GRIDMET
2. **Humidity** - GRIDMET
3. **Wind Speed** - GRIDMET
4. **Wind Direction** - GRIDMET
5. **Precipitation** - GRIDMET
6. **Vegetation (NDVI)** - MODIS
7. **Elevation** - SRTM
8. **Drought Index (PDSI)** - GRIDMET
9. **Energy Release Component** - GRIDMET
10. **Population Density** - GPWv4

## 🤖 Machine Learning Models

### Pre-Fire Risk Assessment
- **Ensemble Model**: Weighted combination of CatBoost (50%), LightGBM (30%), XGBoost (20%)
- **Features**: 11 environmental parameters
- **Output**: Risk score (0-1) and risk level (Low/Medium/High)

### Fire Spread Prediction
- **Architecture**: U-Net with PyTorch
- **Input**: 12-channel spatial data (128x128)
- **Output**: Predicted fire spread polygon

## 🐳 Docker Deployment

```bash
# Build and start services
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild from scratch
docker-compose build --no-cache
```

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc
- **Google Earth Engine Setup**: See `GUIDE_GEE_AUTH.md`

## 🔍 Troubleshooting

### Models Not Loading
- System runs in MOCK mode when models are not present
- To enable full ML mode, place trained models in `data/models/pre_fire/`:
  - `catboost_s_tier_model.pkl`
  - `lightgbm_best_model.pkl`
  - `xgboost_enhanced_model.joblib`

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use
```bash
# Change ports in commands:
# Backend: --port 8001
# Frontend: --server.port 8502
```

## 🤝 Contributing

1. Train models using scripts in `scripts/training/`
2. Add new features to `src/`
3. Update tests in `tests/`
4. Run `pytest` to verify
5. Submit improvements

## 📄 License

This project is for educational and research purposes.

---

**Built with:** Python • FastAPI • Streamlit • PyTorch • Scikit-learn • Google Earth Engine • NASA FIRMS

**Status**: ✅ Backend Running | ✅ Frontend Running | ⚠️ Models in MOCK mode

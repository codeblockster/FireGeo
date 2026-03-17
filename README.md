# 🔥 FireGeo AI — Wildfire Detection & Risk Assessment System (v4)

A comprehensive wildfire intelligence platform built with **React + TypeScript**, **FastAPI**, and **machine learning**. FireGeo integrates NASA FIRMS satellite data, Open-Meteo weather APIs, Google Earth Engine (GEE) environmental data, and AI-powered ensemble models for real-time fire detection, risk prediction, and post-fire spread simulation.

![Version](https://img.shields.io/badge/version-4.0.0-red)
![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)

---

## 📋 Table of Contents

- [🌟 Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Quick Start](#-quick-start)
- [💻 Technology Stack](#-technology-stack)
- [🔌 API Documentation](#-api-documentation)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🌟 Features

### 🔥 Active Fire Detection
- **Real-time Detection** via NASA FIRMS satellite imagery (VIIRS SNPP, VIIRS NOAA20, MODIS NRT)
- **Time Frame Selection**: 24h, 48h, 72h, or 7-day detection windows
- **Fire Intensity & Confidence Scores** displayed on an interactive Leaflet map
- **Global Regions**: World, Nepal, Australia, California, Indonesia, India

### ⚠️ AI-Powered Pre-Fire Risk Assessment
- **Ensemble ML Models**: CatBoost (primary), XGBoost, LightGBM trained on **81 environmental features**
- **Risk Levels**: Critical (≥80), High (60–79), Medium (40–59), Low (<40)
- **Feature Engineering**: Lag variables, rolling averages, 20+ derived features
- **GEE Integration**: 11 live data sources — MODIS, SRTM, GRIDMET, VIIRS, GPWv4
- **Click-to-Assess**: Click any point on the map to get instant risk predictions

### 🌊 Post-Fire Spread Prediction (Cellular Automata)
- **CA Simulation**: Simulates fire propagation outward from ignition point over configurable time steps
- **RF Ensemble Integration**: Queries real-time GEE features per cell for spread probability
- **Wind Bias**: Directional wind multipliers shape asymmetric spread patterns
- **Risk Zones**: Visualized as merged polygons — Critical (red), High (orange), Moderate (yellow)
- **Interactive Visualization**: Animated zone overlay rendered in React via Leaflet

| Component | Details |
|-----------|---------|
| Model | RF + Extra Trees Ensemble (81 features) |
| Grid | Moore Neighborhood (8 directions) |
| Cell Size | Default 0.0045° (~500m) - High resolution |
| Time Steps | Configurable (default: 5) |
| Risk Threshold | High >0.75, Medium 0.50–0.75, Low <0.50 |
| Data Source | Google Earth Engine (live) |

**Wind Bias Formula:**
```
multiplier = 0.7 + (180 - angular_difference) / 180.0 * 0.5
```
- Frontal (with wind): 1.2× boost
- Flanking (90° off wind): 1.0× neutral
- Backing (against wind): 0.7× penalty

### 🌤️ Environmental Monitoring
- **Real-time Weather**: Temperature, humidity, wind speed/direction via Open-Meteo
- **Vegetation Indices**: NDVI, GNDVI, SAVI, EVI, NBR, NDWI, NDSI
- **Drought Monitoring**: Palmer Drought Severity Index (PDSI)
- **Soil Conditions**: Soil temperature and moisture at multiple depths
- **Historical Trends**: 14-day and 30-day analysis

### 🗺️ Interactive Map
- **Map Styles**: Dark Mode, Satellite View (Esri), Light Mode
- **Border Toggle**: Country border layers on/off
- **Predefined Regions**: Nepal (Kathmandu, Pokhara, Chitwan, Himalayan), Australia, California, Indonesia, India

### 🎨 Modern UI
- **Glass Morphism** with backdrop blur effects
- **Smooth Animations** via Framer Motion
- **Animated Risk Gauge**: Circular progress indicator
- **Dark-first Design** with vibrant accent colors
- **Responsive Layout** for all screen sizes

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   FRONTEND (React + TypeScript + Vite)           │
│  ┌────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐  │
│  │ Navbar │ │ ControlPanel │ │  WeatherTab │ │ PostFireTab  │  │
│  └────────┘ └──────────────┘ └─────────────┘ └──────────────┘  │
│                      │                                           │
│            ┌──────────────────────┐                             │
│            │    Map.tsx (Leaflet) │                             │
│            │  Fire/Risk/Spread    │                             │
│            └──────────────────────┘                             │
│                      │                                           │
│         ┌────────────────────────┐                              │
│         │    Zustand Store       │  React Query Hooks           │
│         │  mode, location, data  │  useDetectFires()            │
│         └────────────────────────┘  useAssessRisk()             │
│                                     useEnvData()                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP (Vite Proxy → localhost:8000)
┌──────────────────────────┴───────────────────────────────────────┐
│                       BACKEND (FastAPI)                          │
│  /api/detect-fires  /api/env-data  /api/assess-risk             │
│  /api/weather       /api/post-fire-spread                        │
│                                                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ FireDetector│  │ PreFireAnalyzer  │  │ ActiveFireCA     │   │
│  │ (NASA FIRMS)│  │ (CatBoost/XGB/  │  │ (CA Simulation + │   │
│  └─────────────┘  │  LightGBM)       │  │  RF Ensemble)    │   │
│                   └──────────────────┘  └──────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               DATA COLLECTION LAYER                       │  │
│  │  NASA FIRMS API  │  Open-Meteo API  │  Google Earth Engine│  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Fire Detection**: `User → /api/detect-fires → NASA FIRMS → Map markers`
2. **Risk Assessment**: `User clicks map → /api/assess-risk → GEE + CatBoost → Risk gauge`
3. **Fire Spread**: `User sets ignition point → /api/post-fire-spread → CA simulation → Risk zone polygons`
4. **Weather**: `Location selected → /api/weather + /api/env-data → Open-Meteo + GEE → WeatherTab`

---

## 🚀 Quick Start

### Option 1: Windows Startup Script (Recommended)

Double-click `run_app.bat` from the project root:

```cmd
run_app.bat
```

This will:
- Start the FastAPI backend (`backend/main.py`) on port **8000**
- Start the Vite frontend dev server on port **5173**

See [`STARTUP.md`](STARTUP.md) for detailed startup instructions.

### Option 2: Manual Setup

#### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| NASA FIRMS API Key | Required (free) |
| Google Earth Engine | Optional (recommended) |

#### Backend

```cmd
cd backend
..\venv_py311\Scripts\python.exe main.py
```

Or with your own venv:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## 💻 Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18 | UI Framework |
| TypeScript | 5.3 | Type Safety |
| Vite | 5 | Build Tool & Dev Server |
| TailwindCSS | 3.4 | Styling |
| Framer Motion | 11 | Animations |
| React-Leaflet | 4 | Interactive Maps |
| Zustand | 4 | State Management |
| React Query | 5 | Data Fetching & Caching |
| React Hot Toast | — | Notifications |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115 | Web Framework |
| Uvicorn | 0.32 | ASGI Server |
| Pydantic | 2.x | Data Validation |
| Python | 3.11+ | Runtime |

### ML / Data Science

| Technology | Purpose |
|------------|---------|
| CatBoost | Primary risk prediction model |
| XGBoost | Ensemble model |
| LightGBM | Fast gradient boosting |
| Scikit-learn | RF/Extra Trees ensemble (post-fire CA) |
| Pandas / NumPy | Data processing |

### External APIs

| API | Purpose |
|-----|---------|
| NASA FIRMS | Active fire satellite data |
| Open-Meteo | Weather & climate data |
| Google Earth Engine | Environmental satellite data |

---

## 🔌 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/api/detect-fires` | Active fire detection |
| POST | `/api/env-data` | Environmental/weather data |
| GET | `/api/weather` | Standalone weather data |
| POST | `/api/assess-risk` | AI risk assessment |
| POST | `/api/post-fire-spread` | Fire spread simulation (CA) |

---

### `POST /api/detect-fires`

```json
// Request
{
  "location": { "id": "np", "name": "Nepal", "lat": 28.3949, "lng": 84.124 },
  "hours": 24
}

// Response
{
  "fires": [
    { "id": "fire-27.5-85.3", "lat": 27.5, "lng": 85.3, "intensity": 75,
      "confidence": 85, "satellite": "VIIRS", "frp": 45.2, "brightness": 362.5 }
  ],
  "count": 1,
  "source": "NASA FIRMS (VIIRS_SNPP_NRT)",
  "timestamp": "2025-03-05T..."
}
```

---

### `POST /api/assess-risk`

```json
// Request
{
  "location": { "id": "np", "name": "Kathmandu Valley", "lat": 27.7172, "lng": 85.324 },
  "envData": { "temperature": 25, "humidity": 35, "windSpeed": 15, "windDirection": 180 }
}

// Response
{
  "risk": {
    "level": "high", "score": 72, "probability": 0.72,
    "factors": { "weather": 65, "vegetation": 50, "topography": 55, "historical": 72 }
  },
  "timestamp": "2025-03-05T..."
}
```

---

### `POST /api/post-fire-spread`

```json
// Request
{
  "latitude": 27.7172,
  "longitude": 85.324,
  "wind_direction": 90,
  "wind_speed": 15,
  "time_steps": 5,
  "cell_size_deg": 0.005
}

// Response
{
  "ignition_point": { "latitude": 27.7172, "longitude": 85.324 },
  "spread_points": [
    { "latitude": 27.7227, "longitude": 85.329, "probability": 95, "time_step": 1 }
  ],
  "spread_radius_km": 12.5,
  "spread_probability": 78.5,
  "model_info": {
    "model_type": "ActiveFireCA (Cellular Automata + RF/ET Ensemble)",
    "features": "81 environmental features from GEE",
    "spread_logic": "Moore Neighborhood (8-direction)"
  },
  "timestamp": "2025-03-05T..."
}
```

---

### `GET /api/weather?lat=28.39&lon=84.12`

```json
{
  "data": {
    "temp": 18.5, "humidity": 45, "windSpeed": 12, "windDirection": 180,
    "dewpoint": 6.2, "pressure": 1013.25, "precipitation": 0.0,
    "soilTemp": 15.3, "soilMoisture": 0.28
  },
  "timestamp": "2025-03-05T..."
}
```

---

## 📁 Project Structure

```
v4 cleanup/
├── frontend/                         # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── Map.tsx              # Leaflet map with fire/risk/spread layers
│   │   │   ├── ControlPanel.tsx     # Main panel (location, mode, fire/risk controls)
│   │   │   ├── Navbar.tsx           # Top navigation bar
│   │   │   ├── WeatherTab.tsx       # Weather & environmental data
│   │   │   ├── PostFireTab.tsx      # Post-fire spread simulation UI
│   │   │   └── ui/
│   │   │       └── GlassCard.tsx    # Reusable glass-morphism card
│   │   ├── hooks/
│   │   │   └── useApi.ts            # React Query hooks for all API calls
│   │   ├── store/
│   │   │   └── useStore.ts          # Zustand global state store
│   │   ├── App.tsx                  # Root app component
│   │   ├── main.tsx                 # Entry point
│   │   └── index.css                # Global styles (Tailwind + custom)
│   ├── public/
│   │   ├── fire-icon.svg
│   │   └── fire-icon.png
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
│
├── backend/                          # FastAPI Backend
│   ├── main.py                      # All API endpoints
│   ├── config.py                    # Paths & configuration constants
│   ├── requirements.txt
│   ├── firedetect/
│   │   └── fire_detector.py         # NASA FIRMS integration
│   ├── prefire/
│   │   ├── pre_fire_analyzer.py     # CatBoost risk analysis pipeline
│   │   ├── catboost_predictor.py    # CatBoost model wrapper
│   │   ├── feature_engineer.py      # Feature engineering (81 features)
│   │   ├── calculations.py          # Risk score calculations
│   │   └── models/                  # Trained ML models (.pkl, .json)
│   ├── postfire/
│   │   ├── data_collector.py        # GEE data collection for CA
│   │   ├── models/
│   │   │   └── active_fire_ca.py    # Cellular Automata + RF spread model
│   │   └── ...
│   └── src/
│       └── data_collection/
│           ├── nasa_firms.py        # NASA FIRMS API
│           ├── weather_api.py       # Open-Meteo API
│           ├── gee_extractor.py     # Google Earth Engine extraction
│           └── sentinel_manager.py  # Sentinel satellite data
│
├── STARTUP.md                        # Detailed startup guide
├── run_app.bat                       # Windows one-click launcher
├── requirements.txt                  # Root Python dependencies
├── model_features.txt                # 81 ML feature documentation
├── authenticate_gee.py               # GEE authentication helper
└── .env                              # Environment variables (not committed)
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# NASA FIRMS API
NASA_FIRMS_API_KEY=your_api_key_here

# Google Earth Engine (optional, but recommended)
GEE_PROJECT=your_gee_project_id
GEE_SERVICE_ACCOUNT=your_service_account@project.iam.gserviceaccount.com
GEE_KEY_FILE=path/to/service_account_key.json
```

### Key Config (`backend/config.py`)

- `MODEL_DIR`: Path to trained ML model files
- `CATBOOST_MODEL_PATH`: Primary CatBoost model
- `RF_MODEL_PATH`: RF ensemble model for post-fire spread
- `NASA_FIRMS_BASE_URL`: FIRMS API base URL

### GEE Authentication

```bash
python authenticate_gee.py
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| Backend won't start | Check Python 3.11+ is installed; ensure `backend/requirements.txt` deps are installed |
| GEE errors | Run `python authenticate_gee.py`; verify project/service account credentials |
| Fire data not loading | Verify `NASA_FIRMS_API_KEY` in `.env` is valid |
| CORS errors | Ensure backend is running on port 8000 and frontend proxy in `vite.config.ts` is configured |
| `ModuleNotFoundError` | Activate virtual environment and run `pip install -r requirements.txt` |
| Fire spread returns no data | GEE must be authenticated; check `backend/config.py` model paths |

---

## 📄 License

MIT License. See `LICENSE` for details.

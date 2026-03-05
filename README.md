# 🔥 FIRE ALERT - Wildfire Detection & Risk Assessment System

A comprehensive wildfire detection and risk assessment platform built with React, FastAPI, and machine learning. The system integrates NASA FIRMS satellite data, Open-Meteo weather APIs, Google Earth Engine environmental data, and AI-powered CatBoost models for real-time fire risk prediction.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![React](https://img.shields.io/badge/React-18.2-blue)
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
- [🎨 UI/UX Design](#-uiux-design)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Features

### 🔥 Fire Detection

- **Real-time Detection**: Identifies active fires using NASA FIRMS satellite imagery
- **Multiple Time Frames**: Select detection periods - 24h, 48h, 72h, or 7 days
- **Intensity Mapping**: Visual representation of fire intensity levels (0-100 scale)
- **Confidence Scores**: Shows detection confidence percentage (VIIRS/MODIS)
- **Multiple Satellite Sources**: VIIRS SNPP, VIIRS NOAA20, MODIS NRT
- **Global Coverage**: Pre-configured regions including World, Nepal, California, Australia, Indonesia, India

### ⚠️ AI-Powered Risk Assessment

- **CatBoost ML Model**: Trained on 81 environmental features
- **Multi-factor Analysis**: Evaluates weather, vegetation, topography, and historical data
- **Risk Levels**: Categorizes risk as Critical (80+), High (60-79), Medium (40-59), or Low (<40)
- **Feature Engineering**: 20+ derived features including lag variables and rolling averages
- **Google Earth Engine**: Integration for 11 environmental data sources (MODIS, SRTM, GRIDMET, VIIRS, GPWv4)

### 🌊 Active Fire Spread Prediction (Cellular Automata)

- **Grid Simulation**: Simulates outward fire propagation over configurable time steps.
- **RF Ensemble Integration**: Queries the real-time ML risk model (Live GEE features) for each cell to determine spread likelihood.
- **Wind Modification**: Calculates directional biases based on real-time Wind Speed & Direction to shape accurate, asymmetrical spreads.
- **Interactive Visualization**: Animated point-probability layout rendered seamlessly in React.

#### Technical Details

The Active Fire Spread Prediction module uses a Cellular Automata (CA) approach combined with a pre-trained Random Forest (RF) ensemble model:

| Component | Description |
|-----------|-------------|
| **Model** | RF + Extra Trees Ensemble (81 features) |
| **Grid** | Moore Neighborhood (8 directions) |
| **Cell Size** | Default 0.01° (~1.1 km) |
| **Time Steps** | Configurable (default: 5) |
| **Risk Threshold** | High: >0.75, Medium: 0.50-0.75, Low: <0.50 |
| **Data Source** | Google Earth Engine (GEE) |

##### Propagation Rules

- **High Risk (RF prob > 0.75)**: Fire spreads rapidly to adjacent cell in current time step
- **Medium Risk (0.50 < RF prob ≤ 0.75)**: Fire spreads with delay to next time step
- **Low Risk (RF prob ≤ 0.50)**: Fire spread is halted in this direction

##### Wind Bias Calculation

The model calculates wind-adjusted spread probability using:
```
multiplier = 0.7 + (180 - angular_difference) / 180.0 * 0.5
```
- Frontal spread (wind direction): 1.2x boost
- Flanking spread (90° from wind): 1.0x (neutral)
- Backing spread (opposite to wind): 0.7x penalty

### 🗺️ Interactive Map

- **Multiple Map Styles**:
  - 🌙 Dark Mode - Default dark theme for fire visualization
  - 🛰️ Satellite View - Real satellite imagery from Esri
  - ☀️ Light Mode - Light color scheme for daytime use
- **Border Controls**: Toggle country borders on/off
- **Location Selection**: Predefined regions including:
  - World (Global view)
  - Nepal (Kathmandu Valley, Pokhara, Chitwan, Himalayan Region)
  - Australia
  - California (USA)
  - Indonesia
  - India
- **Click-to-Assess**: Click anywhere on the map in Risk mode to get instant risk assessment

### 🌤️ Environmental Monitoring

- **Real-time Weather**: Temperature, humidity, wind speed/direction from Open-Meteo
- **Vegetation Indices**: NDVI, GNDVI, SAVI, EVI, NBR, NDWI, NDSI
- **Drought Monitoring**: Palmer Drought Severity Index (PDSI)
- **Soil Conditions**: Soil temperature and moisture at multiple depths
- **Historical Analysis**: 14-day and 30-day historical data for trend analysis

### 🎨 Modern UI

- **Pixel OS Color Scheme**: Vibrant retro-inspired color palette
- **Glass Morphism**: Modern frosted glass effects with backdrop blur
- **Smooth Animations**: Fluid transitions using Framer Motion
- **Risk Gauge**: Animated circular progress indicator for risk scores
- **Responsive Design**: Adapts to different screen sizes

---

## 🏗️ System Architecture

```mermaid
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React + TypeScript)                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                           React Components                                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐  ││
│  │  │   Navbar    │  │     Map     │  │  ControlPanel   │  │ WeatherTab  │  ││
│  │  │  Component  │  │  (Leaflet)  │  │   (Risk/Fire)   │  │             │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  └─────────────┘  ││
│  │                                      │                                         ││
│  │                    ┌─────────────────┴─────────────────┐                       ││
│  │                    │         Zustand Store            │                       ││
│  │                    │  - Mode (fire/risk)             │                       ││
│  │                    │  - MapStyle (dark/sat/light)    │                       ││
│  │                    │  - SelectedLocation             │                       ││
│  │                    │  - FireLocations                 │                       ││
│  │                    │  - RiskAssessment               │                       ││
│  │                    └─────────────────────────────────┘                       ││
│  │                                      │                                         ││
│  │                    ┌─────────────────┴─────────────────┐                       ││
│  │                    │        React Query Hooks            │                       ││
│  │                    │  - useDetectFires()                 │                       ││
│  │                    │  - useEnvData()                      │                       ││
│  │                    │  - useAssessRisk()                  │                       ││
│  │                    └─────────────────────────────────┘                       ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                       │                                            │
│                              ┌────────┴────────┐                                   │
│                              │   Vite Proxy    │                                   │
│                              │ (localhost:5173)│                                   │
│                              └────────┬────────┘                                   │
└───────────────────────────────────────┼─────────────────────────────────────────────┘
                                        │ HTTP Requests
┌───────────────────────────────────────┼─────────────────────────────────────────────┐
│                               BACKEND (FastAPI)                                    │
│                          ┌────────────┴────────────┐                               │
│                          │      Main API Endpoints  │                               │
│                          │  - /api/detect-fires     │                               │
│                          │  - /api/env-data         │                               │
│                          │  - /api/assess-risk      │                               │
│                          │  - /api/weather          │                               │
│                          └────────────┬────────────┘                               │
│                                       │                                             │
│    ┌──────────────────────────────────┼──────────────────────────────────────┐     │
│    │                        CORE MODULES                                      │     │
│    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐│     │
│    │  │  FireDetector  │  │  PreFireAnalyzer│  │   WeatherDataFetcher    ││     │
│    │  │  (NASA FIRMS)  │  │   (CatBoost)    │  │    (Open-Meteo)         ││     │
│    │  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘│     │
│    │           │                     │                        │              │     │
│    └───────────┼─────────────────────┼────────────────────────┼──────────────┘     │
│                │                     │                        │                    │
│    ┌───────────┴─────────────────────┴────────────────────────┴──────────────┐   │
│    │                        DATA COLLECTION LAYER                              │   │
│    │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │   │
│    │  │  NASA FIRMS API  │  │ Open-Meteo API   │  │ Google Earth Engine  │ │   │
│    │  │  - VIIRS SNPP    │  │  - Forecast      │  │  - MODIS              │ │   │
│    │  │  - VIIRS NOAA20  │  │  - Archive       │  │  - SRTM               │ │   │
│    │  │  - MODIS NRT     │  │  - Historical    │  │  - GRIDMET            │ │   │
│    │  └──────────────────┘  └──────────────────┘  │  - VIIRS               │ │   │
│    │                                                │  - GPWv4              │ │   │
│    │                                                └───────────────────────┘ │   │
│    │                                                                          │   │
│    │  ┌──────────────────────────────────────────────────────────────────────┐│   │
│    │  │                    ML MODELS (CatBoost/XGBoost/LightGBM)           ││   │
│    │  │                    81 Features - Trained Risk Prediction Model      ││   │
│    │  └──────────────────────────────────────────────────────────────────────┘│   │
│    └──────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Fire Detection Flow**:
   ```
   User Selects Location → API Request → FireDetector → NASA FIRMS API → Response → Map Display
   ```

2. **Risk Assessment Flow**:
   ```
   User Clicks Map/Location → API Request → PreFireAnalyzer → Feature Engineering → 
   CatBoost Model → Risk Prediction → Response → Risk Gauge Display
   ```

3. **Environmental Data Flow**:
   ```
   Location Selected → API Request → WeatherDataFetcher → Open-Meteo API → 
   GEE Extractor (optional) → Response → Weather Tab Display
   ```

---

## 🚀 Quick Start

### Option 1: Use Startup Script (Windows - Recommended)

Simply double-click `run_app.bat` in the project root to start both servers.

```cmd
run_app.bat
```

### Option 2: Manual Start

#### Prerequisites

- **Python**: 3.11+ (with pip)
- **Node.js**: 18+ (with npm)
- **NASA FIRMS API Key**: Required for fire detection (free registration)
- **Google Earth Engine** (Optional): For enhanced environmental data
- **Virtual Environment**: Use `venv_py311` provided in project

#### Backend Setup (Manual)

```cmd
# Navigate to project root
cd "New folder/v4 cleanup"

# Navigate to backend folder
cd backend

# Run backend using the provided virtual environment
..\venv_py311\Scripts\python.exe main.py
```

Or if you want to create your own venv:

```bash
# Navigate to project root
cd "New folder/v4 cleanup"

# Navigate to backend
cd backend

# Create virtual environment (optional)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python main.py
```

#### Frontend Setup (Manual)

```bash
# Open a new terminal
cd frontend

# Install dependencies (if not already installed)
npm install

# Run frontend
npm run dev
```

### Access Points

After starting both servers:

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## 💻 Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2 | UI Framework |
| TypeScript | 5.3 | Type Safety |
| Vite | 5.0 | Build Tool & Dev Server |
| TailwindCSS | 3.4 | Styling |
| Framer Motion | 10.0 | Animations |
| React-Leaflet | 4.2 | Interactive Maps |
| Zustand | 4.4 | State Management |
| React Query | 5.0 | Data Fetching & Caching |
| React Hot Toast | - | Notifications |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115 | Web Framework |
| Uvicorn | 0.32 | ASGI Server |
| Pydantic | 2.10 | Data Validation |
| Python | 3.11+ | Runtime |

### ML/Data Science (Backend)

| Technology | Purpose |
|------------|---------|
| CatBoost | Primary Risk Prediction Model |
| XGBoost | Alternative Model |
| LightGBM | Fast Ensemble Model |
| Scikit-learn | ML Pipeline |
| Pandas | Data Processing |
| NumPy | Numerical Computing |

### External APIs

| API | Purpose |
|-----|---------|
| NASA FIRMS | Active Fire Satellite Data |
| Open-Meteo | Weather & Climate Data |
| Google Earth Engine | Environmental Satellite Data |

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check with module status |
| POST | `/api/detect-fires` | Detect active fires in region |
| POST | `/api/env-data` | Get environmental/weather data |
| GET | `/api/weather` | Get standalone weather data |
| POST | `/api/assess-risk` | AI-powered risk assessment |
| POST | `/api/post-fire-spread` | Active fire spread prediction (Cellular Automata) |

---

### 1. Root Endpoint

```http
GET /
```

**Response:**

```json
{
  "message": "Wildfire Detection API v2.0",
  "version": "2.0.0",
  "modules_loaded": true,
  "endpoints": {
    "detect-fires": "POST /api/detect-fires - Detect fires using NASA FIRMS",
    "env-data": "POST /api/env-data - Get environmental/weather data",
    "assess-risk": "POST /api/assess-risk - AI-powered risk assessment",
    "weather": "GET /api/weather - Standalone weather data",
    "health": "GET /health - Health check"
  }
}
```

---

### 2. Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000000",
  "modules": {
    "prefire_analyzer": true,
    "fire_detector": true,
    "weather_fetcher": true
  }
}
```

---

### 3. Detect Fires

```http
POST /api/detect-fires
```

**Request Body:**

```json
{
  "location": {
    "id": "np",
    "name": "Nepal",
    "lat": 28.3949,
    "lng": 84.124
  },
  "hours": 24
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| location | object | Yes | Location object |
| location.id | string | Yes | Unique identifier |
| location.name | string | Yes | Location name |
| location.lat | number | Yes | Latitude (-90 to 90) |
| location.lng | number | Yes | Longitude (-180 to 180) |
| hours | integer | No | Time frame in hours (default: 24) |

**Response:**

```json
{
  "fires": [
    {
      "id": "fire-27.5-85.3",
      "lat": 27.5,
      "lng": 85.3,
      "intensity": 75,
      "confidence": 85,
      "timestamp": "2024-01-15T08:30:00",
      "brightness": 362.5,
      "frp": 45.2,
      "satellite": "VIIRS",
      "acq_datetime": "2024-01-15T08:30:00"
    }
  ],
  "count": 1,
  "source": "NASA FIRMS (VIIRS_SNPP_NRT)",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

**Fire Location Properties:**

| Property | Type | Description |
|----------|------|-------------|
| id | string | Unique fire identifier |
| lat | number | Latitude of fire detection |
| lng | number | Longitude of fire detection |
| intensity | integer | Fire intensity (0-100 scale) |
| confidence | integer | Detection confidence (0-100%) |
| timestamp | string | ISO timestamp of detection |
| brightness | number | Brightness temperature (Kelvin) |
| frp | number | Fire Radiative Power (MW) |
| satellite | string | Satellite source (VIIRS/MODIS) |
| acq_datetime | string | Acquisition datetime |

---

### 4. Get Environmental Data

```http
POST /api/env-data
```

**Request Body:**

```json
{
  "location": {
    "id": "np",
    "name": "Nepal",
    "lat": 28.3949,
    "lng": 84.124
  }
}
```

**Response:**

```json
{
  "data": {
    "temperature": 18.5,
    "humidity": 45,
    "windSpeed": 12,
    "windDirection": 180,
    "vegetationIndex": 0.65,
    "droughtIndex": 0.32,
    "dewpoint": 6.2,
    "cloudCover": 15.0,
    "pressure": 1013.25,
    "precipitation": 0.0
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

---

### 5. Get Standalone Weather

```http
GET /api/weather?lat=28.3949&lon=84.124
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| lat | number | Yes | Latitude |
| lon | number | Yes | Longitude |

**Response:**

```json
{
  "data": {
    "temp": 18.5,
    "humidity": 45.0,
    "dewpoint": 6.2,
    "cloudCover": 15.0,
    "windSpeed": 12.0,
    "windDirection": 180.0,
    "windU": 5.2,
    "windV": -10.4,
    "precip": 0.0,
    "skinTemp": 22.1,
    "soilTemp": 15.3,
    "soilMoisture": 0.28
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

---

### 6. Assess Risk

```http
POST /api/assess-risk
```

**Request Body:**

```json
{
  "location": {
    "id": "np",
    "name": "Kathmandu Valley",
    "lat": 27.7172,
    "lng": 85.324
  },
  "envData": {
    "temperature": 25.0,
    "humidity": 35,
    "windSpeed": 15,
    "windDirection": 180,
    "vegetationIndex": 0.5,
    "droughtIndex": 0.7
  }
}
```

**Response:**

```json
{
  "risk": {
    "level": "high",
    "score": 72,
    "probability": 0.72,
    "alert_priority": "High",
    "confidence": "High",
    "factors": {
      "weather": 65,
      "vegetation": 50,
      "topography": 55,
      "historical": 72
    }
  },
  "location": {
    "id": "np",
    "name": "Kathmandu Valley",
    "lat": 27.7172,
    "lng": 85.324
  },
  "features": {
    "lst_day_c": 32.5,
    "relative_humidity_pct": 35,
    "wind_speed_ms": 15,
    "vapor_pressure_deficit_kpa": 2.8,
    "landsat_savi": 0.5,
    ...
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

**Risk Assessment Properties:**

| Property | Type | Description |
|----------|------|-------------|
| level | string | Risk level: "low", "medium", "high", "critical" |
| score | integer | Risk score (0-100) |
| probability | number | Model probability (0-1) |
| alert_priority | string | Alert priority level |
| confidence | string | Model confidence level |
| factors | object | Individual risk factors |

**Risk Factors:**

| Factor | Type | Description |
|--------|------|-------------|
| weather | integer | Weather-related risk (0-100) |
| vegetation | integer | Vegetation/fuel risk (0-100) |
| topography | integer | Terrain-related risk (0-100) |
| historical | integer | Historical fire risk (0-100) |

---

### 6. Post-Fire Spread Prediction (Cellular Automata)

```http
POST /api/post-fire-spread
```

Predicts active fire spread using Cellular Automata with RF ensemble model.

**Request Body:**

```json
{
  "latitude": 27.7172,
  "longitude": 85.324,
  "wind_direction": 90,
  "wind_speed": 15,
  "time_steps": 5,
  "cell_size_deg": 0.01
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| latitude | float | Yes | - | Ignition point latitude (-90 to 90) |
| longitude | float | Yes | - | Ignition point longitude (-180 to 180) |
| wind_direction | float | No | 90 | Wind direction in degrees (0-360) |
| wind_speed | float | No | 15 | Wind speed in km/h |
| time_steps | integer | No | 5 | Number of simulation time steps |
| cell_size_deg | float | No | 0.01 | Grid cell size in degrees (~1.1km) |

**Response:**

```json
{
  "ignition_point": {
    "latitude": 27.7172,
    "longitude": 85.324
  },
  "spread_radius_km": 55.66,
  "spread_probability": 78.5,
  "spread_points": [
    {
      "latitude": 27.7272,
      "longitude": 85.334,
      "probability": 95,
      "time_step": 1
    },
    {
      "latitude": 27.7372,
      "longitude": 85.344,
      "probability": 75,
      "time_step": 2
    }
  ],
  "conditions": {
    "ndvi": 0.45,
    "temperature_celsius": 25.5,
    "humidity_percent": 35,
    "wind_direction_deg": 90,
    "wind_speed_ms": 4.17,
    "data_source": "ActiveFireCA (GEE+RF Ensemble)"
  },
  "wind_direction": 90,
  "wind_speed": 15,
  "time_steps_simulated": 5,
  "model_info": {
    "model_type": "ActiveFireCA (Cellular Automata + RF/ET Ensemble)",
    "model_path": "backend/postfire/models/models/rf_fire_risk_model.pkl",
    "features": "81 environmental features from GEE",
    "spread_logic": "Moore Neighborhood (8-direction)"
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

**Spread Point Properties:**

| Property | Type | Description |
|----------|------|-------------|
| latitude | number | Cell center latitude |
| longitude | number | Cell center longitude |
| probability | integer | Spread probability (15-95%) |
| time_step | integer | Time step when cell ignites (1-n) |

**Spread Rules:**
- **High Risk (prob > 75%)**: Ignites in current time step
- **Medium Risk (50-75%)**: Ignites in next time step
- **Low Risk (< 50%)**: Does not ignite

---

## 📁 Project Structure

```
v4 cleanup/
├── frontend/                         # React Frontend Application
│   ├── src/
│   │   ├── components/              # React Components
│   │   │   ├── Map.tsx             # Interactive Leaflet map component
│   │   │   ├── ControlPanel.tsx    # Main control panel with location/time selection
│   │   │   ├── Navbar.tsx          # Top navigation bar
│   │   │   ├── WeatherTab.tsx      # Weather data display tab
│   │   │   └── ui/
│   │   │       └── GlassCard.tsx   # Reusable glass-morphism UI components
│   │   ├── hooks/
│   │   │   └── useApi.ts          # React Query hooks for API calls
│   │   ├── store/
│   │   │   └── useStore.ts        # Zustand state management store
│   │   ├── App.tsx                # Main application component
│   │   ├── main.tsx               # Application entry point
│   │   └── index.css              # Global styles with Tailwind
│   ├── public/                    # Static assets
│   │   ├── fire-icon.svg          # Fire icon
│   │   └── fire-icon.png          # Fire icon (PNG)
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.ts            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS configuration
│   └── tsconfig.json             # TypeScript configuration
│
├── backend/                        # FastAPI Backend Application
│   ├── main.py                   # FastAPI app with all endpoints
│   ├── config.py                 # Central configuration & paths
│   ├── requirements.txt          # Python dependencies
│   ├── firedetect/               # Fire Detection Module
│   │   ├── __init__.py
│   │   └── fire_detector.py     # Fire detection using NASA FIRMS
│   ├── prefire/                  # Pre-Fire Risk Analysis Module
│   │   ├── __init__.py
│   │   ├── pre_fire_analyzer.py # Main analyzer with CatBoost
│   │   ├── catboost_predictor.py # CatBoost model wrapper
│   │   ├── feature_engineer.py   # Feature engineering pipeline
│   │   ├── calculations.py       # Risk calculations
│   │   └── models/               # Trained ML models
│   │       ├── catboost_s_tier_model.pkl
│   │       ├── lightgbm_best_model.pkl
│   │       ├── xgboost_enhanced_model.pkl
│   │       └── *.json            # Model metrics & hyperparameters
│   └── src/
│       └── data_collection/      # Data Collection Modules
│           ├── __init__.py
│           ├── nasa_firms.py     # NASA FIRMS API integration
│           ├── weather_api.py    # Open-Meteo API integration
│           ├── gee_extractor.py  # Google Earth Engine extraction
│           └── sentinel_manager.py # Sentinel satellite data
│
├── tests/                         # Test Files
│   ├── test_api.py              # API endpoint tests
│   └── debug_firms.py            # NASA FIRMS debugging
│
├── .env.example                  # Environment variables template
├── requirements.txt              # Main Python requirements
├── model_features.txt            # 81 ML features documentation
├── run_app.bat                   # Windows startup script
├── authenticate_gee.py           # GEE authentication script
├── STARTUP.md                    # Quick start guide
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# ========================
# Required for Fire Detection
# ========================
# Get your free API key from: https://firms.modaps.eosdis.nasa.gov/api/area/token
NASA_FIRMS_API_KEY=your_nasa_firms_key_here

# ========================
# Optional: Google Earth Engine
# ========================
# For enhanced environmental data
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH=path/to/your/private/key.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# ========================
# Application Settings
# ========================
# Set to True to disable fallback/mock data
STRICT_MODE=False
```

### Getting API Keys

#### NASA FIRMS API Key

1. Visit https://firms.modaps.eosdis.nasa/token
2..gov/api/area Register for a free account
3. Copy your API token
4. Add to `.env` file as `NASA_FIRMS_API_KEY`

#### Google Earth Engine (Optional)

1. Sign up for Google Earth Engine at https://earthengine.google.com/
2. Create a Google Cloud Project
3. Create a Service Account
4. Download the private key JSON file
5. Add credentials to `.env` file

---

## 🎨 UI/UX Design

### Color Scheme (Pixel OS)

The application uses a vibrant, retro-inspired color palette:

| Color Name | Hex Code | Usage |
|------------|----------|-------|
| Background Dark | `#0f0f23` | Main background |
| Background Secondary | `#1a1a2e` | Cards, panels |
| Accent Purple | `#9b59b6` | Primary highlights, buttons |
| Accent Cyan | `#00d2d3` | Secondary highlights |
| Fire Red | `#ff4757` | Critical risk, fire markers |
| Fire Orange | `#ff6b35` | High risk |
| Fire Yellow | `#ffa502` | Medium risk |
| Success Green | `#2ed573` | Low risk, healthy vegetation |

### Map Controls

The map includes custom controls:

1. **Map Style Selector** (top-right):
   - 🌙 Dark Mode - Best for fire visualization
   - 🛰️ Satellite View - Real satellite imagery
   - ☀️ Light Mode - Light theme

2. **Location Selection** (left panel):
   - Dropdown with predefined regions
   - Custom coordinates support

3. **Time Frame Selection** (fire mode):
   - 24 Hours
   - 48 Hours
   - 72 Hours
   - 7 Days

### Animations

- **Page Load**: Fade-in with scale animation
- **Card Hover**: Subtle lift effect with shadow
- **Button Press**: Scale feedback (0.95)
- **Map Transitions**: Smooth pan and zoom
- **Risk Gauge**: Animated circular progress (1s duration)
- **Risk Factor Bars**: Animated width transition (0.5s)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Backend Won't Start

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

#### 2. NASA FIRMS Returns No Data

**Problem**: Fire detection returns empty results

**Solution**:
1. Check API key is set in `.env` file
2. Verify API key is valid at https://firms.modaps.eosdis.nasa.gov/api/area/token
3. Check network connectivity

#### 3. Frontend Not Connecting to Backend

**Problem**: CORS errors or connection refused

**Solution**:
1. Ensure backend is running on port 8000
2. Check Vite proxy configuration in `vite.config.ts`
3. Clear browser cache

#### 4. Google Earth Engine Errors

**Problem**: GEE authentication failures

**Solution**:
1. Verify service account credentials
2. Ensure private key file path is correct
3. Check Google Cloud Project is set
4. System will fall back to mock data automatically

#### 5. CatBoost Model Not Loading

**Problem**: Risk assessment shows "model not available"

**Solution**:
1. Check model file exists: `backend/prefire/models/catboost_s_tier_model.pkl`
2. System uses fallback rule-based calculation

#### 6. Port Already in Use

**Problem**: `Port 8000 is already in use`

**Solution**:
```bash
# Find process using port
netstat -ano | findstr :8000
# Kill process
taskkill /PID <process_id> /F
```

### Logs Location

- **Backend**: Console output (stdout/stderr)
- **Frontend**: Browser console (F12)

### Development Mode

For development with auto-reload:

```bash
# Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run dev
```

---

## 📦 Model Features

The CatBoost model uses 81 environmental features for risk prediction:

### Vegetation Indices
- NDVI, GNDVI, SAVI, EVI, NBR, NDWI, NDSI (Sentinel-2 & Landsat)

### Thermal Data
- Land Surface Temperature (LST) Day/Night
- Skin Temperature

### Weather Variables
- Temperature (2m, min, max)
- Relative Humidity
- Dewpoint
- Wind Speed & Direction (u/v components)
- Vapor Pressure Deficit (VPD)
- Precipitation

### Soil Conditions
- Soil Temperature (0-7cm)
- Soil Moisture (0-7cm)

### Topography
- Elevation (min, max, stddev, range)
- Slope (min, max, stddev)
- Aspect (mean, stddev)
- MTPI (Terrain Position Index)

### Temporal Features
- Lag variables (1, 3, 7, 14, 30 days)
- Rolling averages (7, 14, 30 days)
- Vegetation indices history

### Drought & Fire Energy
- Palmer Drought Severity Index (PDSI)
- Fire Energy Release (MODIS)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing-feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) - Fire satellite data
- [Open-Meteo](https://open-meteo.com/) - Weather API
- [Google Earth Engine](https://earthengine.google.com/) - Environmental data
- [Leaflet](https://leafletjs.com/) - Open-source maps
- [CARTO](https://carto.com/) - Map tiles
- [Esri](https://www.esri.com/) - Satellite imagery
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [CatBoost](https://catboost.ai/) - ML framework

# 🚀 FIRE ALERT - Quick Start Guide

A comprehensive wildfire detection and risk assessment platform. This guide will help you get up and running in minutes.

---

## ⚡ Quick Start (Windows - Recommended)

### Option 1: One-Click Startup

Simply **double-click** `run_app.bat` in the project root directory.

The script will automatically:
1. Detect the virtual environment (venv_py311)
2. Start the backend server on port 8000
3. Start the frontend server on port 5173
4. Open your default browser to the application

### Option 2: Command Line

```cmd
run_app.bat
```

---

## 🔧 Manual Setup

If you prefer manual control or are on a different operating system:

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend runtime |
| npm | 9+ | Package manager |

### Step 1: Clone/Download the Project

Extract the project to your desired location:

```bash
cd "New folder/v4 cleanup"
```

### Step 2: Backend Setup

#### Using Provided Virtual Environment (Recommended)

The project includes a pre-configured virtual environment at `../venv_py311` (one level up from project root).

```cmd
cd backend

# Run using the provided venv
..\..\venv_py311\Scripts\python.exe main.py
```

Or use absolute path:

```cmd
"F:\v4 cleanup\venv_py311\Scripts\python.exe" main.py
```

#### Creating Your Own Virtual Environment (Alternative)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
```

#### Activate Virtual Environment

**Windows:**
```cmd
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required for fire detection
NASA_FIRMS_API_KEY=your_nasa_firms_api_key_here

# Optional: Google Earth Engine
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT=your-account@project.iam.gserviceaccount.com
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH=path/to/key.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

> **Note**: Get your free NASA FIRMS API key at https://firms.modaps.eosdis.nasa.gov/api/area/token

#### Start Backend Server

```bash
python main.py
```

The backend will start at `http://localhost:8000`

### Step 3: Frontend Setup

#### Open New Terminal

```bash
cd frontend
```

#### Install Dependencies

```bash
npm install
```

#### Start Development Server

```bash
npm run dev
```

The frontend will start at `http://localhost:5173`

---

## 🌐 Access Points

After starting both servers, access the application:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | Main application UI |
| **Backend API** | http://localhost:8000 | REST API |
| **Swagger Docs** | http://localhost:8000/docs | Interactive API documentation |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |

---

## 🎯 Quick Usage Guide

### Fire Detection Mode

1. Select a region from the dropdown (e.g., Nepal, California, Australia)
2. Choose a time frame (24h, 48h, 72h, 7 days)
3. Click "Detect Fires" button
4. View active fire markers on the map

### Risk Assessment Mode

1. Click the "Risk" toggle in the control panel
2. Either:
   - Select a predefined location from the dropdown
   - Click anywhere directly on the map
3. View the AI-powered risk assessment with:
   - Risk score (0-100)
   - Risk level (Low/Medium/High/Critical)
   - Factor breakdown (Weather, Vegetation, Topography, Historical)

### Map Controls

- **Map Style**: Click the map style button (top-right) to switch between Dark/Satellite/Light modes
- **Zoom**: Mouse wheel or pinch to zoom
- **Pan**: Click and drag to move around

---

## 🔍 Verifying Installation

### Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "modules": {
    "prefire_analyzer": true,
    "fire_detector": true,
    "weather_fetcher": true
  }
}
```

### Check API Root

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "message": "Wildfire Detection API v2.0",
  "version": "2.0.0",
  "modules_loaded": true
}
```

---

## 🐛 Troubleshooting

### Port Already in Use

If you see "Port 8000 is already in use":

```bash
# Windows: Find and kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Module Not Found Errors

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### NASA FIRMS Not Working

1. Check API key is in `.env` file
2. Verify API key at https://firms.modaps.eosdis.nasa.gov/api/area/token
3. System will use mock data if API is unavailable

### Frontend Not Loading

1. Check backend is running
2. Clear browser cache
3. Check browser console for errors

---

## 📱 Features Overview

| Feature | Description |
|---------|-------------|
| 🔥 Fire Detection | Real-time satellite fire data from NASA FIRMS |
| ⚠️ Risk Assessment | AI-powered risk prediction using CatBoost |
| 🗺️ Interactive Map | Multiple styles with click-to-assess |
| 🌤️ Weather Data | Real-time weather from Open-Meteo |
| 🌍 Global Coverage | Nepal, Australia, California, India, Indonesia |
| 🎨 Modern UI | Glass-morphism design with animations |

---

## 📞 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review API endpoints at http://localhost:8000/docs
- Check troubleshooting section in README.md

---

**Happy Fire Monitoring! 🔥**

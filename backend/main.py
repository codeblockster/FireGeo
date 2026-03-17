"""
FastAPI Backend for Wildfire Detection & Risk Assessment
Provides REST endpoints for fire detection, environmental data, and risk assessment
Now connected to real ML models and APIs!
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import real modules
try:
    from backend.prefire.pre_fire_analyzer import PreFireAnalyzer
    from backend.firedetect.fire_detector import FireDetector
    from backend.src.data_collection.weather_api import WeatherDataFetcher
    from backend.postfire.models.active_fire_spread import ActiveFireCA
    from backend.config import ACTIVE_FIRE_CA_MODEL
    MODULES_AVAILABLE = True
    logger.info("✅ Successfully imported PreFireAnalyzer, FireDetector, WeatherDataFetcher, and ActiveFireCA")
except ImportError as e:
    logger.warning(f"⚠️ Could not import ML modules: {e}")
    MODULES_AVAILABLE = False

app = FastAPI(
    title="Wildfire Detection API",
    description="Wildfire Detection & Risk Assessment Backend - Connected to AI Models",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:9000", "http://127.0.0.1:5173", "http://127.0.0.1:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Singleton Module Instances
# ========================

_fire_detector: Optional[FireDetector] = None
_prefire_analyzer: Optional[PreFireAnalyzer] = None
_weather_fetcher: Optional[WeatherDataFetcher] = None
_active_fire_ca: Optional[ActiveFireCA] = None

def get_active_fire_ca() -> ActiveFireCA:
    """Get or create ActiveFireCA singleton"""
    global _active_fire_ca
    if _active_fire_ca is None:
        if MODULES_AVAILABLE:
            try:
                _active_fire_ca = ActiveFireCA(str(ACTIVE_FIRE_CA_MODEL))
                logger.info("✅ ActiveFireCA initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ActiveFireCA: {e}")
                raise
        else:
            raise RuntimeError("ActiveFireCA module not available")
    return _active_fire_ca

def get_fire_detector() -> FireDetector:
    """Get or create FireDetector singleton"""
    global _fire_detector
    if _fire_detector is None:
        if MODULES_AVAILABLE:
            try:
                _fire_detector = FireDetector()
                logger.info("✅ FireDetector initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize FireDetector: {e}")
                raise
        else:
            raise RuntimeError("FireDetector module not available")
    return _fire_detector

def get_prefire_analyzer() -> PreFireAnalyzer:
    """Get or create PreFireAnalyzer singleton"""
    global _prefire_analyzer
    if _prefire_analyzer is None:
        if MODULES_AVAILABLE:
            try:
                _prefire_analyzer = PreFireAnalyzer()
                logger.info("✅ PreFireAnalyzer initialized")
                if _prefire_analyzer.model_loaded:
                    logger.info("✅ CatBoost model loaded successfully")
                else:
                    logger.warning("⚠️ CatBoost model not loaded - will use fallback")
            except Exception as e:
                logger.error(f"❌ Failed to initialize PreFireAnalyzer: {e}")
                raise
        else:
            raise RuntimeError("PreFireAnalyzer module not available")
    return _prefire_analyzer

def get_weather_fetcher() -> WeatherDataFetcher:
    """Get or create WeatherDataFetcher singleton"""
    global _weather_fetcher
    if _weather_fetcher is None:
        if MODULES_AVAILABLE:
            try:
                _weather_fetcher = WeatherDataFetcher()
                logger.info("✅ WeatherDataFetcher initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize WeatherDataFetcher: {e}")
                raise
        else:
            raise RuntimeError("WeatherDataFetcher module not available")
    return _weather_fetcher


# ========================
# Request/Response Models
# ========================

class Location(BaseModel):
    id: str
    name: str
    lat: float
    lng: float

class LocationRequest(BaseModel):
    location: Optional[Location] = None
    hours: Optional[int] = 24  # Time frame in hours, default 24

class FireLocation(BaseModel):
    id: str
    lat: float
    lng: float
    intensity: int
    confidence: int
    timestamp: str
    brightness: Optional[float] = None
    frp: Optional[float] = None
    satellite: Optional[str] = None
    acq_datetime: Optional[str] = None

class DetectFiresResponse(BaseModel):
    fires: List[FireLocation]
    count: int
    source: str
    timestamp: str

class EnvironmentalData(BaseModel):
    temperature: float
    humidity: int
    windSpeed: int
    windDirection: int
    vegetationIndex: float
    droughtIndex: float
    dewpoint: Optional[float] = None
    cloudCover: Optional[float] = None
    pressure: Optional[float] = None
    precipitation: Optional[float] = None

class EnvDataResponse(BaseModel):
    data: EnvironmentalData
    timestamp: str

class RiskFactors(BaseModel):
    weather: int
    vegetation: int
    topography: int
    historical: int

class RiskAssessment(BaseModel):
    level: str
    score: float
    probability: float
    alert_priority: str
    confidence: str
    factors: RiskFactors

class AssessRiskRequest(BaseModel):
    location: Location
    envData: Optional[EnvironmentalData] = None

class AssessRiskResponse(BaseModel):
    risk: RiskAssessment
    location: Location
    features: Optional[Dict[str, Any]] = None
    timestamp: str


# ========================
# API Endpoints
# ========================

@app.get("/")
async def root():
    return {
        "message": "Wildfire Detection API v2.0",
        "version": "2.0.0",
        "modules_loaded": MODULES_AVAILABLE,
        "endpoints": {
            "detect-fires": "POST /api/detect-fires - Detect fires using NASA FIRMS",
            "env-data": "POST /api/env-data - Get environmental/weather data",
            "assess-risk": "POST /api/assess-risk - AI-powered risk assessment",
            "weather": "GET /api/weather - Standalone weather data",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "prefire_analyzer": False,
            "fire_detector": False,
            "weather_fetcher": False
        }
    }
    
    if MODULES_AVAILABLE:
        try:
            # Test module initialization
            get_prefire_analyzer()
            status["modules"]["prefire_analyzer"] = True
        except:
            pass
        
        try:
            get_fire_detector()
            status["modules"]["fire_detector"] = True
        except:
            pass
        
        try:
            get_weather_fetcher()
            status["modules"]["weather_fetcher"] = True
        except:
            pass
    
    return status


@app.post("/api/detect-fires", response_model=DetectFiresResponse)
async def detect_fires(request: LocationRequest):
    """
    Detect active fires using NASA FIRMS data
    """
    if not request.location:
        raise HTTPException(status_code=400, detail="Location is required")
    
    location = request.location
    hours = request.hours or 24  # Default to 24 hours if not provided
    
    try:
        # Use real FireDetector
        detector = get_fire_detector()
        
        # Map location name to region
        region = _map_location_to_region(location.name, location.lat, location.lng)
        
        # Detect fires with specified time frame
        result = detector.detect_fires(region, hours=hours)
        
        # Convert to response format
        fires = []
        for fire in result.get('fires', []):
            fires.append(FireLocation(
                id=f"fire-{fire.get('latitude', 0)}-{fire.get('longitude', 0)}",
                lat=fire.get('latitude', 0),
                lng=fire.get('longitude', 0),
                intensity=int(fire.get('brightness', 50) / 4),  # Scale to 0-100
                confidence=int(fire.get('confidence', 50)),
                timestamp=fire.get('acq_date', datetime.now().isoformat()),
                brightness=fire.get('brightness'),
                frp=fire.get('frp'),
                satellite=fire.get('satellite'),
                acq_datetime=fire.get('acq_datetime')
            ))
        
        return DetectFiresResponse(
            fires=fires,
            count=len(fires),
            source=result.get('source', 'NASA FIRMS'),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error detecting fires: {e}")
        # Fallback to mock data if real detection fails
        return _generate_mock_fires(location)


@app.post("/api/env-data", response_model=EnvDataResponse)
async def get_env_data(request: LocationRequest):
    """
    Get environmental/weather data using Open-Meteo API and GEE
    """
    if not request.location:
        raise HTTPException(status_code=400, detail="Location is required")
    
    location = request.location
    
    try:
        # Use real WeatherDataFetcher
        weather_api = get_weather_fetcher()
        
        # Fetch current weather
        current = weather_api.fetch_current_weather(location.lat, location.lng)
        
        # Get vegetation from GEE (CatBoost features)
        try:
            analyzer = get_prefire_analyzer()
            features = analyzer.feature_engineer.get_all_features(location.lat, location.lng)
            if features:
                # Use actual NDVI from satellite (Landsat or Sentinel-2)
                vegetation_index = features.get('landsat_ndvi', features.get('s2_ndvi', None))
                # Ensure we have a valid vegetation value (0-1 range), not 0
                if vegetation_index is None or vegetation_index <= 0:
                    vegetation_index = 0.5  # Default moderate vegetation
            else:
                vegetation_index = 0.5
        except Exception as e:
            logger.warning(f"Could not get GEE vegetation: {e}")
            # Fallback to historical calculation
            hist = weather_api.get_historical_weather(location.lat, location.lng, days_back=30)
            vegetation_index = _calculate_vegetation_index(hist)
        
        # Get historical data for drought index
        hist = weather_api.get_historical_weather(location.lat, location.lng, days_back=30)
        drought_index = _calculate_drought_index(hist)
        
        env_data = EnvironmentalData(
            temperature=current.get('temp', 20.0),
            humidity=int(current.get('humidity', 50)),
            windSpeed=int(current.get('wind_speed', 10)),
            windDirection=int(current.get('wind_direction', 180)),
            vegetationIndex=vegetation_index,
            droughtIndex=drought_index,
            dewpoint=current.get('dewpoint'),
            cloudCover=current.get('cloud_cover'),
            pressure=current.get('pressure'),
            precipitation=current.get('precipitation')
        )
        
        return EnvDataResponse(
            data=env_data,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching env data: {e}")
        # Fallback to mock data
        return _generate_mock_env_data(location)


@app.get("/api/weather")
async def get_weather(lat: float, lon: float):
    """
    Standalone weather endpoint
    """
    try:
        weather_api = get_weather_fetcher()
        current = weather_api.fetch_current_weather(lat, lon)
        return {
            "data": current,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assess-risk", response_model=AssessRiskResponse)
async def assess_risk(request: AssessRiskRequest):
    """
    Assess fire risk using AI-powered PreFireAnalyzer (CatBoost model)
    """
    if not request.location:
        raise HTTPException(status_code=400, detail="Location is required")
    
    location = request.location
    
    try:
        # Use real PreFireAnalyzer
        analyzer = get_prefire_analyzer()
        
        # Get AI prediction
        result = analyzer.analyze_location(location.lat, location.lng)
        
        if "error" in result and not result.get("model_used"):
            # Model not available, use rule-based fallback
            risk_data = _calculate_risk_from_features(result.get("features", {}))
        else:
            # Use AI prediction from CatBoost model
            probability = result.get('probability', 0.5)
            
            # Derive factors from CatBoost model prediction and features
            # The model's probability is the primary indicator
            features = result.get('features', {})
            
            # Weather factor: derived from weather features
            weather_factor = min(100, int(
                (60 - features.get('relative_humidity_pct', 50)) * 1.5 +
                features.get('vapor_pressure_deficit_kpa', 1) * 15 +
                features.get('wind_speed_ms', 5) * 2 +
                (features.get('temperature_2m_celsius_roll7_mean', 20) - 15) * 3
            ))
            
            # Vegetation factor: from NDVI and vegetation indices
            vegetation_factor = int(
                (features.get('landsat_ndvi', 0.5) + features.get('s2_ndvi', 0.5)) / 2 * 100
            )
            
            # Topography factor: from elevation and slope
            topography_factor = int(
                min(100, features.get('elevation_range_m', 500) / 10 + features.get('slope_max_deg', 10) * 2)
            )
            
            # Historical factor: based on model probability + vegetation
            historical_factor = min(100, int(probability * 200 + vegetation_factor * 0.3))
            
            # Score calculation - more meaningful representation
            # Use combination of probability and key risk factors
            # Return as float with 4 decimal places
            score = min(100.0, float(
                probability * 100 * 10 +  # Scale up probability
                weather_factor * 0.2 +
                vegetation_factor * 0.3 +
                topography_factor * 0.1
            ))
            
            # Convert probability to risk level (more sensitive thresholds)
            if probability >= 0.5:
                level = "critical"
            elif probability >= 0.3:
                level = "high"
            elif probability >= 0.15:
                level = "medium"
            else:
                level = "low"
            
            risk_data = {
                "level": level,
                "score": score,
                "probability": probability,
                "alert_priority": result.get('alert_priority', 'Monitor'),
                "confidence": str(result.get('confidence', 'Medium')),
                "factors": {
                    "weather": weather_factor,
                    "vegetation": vegetation_factor,
                    "topography": topography_factor,
                    "historical": historical_factor
                },
                "features": features
            }
        
        risk = RiskAssessment(
            level=risk_data["level"],
            score=risk_data["score"],
            probability=risk_data.get("probability", risk_data["score"] / 100),
            alert_priority=risk_data.get("alert_priority", "Monitor"),
            confidence=risk_data.get("confidence", "Medium"),
            factors=RiskFactors(**risk_data["factors"])
        )
        
        return AssessRiskResponse(
            risk=risk,
            location=location,
            features=risk_data.get("features"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        # Fallback to simple calculation
        return _generate_mock_risk_assessment(location)


# ========================
# Helper Functions
# ========================

def _map_location_to_region(name: str, lat: float, lon: float) -> str:
    """Map location name to FireDetector region"""
    name_lower = name.lower()
    
    if 'world' in name_lower:
        return "Whole World"
    elif 'nepal' in name_lower or 'kathmandu' in name_lower or 'pokhara' in name_lower:
        return "Nepal"
    elif 'california' in name_lower:
        return "California"
    elif 'australia' in name_lower:
        return "Australia"
    elif 'indonesia' in name_lower:
        return "Indonesia"
    elif 'india' in name_lower:
        return "India"
    else:
        return "Nepal"  # Default to Nepal

def _calculate_vegetation_index(hist_data: Dict) -> float:
    """Calculate vegetation index from historical data"""
    if not hist_data or not hist_data.get('temp_mean'):
        return 0.5
    
    # Simple proxy: more rain = more vegetation
    precip = hist_data.get('precipitation_sum', [0] * 30)
    if isinstance(precip, list) and precip:
        total_precip = sum(precip[-14:])  # Last 2 weeks
        return min(1.0, total_precip / 100)
    return 0.5

def _calculate_drought_index(hist_data: Dict) -> float:
    """Calculate drought index from historical data"""
    if not hist_data or not hist_data.get('temp_mean'):
        return 0.5
    
    # Simple proxy: high temp + low rain = drought
    temps = hist_data.get('temp_mean', [25] * 30)
    precips = hist_data.get('precipitation_sum', [0] * 30)
    
    if isinstance(temps, list) and temps:
        avg_temp = sum(temps[-14:]) / 14
        total_precip = sum(precips[-14:]) if isinstance(precips, list) else 0
        
        drought = (avg_temp / 40) - (total_precip / 200)
        return max(0, min(1.0, drought))
    return 0.5

def _calculate_weather_factor(features: Dict) -> int:
    """Calculate weather risk factor from features"""
    temp = features.get('temperature_2m_celsius_roll7_mean', features.get('lst_day_c', 25))
    humidity = features.get('relative_humidity_pct', 50)
    wind = features.get('wind_speed_ms', 10)
    vpd = features.get('vapor_pressure_deficit_kpa', 1)
    precip = features.get('precipitation_mm_roll7_sum', 0)
    
    score = 0
    # High temperature risk
    if temp > 30: score += 25
    if temp > 35: score += 20
    if temp > 25: score += 10
    # Low humidity risk
    if humidity < 30: score += 30
    if humidity < 40: score += 20
    if humidity < 50: score += 10
    # High wind risk
    if wind > 15: score += 20
    if wind > 10: score += 10
    # High VPD risk
    if vpd > 2: score += 20
    if vpd > 1.5: score += 10
    # Low precipitation risk
    if precip < 5: score += 15
    if precip < 2: score += 10
    
    return min(100, score)

def _calculate_risk_from_features(features: Dict) -> Dict:
    """Calculate risk from features when model unavailable"""
    temp = features.get('lst_day_c', 25)
    humidity = features.get('relative_humidity_pct', 50)
    vpd = features.get('vapor_pressure_deficit_kpa', 1)
    precip = features.get('precipitation_mm_lag1', 0)
    vegetation = features.get('landsat_savi', 0.4)
    
    # Simple rule-based calculation
    probability = 0.3
    
    if temp > 35: probability += 0.2
    if humidity < 30: probability += 0.2
    if vpd > 2.5: probability += 0.15
    if precip < 1: probability += 0.1
    if vegetation < 0.3: probability += 0.1
    
    probability = min(0.95, probability)
    
    if probability >= 0.8:
        level = "critical"
    elif probability >= 0.6:
        level = "high"
    elif probability >= 0.4:
        level = "medium"
    else:
        level = "low"
    
    return {
        "level": level,
        "score": int(probability * 100),
        "probability": probability,
        "alert_priority": level.capitalize(),
        "confidence": "Rule-based",
        "factors": {
            "weather": _calculate_weather_factor(features),
            "vegetation": int(vegetation * 100),
            "topography": random.randint(40, 70),
            "historical": int(probability * 100)
        },
        "features": features
    }

def _generate_mock_fires(location: Location) -> DetectFiresResponse:
    """Generate mock fire data for fallback"""
    fires = []
    for i in range(5):
        lat_offset = random.uniform(-1.5, 1.5)
        lng_offset = random.uniform(-1.5, 1.5)
        fires.append(FireLocation(
            id=f"fire-{i+1}",
            lat=location.lat + lat_offset,
            lng=location.lng + lng_offset,
            intensity=random.randint(50, 100),
            confidence=random.randint(70, 98),
            timestamp=datetime.now().isoformat()
        ))
    
    return DetectFiresResponse(
        fires=fires,
        count=len(fires),
        source="Mock Data (API unavailable)",
        timestamp=datetime.now().isoformat()
    )

def _generate_mock_env_data(location: Location) -> EnvDataResponse:
    """Generate mock environmental data for fallback"""
    return EnvDataResponse(
        data=EnvironmentalData(
            temperature=round(20 + random.uniform(-5, 15), 1),
            humidity=random.randint(10, 60),
            windSpeed=random.randint(5, 40),
            windDirection=random.randint(0, 360),
            vegetationIndex=round(random.uniform(0.2, 0.8), 2),
            droughtIndex=round(random.uniform(0.3, 0.9), 2)
        ),
        timestamp=datetime.now().isoformat()
    )

def _generate_mock_risk_assessment(location: Location) -> AssessRiskResponse:
    """Generate mock risk assessment for fallback"""
    score = random.randint(20, 90)
    
    if score >= 80:
        level = "critical"
    elif score >= 60:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"
    
    risk = RiskAssessment(
        level=level,
        score=score,
        probability=score / 100,
        alert_priority=level.capitalize(),
        confidence="Mock",
        factors=RiskFactors(
            weather=random.randint(30, 80),
            vegetation=random.randint(30, 80),
            topography=random.randint(30, 80),
            historical=random.randint(30, 80)
        )
    )
    
    return AssessRiskResponse(
        risk=risk,
        location=location,
        features=None,
        timestamp=datetime.now().isoformat()
    )


# ========================
# Startup Event
# ========================

@app.on_event("startup")
async def startup_event():
    """Initialize modules on startup"""
    logger.info("🚀 Starting Wildfire Detection API v2.0")
    logger.info(f"📦 Modules available: {MODULES_AVAILABLE}")
    
    if MODULES_AVAILABLE:
        try:
            # Initialize all modules
            get_weather_fetcher()
            logger.info("✅ Weather fetcher ready")
        except Exception as e:
            logger.warning(f"⚠️ Weather fetcher initialization failed: {e}")
        
        try:
            get_fire_detector()
            logger.info("✅ Fire detector ready")
        except Exception as e:
            logger.warning(f"⚠️ Fire detector initialization failed: {e}")
        
        try:
            analyzer = get_prefire_analyzer()
            if analyzer.model_loaded:
                logger.info("✅ Pre-fire analyzer with CatBoost model ready")
            else:
                logger.warning("⚠️ Pre-fire analyzer running in fallback mode")
        except Exception as e:
            logger.warning(f"⚠️ Pre-fire analyzer initialization failed: {e}")
    
    logger.info("✅ API ready to accept requests")


# ============================================================
# POST-FIRE ENDPOINT - Active Fire Spread Prediction (Cellular Automata)
# ============================================================

class PostFireSpreadRequest(BaseModel):
    latitude: float
    longitude: float
    wind_direction: Optional[float] = 90.0
    wind_speed: Optional[float] = 15.0
    time_steps: Optional[int] = 5
    cell_size_deg: Optional[float] = 0.005

class SpreadPoint(BaseModel):
    latitude: float
    longitude: float
    probability: float
    time_step: int

class FireSpreadConditions(BaseModel):
    ndvi: Optional[float] = None
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    data_source: str

class FireSpreadResponse(BaseModel):
    ignition_point: Dict[str, float]
    spread_radius_km: float
    spread_probability: float
    spread_points: List[SpreadPoint]
    conditions: FireSpreadConditions
    wind_direction: float
    wind_speed: float
    time_steps_simulated: int
    model_info: Dict[str, str]
    timestamp: str

@app.post("/api/post-fire-spread", response_model=FireSpreadResponse)
async def post_fire_spread(request: PostFireSpreadRequest):
    """
    Calculate fire spread using ActiveFireCA Module (Cellular Automata + RF Risk Model)
    
    Uses the pre-trained Random Forest ensemble model to predict fire spread
    based on environmental features fetched from Google Earth Engine.
    
    Parameters:
    - latitude: Ignition point latitude
    - longitude: Ignition point longitude  
    - wind_direction: Wind direction in degrees (0-360)
    - wind_speed: Wind speed in km/h
    - time_steps: Number of time steps to simulate (default: 5)
    - cell_size_deg: Grid cell size in degrees (default: 0.01 ~1.1km)
    
    Returns:
    - Spread points with probability and time step information
    """
    try:
        fire_lat = request.latitude
        fire_lon = request.longitude
        
        if not fire_lat or not fire_lon:
            raise HTTPException(status_code=400, detail="Latitude and Longitude required")
        
        # Validate coordinates
        if not (-90 <= fire_lat <= 90) or not (-180 <= fire_lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid coordinates: lat must be -90 to 90, lon must be -180 to 180")
        
        wind_direction = request.wind_direction or 90.0
        wind_speed = request.wind_speed or 15.0
        steps = request.time_steps or 5
        cell_size = request.cell_size_deg or 0.01
        
        # Initialize ActiveFireCA
        try:
            ca = get_active_fire_ca()
        except Exception as e:
            logger.error(f"Failed to initialize ActiveFireCA: {e}")
            raise HTTPException(status_code=503, detail=f"ActiveFireCA module unavailable: {str(e)}")
        
        # Get environmental conditions at ignition point
        conditions = {
            "ndvi": None,
            "temperature_celsius": None,
            "humidity_percent": None,
            "wind_direction_deg": wind_direction,
            "wind_speed_ms": wind_speed / 3.6,  # Convert km/h to m/s
            "data_source": "ActiveFireCA (GEE+RF Ensemble)"
        }
        
        try:
            # Fetch features at ignition point for environmental conditions
            date_str = datetime.now().strftime('%Y-%m-%d')
            features = ca.get_cell_features(fire_lat, fire_lon, date_str)
            if features:
                conditions["ndvi"] = features.get('landsat_ndvi') or features.get('s2_ndvi')
                conditions["temperature_celsius"] = features.get('temperature_2m_celsius')
                conditions["humidity_percent"] = features.get('relative_humidity_pct')
        except Exception as e:
            logger.warning(f"Could not fetch detailed conditions: {e}")
        
        logger.info(f"Starting CA Simulation for {fire_lat}, {fire_lon} with T={steps}")
        
        # Run the cellular automata simulation
        try:
            grid_state = ca.simulate_spread(
                start_lat=fire_lat, 
                start_lon=fire_lon, 
                steps=steps, 
                cell_size_deg=cell_size,
                override_wind_dir=wind_direction,
                override_wind_speed=wind_speed / 3.6  # Convert to m/s
            )
        except Exception as e:
            logger.error(f"CA Simulation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Fire spread simulation failed: {str(e)}")
        
        # Translate Grid to SpreadPoints for React frontend
        spread_points = []
        center = steps
        
        for y in range(grid_state.shape[0]):
            for x in range(grid_state.shape[1]):
                t_val = grid_state[y, x]
                if t_val > 0:
                    # Map time-step to decreasing probability (earlier = higher prob)
                    # 1 -> 95%, 2 -> 75%, 3 -> 55%, etc.
                    prob = max(15, 95 - (t_val - 1) * 20)
                    
                    cell_lat = fire_lat + (center - y) * cell_size
                    cell_lon = fire_lon + (x - center) * cell_size
                    
                    spread_points.append(SpreadPoint(
                        latitude=round(cell_lat, 5),
                        longitude=round(cell_lon, 5),
                        probability=int(prob),
                        time_step=int(t_val)
                    ))
        
        # Sort by probability (highest first)
        spread_points.sort(key=lambda p: p.probability, reverse=True)
        
        logger.info(f"CA Simulation successfully mapped {len(spread_points)} burned cells.")
        
        # Calculate spread metrics
        max_time_step = max([p.time_step for p in spread_points]) if spread_points else 0
        avg_probability = sum([p.probability for p in spread_points]) / len(spread_points) if spread_points else 0
        
        return FireSpreadResponse(
            ignition_point={
                "latitude": fire_lat,
                "longitude": fire_lon
            },
            spread_radius_km=round((max_time_step * cell_size) * 111.32, 2),  # Approx km
            spread_probability=round(avg_probability, 2),
            spread_points=spread_points,
            conditions=FireSpreadConditions(**conditions),
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            time_steps_simulated=steps,
            model_info={
                "model_type": "ActiveFireCA (Cellular Automata + RF/ET Ensemble)",
                "model_path": str(ACTIVE_FIRE_CA_MODEL),
                "features": "81 environmental features from GEE",
                "spread_logic": "Moore Neighborhood (8-direction)"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Post-fire CA calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

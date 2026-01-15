from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_collection.gee_extractor import get_gee_extractor
from src.data_collection.nasa_firms import get_nasa_firms_api
from src.analysis.post_fire import PostFireAnalyzer
from src.analysis.pre_fire import PreFireAnalyzer
from src.analysis.spread_prediction import SpreadPredictor

router = APIRouter()

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    date: str
    fire_detected: bool = False

@router.post("/predict")
def predict_risk(request: PredictionRequest):
    """
    Predict wildfire risk or spread based on location and conditions.
    Uses all 11 environmental features from GEE.
    """
    try:
        # Get environmental data
        gee = get_gee_extractor()
        env_data = gee.get_environmental_data(
            request.latitude, 
            request.longitude, 
            request.date
        )
        
        # Simple risk calculation based on environmental factors
        risk_score = 0
        
        # High temperature increases risk
        if env_data['temp_max'] > 35:
            risk_score += 2
        elif env_data['temp_max'] > 30:
            risk_score += 1
        
        # Low humidity increases risk
        if env_data['humidity'] < 30:
            risk_score += 2
        elif env_data['humidity'] < 50:
            risk_score += 1
        
        # High wind speed increases risk
        if env_data['wind_speed'] > 25:
            risk_score += 2
        elif env_data['wind_speed'] > 15:
            risk_score += 1
        
        # Low vegetation (dry) increases risk
        if env_data['vegetation'] < 0.3:
            risk_score += 2
        elif env_data['vegetation'] < 0.5:
            risk_score += 1
        
        # Low precipitation increases risk
        if env_data['precipitation'] < 5:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk": risk_level,
            "risk_score": risk_score,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "prediction_date": request.date,
            "environmental_data": env_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/environmental")
def get_environmental_data(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD)")
):
    """
    Get all 11 environmental features for a location from Google Earth Engine.
    
    Features: drought, elevation, energy_release, humidity, temp_min, temp_max,
    population, precipitation, vegetation, wind_direction, wind_speed
    """
    try:
        gee = get_gee_extractor()
        data = gee.get_environmental_data(lat, lon, date)
        
        return {
            "latitude": lat,
            "longitude": lon,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "features": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching environmental data: {str(e)}")

@router.post("/pre-fire/risk-map")
def get_pre_fire_risk_map(request: PredictionRequest):
    """
    Generate a Pre-Fire Risk Map for the area.
    Classifies fire risk zones based on environmental parameters using XGBoost.
    """
    try:
        analyzer = PreFireAnalyzer()
        # Default to 20km box, resolution 5x5
        result = analyzer.generate_risk_map(
            request.latitude, 
            request.longitude, 
            size_km=20, 
            grid_resolution=10
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk mapping error: {str(e)}")

@router.post("/post-fire/assessment")
def get_post_fire_assessment(request: PredictionRequest):
    """
    Analyze Post-Fire Burn Severity using dNBR.
    Returns map tiles and burn statistics.
    """
    try:
        analyzer = PostFireAnalyzer()
        # Use request date as 'post' date, default 'pre' logic inside analyzer
        result = analyzer.analyze_burn_severity(
            request.latitude, 
            request.longitude,
            post_date=request.date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-fire assessment error: {str(e)}")

@router.post("/spread/next-day")
def predict_fire_spread(request: PredictionRequest):
    """
    Predict wildfire spread for the next 24 hours using U-Net.
    """
    try:
        predictor = SpreadPredictor()
        result = predictor.predict_spread(request.latitude, request.longitude, request.date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spread prediction error: {str(e)}")

@router.get("/fires/active")
def get_active_fires(
    region: Optional[str] = Query(None, description="Country code (e.g., NPL for Nepal)"),
    min_lon: Optional[float] = Query(None, description="Minimum longitude"),
    min_lat: Optional[float] = Query(None, description="Minimum latitude"),
    max_lon: Optional[float] = Query(None, description="Maximum longitude"),
    max_lat: Optional[float] = Query(None, description="Maximum latitude"),
    hours: int = Query(24, description="Hours to look back"),
    source: str = Query('VIIRS_SNPP_NRT', description="Data source (VIIRS_SNPP_NRT, MODIS_NRT, etc.)")
):
    """
    Get active fires from NASA FIRMS.
    
    Provide either:
    - region: Country code (e.g., 'NPL', 'USA')
    - bbox: min_lon, min_lat, max_lon, max_lat
    """
    try:
        firms = get_nasa_firms_api()
        
        # Use bbox if all coordinates provided
        if all([min_lon, min_lat, max_lon, max_lat]):
            fires = firms.get_active_fires(bbox=(min_lon, min_lat, max_lon, max_lat), hours=hours, source=source)
        elif region:
            fires = firms.get_active_fires(region=region, hours=hours, source=source)
        else:
            # Default to Nepal region
            fires = firms.get_active_fires(bbox=(80.0, 26.0, 88.0, 31.0), hours=hours, source=source)
        
        return {
            "count": len(fires),
            "fires": fires,
            "source": "NASA FIRMS",
            "hours": hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching fire data: {str(e)}")

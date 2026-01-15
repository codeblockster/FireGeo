"""
Google Earth Engine Data Extractor
Fetches 11 environmental features from multiple satellite sources:
- MODIS, SRTM, GRIDMET, VIIRS, GPWv4
"""

import ee
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GEEExtractor:
    """Extract environmental data from Google Earth Engine"""

    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode"""
        return getattr(self, 'mock_mode', False)

    def __init__(self):
        """Initialize Google Earth Engine"""
        try:
            # Initialize with service account
            service_account = os.getenv('GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT')
            key_path = os.getenv('GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH')
            
            if service_account and key_path:
                if not os.path.exists(key_path):
                    logger.warning(f"GEE private key not found at: {key_path}")
                    raise FileNotFoundError(f"Key file missing: {key_path}")
                    
                credentials = ee.ServiceAccountCredentials(service_account, key_path)
                ee.Initialize(credentials)
                logger.info("GEE initialized successfully with service account")
                self.mock_mode = False
            else:
                # Fallback to default authentication (local user)
                logger.info("No service account credentials found, attempting default local auth...")
                ee.Initialize()
                logger.info("GEE initialized successfully with default credentials")
                self.mock_mode = False
                
        except Exception as e:
            logger.warning(f"GEE initialization failed: {e}")
            logger.warning("Falling back to MOCK DATA mode. Real satellite data will not be available.")
            self.mock_mode = True

    
    def get_environmental_data(self, lat: float, lon: float, date: str = None) -> Dict:
        """
        Fetch all 11 environmental features for a location
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with 11 environmental features
        """
        if self.mock_mode:
            return self._get_mock_data(lat, lon)
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        point = ee.Geometry.Point([lon, lat])
        
        try:
            data = {
                'drought': self.get_gridmet_drought(point, date),
                'elevation': self.get_srtm_elevation(point),
                'energy_release': self.get_modis_fire_energy(point, date),
                'humidity': self.get_gridmet_humidity(point, date),
                'temp_min': self.get_gridmet_temp_min(point, date),
                'temp_max': self.get_gridmet_temp_max(point, date),
                'population': self.get_gpw_population(point),
                'precipitation': self.get_gridmet_precipitation(point, date),
                'vegetation': self.get_viirs_ndvi(point, date),
                'wind_direction': self.get_gridmet_wind_direction(point, date),
                'wind_speed': self.get_gridmet_wind_speed(point, date)
            }
            logger.info(f"Successfully fetched environmental data for ({lat}, {lon})")
            return data
        except Exception as e:
            logger.error(f"Error fetching GEE data: {e}")
            return self._get_mock_data(lat, lon)
    
    def get_gridmet_drought(self, point: ee.Geometry.Point, date: str) -> float:
        """Get drought index from GRIDMET"""
        try:
            # GRIDMET drought dataset
            dataset = ee.ImageCollection('GRIDMET/DROUGHT') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            # PDSI (Palmer Drought Severity Index)
            pdsi = dataset.select('pdsi')
            value = pdsi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('pdsi').getInfo()
            
            return float(value) if value is not None else 0.0
        except Exception as e:
            logger.warning(f"GRIDMET drought error: {e}")
            return 0.0
    
    def get_srtm_elevation(self, point: ee.Geometry.Point) -> float:
        """Get elevation from SRTM"""
        try:
            srtm = ee.Image('USGS/SRTMGL1_003')
            elevation = srtm.select('elevation')
            
            value = elevation.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('elevation').getInfo()
            
            return float(value) if value is not None else 0.0
        except Exception as e:
            logger.warning(f"SRTM elevation error: {e}")
            return 500.0
    
    def get_modis_fire_energy(self, point: ee.Geometry.Point, date: str) -> float:
        """Get fire radiative power from MODIS"""
        try:
            # MODIS Thermal Anomalies
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            
            dataset = ee.ImageCollection('MODIS/006/MOD14A1') \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            if dataset.size().getInfo() > 0:
                fire_mask = dataset.first().select('FireMask')
                value = fire_mask.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=1000
                ).get('FireMask').getInfo()
                
                return float(value) if value is not None else 0.0
            return 0.0
        except Exception as e:
            logger.warning(f"MODIS fire energy error: {e}")
            return 0.0
    
    def get_gridmet_humidity(self, point: ee.Geometry.Point, date: str) -> float:
        """Get relative humidity from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            rh = dataset.select('rmax')  # Maximum relative humidity
            value = rh.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('rmax').getInfo()
            
            return float(value) if value is not None else 50.0
        except Exception as e:
            logger.warning(f"GRIDMET humidity error: {e}")
            return 50.0
    
    def get_gridmet_temp_min(self, point: ee.Geometry.Point, date: str) -> float:
        """Get minimum temperature from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            tmmn = dataset.select('tmmn')  # Minimum temperature in Kelvin
            value = tmmn.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('tmmn').getInfo()
            
            # Convert Kelvin to Celsius
            return float(value - 273.15) if value is not None else 15.0
        except Exception as e:
            logger.warning(f"GRIDMET temp_min error: {e}")
            return 15.0
    
    def get_gridmet_temp_max(self, point: ee.Geometry.Point, date: str) -> float:
        """Get maximum temperature from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            tmmx = dataset.select('tmmx')  # Maximum temperature in Kelvin
            value = tmmx.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('tmmx').getInfo()
            
            # Convert Kelvin to Celsius
            return float(value - 273.15) if value is not None else 30.0
        except Exception as e:
            logger.warning(f"GRIDMET temp_max error: {e}")
            return 30.0
    
    def get_gpw_population(self, point: ee.Geometry.Point) -> float:
        """Get population density from GPWv4"""
        try:
            # GPW v4 Population Density
            dataset = ee.Image('CIESIN/GPWv411/GPW_Population_Density')
            pop_density = dataset.select('population_density')
            
            value = pop_density.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).get('population_density').getInfo()
            
            return float(value) if value is not None else 0.0
        except Exception as e:
            logger.warning(f"GPW population error: {e}")
            return 100.0
    
    def get_gridmet_precipitation(self, point: ee.Geometry.Point, date: str) -> float:
        """Get precipitation from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            pr = dataset.select('pr')  # Precipitation in mm
            value = pr.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('pr').getInfo()
            
            return float(value) if value is not None else 0.0
        except Exception as e:
            logger.warning(f"GRIDMET precipitation error: {e}")
            return 0.0
    
    def get_viirs_ndvi(self, point: ee.Geometry.Point, date: str) -> float:
        """Get NDVI from VIIRS"""
        try:
            # VIIRS Vegetation Indices
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=16)).strftime('%Y-%m-%d')
            
            dataset = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1') \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            if dataset.size().getInfo() > 0:
                ndvi = dataset.first().select('NDVI')
                value = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=500
                ).get('NDVI').getInfo()
                
                # NDVI is scaled by 10000
                return float(value / 10000.0) if value is not None else 0.5
            return 0.5
        except Exception as e:
            logger.warning(f"VIIRS NDVI error: {e}")
            return 0.5
    
    def get_gridmet_wind_direction(self, point: ee.Geometry.Point, date: str) -> float:
        """Get wind direction from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            th = dataset.select('th')  # Wind direction in degrees
            value = th.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('th').getInfo()
            
            return float(value) if value is not None else 180.0
        except Exception as e:
            logger.warning(f"GRIDMET wind direction error: {e}")
            return 180.0
    
    def get_gridmet_wind_speed(self, point: ee.Geometry.Point, date: str) -> float:
        """Get wind speed from GRIDMET"""
        try:
            dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
            
            vs = dataset.select('vs')  # Wind speed in m/s
            value = vs.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=4000
            ).get('vs').getInfo()
            
            # Convert m/s to km/h
            return float(value * 3.6) if value is not None else 15.0
        except Exception as e:
            logger.warning(f"GRIDMET wind speed error: {e}")
            return 15.0
    
    def _get_mock_data(self, lat: float, lon: float) -> Dict:
        """Return mock data when GEE is not available"""
        import random
        random.seed(int(lat * 1000 + lon * 1000))
        
        return {
            'drought': round(random.uniform(-4.0, 4.0), 2),
            'elevation': round(random.uniform(100, 3000), 1),
            'energy_release': round(random.uniform(0, 100), 2),
            'humidity': round(random.uniform(20, 80), 1),
            'temp_min': round(random.uniform(10, 25), 1),
            'temp_max': round(random.uniform(25, 45), 1),
            'population': round(random.uniform(0, 500), 1),
            'precipitation': round(random.uniform(0, 50), 2),
            'vegetation': round(random.uniform(0.2, 0.8), 3),
            'wind_direction': round(random.uniform(0, 360), 1),
            'wind_speed': round(random.uniform(5, 30), 1)
        }


# Singleton instance
_gee_extractor = None

def get_gee_extractor() -> GEEExtractor:
    """Get or create GEE extractor instance"""
    global _gee_extractor
    if _gee_extractor is None:
        _gee_extractor = GEEExtractor()
    return _gee_extractor

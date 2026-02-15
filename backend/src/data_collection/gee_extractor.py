"""
Google Earth Engine Data Extractor
Fetches 11 environmental features from multiple satellite sources:
- MODIS, SRTM, GRIDMET, VIIRS, GPWv4
"""

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    from unittest.mock import MagicMock
    ee = MagicMock()

import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GEEExtractor:
    """Extract environmental data from Google Earth Engine"""

    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode"""
        return getattr(self, 'mock_mode', False)

    def __init__(self):
        """Initialize Google Earth Engine"""
        if not EE_AVAILABLE:
            logger.warning("Earth Engine API not installed. Forcing Mock Mode.")
            self.mock_mode = True
            return

        try:
            # Get project ID from environment
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            
            # Try service account authentication first
            service_account = os.getenv('GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT')
            key_path = os.getenv('GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH')
            
            if service_account and key_path:
                # Service account authentication (for production/cloud deployment)
                if not os.path.exists(key_path):
                    logger.warning(f"GEE private key not found at: {key_path}")
                    raise FileNotFoundError(f"Key file missing: {key_path}")
                    
                credentials = ee.ServiceAccountCredentials(service_account, key_path)
                
                if project_id:
                    ee.Initialize(credentials, project=project_id)
                    logger.info(f"✓ GEE initialized with service account (project: {project_id})")
                else:
                    ee.Initialize(credentials)
                    logger.info("✓ GEE initialized with service account (no project)")
                
                self.mock_mode = False
            else:
                # User authentication (for local development)
                logger.info("No service account found, using user authentication...")
                
                if project_id:
                    logger.info(f"Initializing GEE with project: {project_id}")
                    ee.Initialize(project=project_id)
                    logger.info(f"✓ GEE initialized successfully with project: {project_id}")
                else:
                    # Try legacy default
                    logger.warning("GOOGLE_CLOUD_PROJECT not set in .env, trying legacy project...")
                    try:
                        ee.Initialize(project='ee-legacy')
                        logger.info("✓ GEE initialized with legacy project")
                    except:
                        ee.Initialize()
                        logger.info("✓ GEE initialized without project")
                
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
            
            dataset = ee.ImageCollection('NASA/VIIRS/002/VNP13A1') \
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
    
    def get_landsat_savi(self, point: ee.Geometry.Point, date: str) -> float:
        """
        Calculate Soil Adjusted Vegetation Index (SAVI) using Landsat 8/9
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L) with L = 0.5
        """
        try:
            # Try Landsat 9 first, then Landsat 8
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            
            l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            dataset = l9.merge(l8)
            
            if dataset.size().getInfo() > 0:
                image = dataset.first()
                
                # Scale factors for Collection 2
                # SR_B5 is NIR, SR_B4 is Red
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
                
                l_factor = 0.5
                savi = nir.subtract(red).divide(nir.add(red).add(l_factor)).multiply(1 + l_factor)
                
                # Reduce to get single value
                savi_value = savi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=30
                ).getInfo()
                
                # Extract the value (key is 'constant' from the expression)
                if savi_value and len(savi_value) > 0:
                    val = list(savi_value.values())[0]
                    if val is not None:
                        return float(val)

            return 0.3  # Default fallback
        except Exception as e:
            logger.warning(f"Landsat SAVI error: {e}")
            return 0.3

    def get_modis_lst(self, point: ee.Geometry.Point, date: str) -> float:
        """Get Land Surface Temperature from MODIS"""
        try:
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
            dataset = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate(start_date, date) \
                .filterBounds(point) \
                .select('LST_Day_1km')
            
            if dataset.size().getInfo() > 0:
                # Scale factor is 0.02, units are Kelvin
                lst_k = dataset.first().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=1000
                ).get('LST_Day_1km').getInfo()
                
                if lst_k:
                    return float(lst_k * 0.02 - 273.15)
            
            return 25.0
        except Exception as e:
            logger.warning(f"MODIS LST error: {e}")
            return 25.0

    def get_terrain_metrics(self, point: ee.Geometry.Point) -> Dict[str, float]:
        """Get slope, aspect, and MTPI metrics from SRTM"""
        try:
            srtm = ee.Image('USGS/SRTMGL1_003')
            elevation = srtm.select('elevation')
            slope = ee.Terrain.slope(elevation)
            aspect = ee.Terrain.aspect(elevation)
            
            # MTPI (Multi-scale Topographic Position Index) - Simplified approximation
            # TPI = Elevation - Mean Elevation(neighborhood)
            # We'll use a small kernel vs large kernel difference
            mean_small = elevation.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(300, 'meters'))
            mean_large = elevation.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(1000, 'meters'))
            mtpi = mean_small.subtract(mean_large)
            
            # Buffer for stats
            buffer = point.buffer(1000)
            
            # Reduce region to get stats
            slope_stats = slope.reduceRegion(
                reducer=ee.Reducer.max().combine(
                    ee.Reducer.min().combine(
                        ee.Reducer.stdDev(), sharedInputs=True
                    ), sharedInputs=True
                ),
                geometry=buffer,
                scale=90
            ).getInfo()
            
            aspect_stats = aspect.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=buffer,
                scale=90
            ).getInfo()
            
            mtpi_stats = mtpi.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.min().combine(
                        ee.Reducer.max().combine(
                            ee.Reducer.stdDev(), sharedInputs=True
                        ), sharedInputs=True
                    ), sharedInputs=True
                ),
                geometry=buffer,
                scale=90
            ).getInfo()
            
            elev_stats = elevation.reduceRegion(
                reducer=ee.Reducer.max().combine(
                    ee.Reducer.min().combine(
                        ee.Reducer.stdDev(), sharedInputs=True
                    ), sharedInputs=True
                ),
                geometry=buffer,
                scale=90
            ).getInfo()
            
            # Extract and clean
            return {
                'slope_max': float(slope_stats.get('slope_max', 15.0)),
                'slope_min': float(slope_stats.get('slope_min', 0.0)),
                'slope_stddev': float(slope_stats.get('slope_stdDev', 5.0)),
                'aspect_mean': float(aspect_stats.get('aspect_mean', 180.0)),
                'aspect_stddev': float(aspect_stats.get('aspect_stdDev', 45.0)),
                'mtpi_mean': float(mtpi_stats.get('elevation_mean', 0.0)),
                'mtpi_min': float(mtpi_stats.get('elevation_min', -10.0)),
                'mtpi_max': float(mtpi_stats.get('elevation_max', 10.0)),
                'mtpi_stddev': float(mtpi_stats.get('elevation_stdDev', 5.0)),
                'elevation_min': float(elev_stats.get('elevation_min', 0.0)),
                'elevation_range': float(elev_stats.get('elevation_max', 500.0) - elev_stats.get('elevation_min', 0.0)),
                'elevation_stddev': float(elev_stats.get('elevation_stdDev', 100.0))
            }
        except Exception as e:
            logger.warning(f"Terrain metrics error: {e}")
            return {
                'slope_max': 15.0,
                'slope_min': 0.0,
                'slope_stddev': 5.0,
                'aspect_mean': 180.0,
                'aspect_stddev': 45.0,
                'mtpi_mean': 0.0,
                'mtpi_min': -10.0,
                'mtpi_max': 10.0,
                'mtpi_stddev': 5.0,
                'elevation_min': 0.0,
                'elevation_range': 500.0,
                'elevation_stddev': 100.0
            }

    def get_landsat_indices(self, point: ee.Geometry.Point, date: str) -> Dict[str, float]:
        """
        Get Landsat 8/9 vegetation indices: NDVI, GNDVI, NBR
        """
        try:
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Merge Landsat 9 and 8
            l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                .filterDate(start_date, date) \
                .filterBounds(point)
            
            dataset = l9.merge(l8)
            
            if dataset.size().getInfo() > 0:
                image = dataset.first()
                
                # Scale factors for Collection 2
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
                green = image.select('SR_B3').multiply(0.0000275).add(-0.2)
                swir = image.select('SR_B7').multiply(0.0000275).add(-0.2)
                
                # NDVI = (NIR - Red) / (NIR + Red)
                ndvi = nir.subtract(red).divide(nir.add(red))
                
                # GNDVI = (NIR - Green) / (NIR + Green)
                gndvi = nir.subtract(green).divide(nir.add(green))
                
                # NBR = (NIR - SWIR) / (NIR + SWIR)
                nbr = nir.subtract(swir).divide(nir.add(swir))
                
                # Reduce to get values
                buffer = point.buffer(100)
                ndvi_val = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30).getInfo()
                gndvi_val = gndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30).getInfo()
                nbr_val = nbr.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30).getInfo()
                
                return {
                    'landsat_ndvi': float(list(ndvi_val.values())[0]) if ndvi_val and len(ndvi_val) > 0 else 0.5,
                    'landsat_gndvi': float(list(gndvi_val.values())[0]) if gndvi_val and len(gndvi_val) > 0 else 0.5,
                    'landsat_nbr': float(list(nbr_val.values())[0]) if nbr_val and len(nbr_val) > 0 else 0.3
                }
            
            return {'landsat_ndvi': 0.5, 'landsat_gndvi': 0.5, 'landsat_nbr': 0.3}
        except Exception as e:
            logger.warning(f"Landsat indices error: {e}")
            return {'landsat_ndvi': 0.5, 'landsat_gndvi': 0.5, 'landsat_nbr': 0.3}

    def get_sentinel2_indices(self, point: ee.Geometry.Point, date: str) -> Dict[str, float]:
        """
        Get Sentinel-2 vegetation indices: NDVI, GNDVI, NDSI, NDWI, SAVI, EVI, and cloud cover
        """
        try:
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Sentinel-2 Surface Reflectance
            dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start_date, date) \
                .filterBounds(point) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            
            if dataset.size().getInfo() > 0:
                image = dataset.first()
                
                # Get bands (already in reflectance 0-10000)
                nir = image.select('B8').divide(10000.0)
                red = image.select('B4').divide(10000.0)
                green = image.select('B3').divide(10000.0)
                blue = image.select('B2').divide(10000.0)
                swir = image.select('B11').divide(10000.0)
                
                # Calculate indices
                # NDVI = (NIR - Red) / (NIR + Red)
                ndvi = nir.subtract(red).divide(nir.add(red))
                
                # GNDVI = (NIR - Green) / (NIR + Green)
                gndvi = nir.subtract(green).divide(nir.add(green))
                
                # NDSI = (Green - SWIR) / (Green + SWIR)
                ndsi = green.subtract(swir).divide(green.add(swir))
                
                # NDWI = (Green - NIR) / (Green + NIR)
                ndwi = green.subtract(nir).divide(green.add(nir))
                
                # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L) with L = 0.5
                l_factor = 0.5
                savi = nir.subtract(red).divide(nir.add(red).add(l_factor)).multiply(1 + l_factor)
                
                # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
                evi = nir.subtract(red).divide(
                    nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1.0)
                ).multiply(2.5)
                
                # Cloud cover
                cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                
                # Reduce to get values
                buffer = point.buffer(100)
                ndvi_val = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                gndvi_val = gndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                ndsi_val = ndsi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                ndwi_val = ndwi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                savi_val = savi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                evi_val = evi.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10).getInfo()
                
                return {
                    's2_ndvi': float(list(ndvi_val.values())[0]) if ndvi_val and len(ndvi_val) > 0 else 0.5,
                    's2_gndvi': float(list(gndvi_val.values())[0]) if gndvi_val and len(gndvi_val) > 0 else 0.5,
                    's2_ndsi': float(list(ndsi_val.values())[0]) if ndsi_val and len(ndsi_val) > 0 else 0.0,
                    's2_ndwi': float(list(ndwi_val.values())[0]) if ndwi_val and len(ndwi_val) > 0 else 0.0,
                    's2_savi': float(list(savi_val.values())[0]) if savi_val and len(savi_val) > 0 else 0.3,
                    's2_evi': float(list(evi_val.values())[0]) if evi_val and len(evi_val) > 0 else 0.3,
                    's2_cloud_cover_percent': float(cloud_cover) if cloud_cover is not None else 20.0
                }
            
            return {
                's2_ndvi': 0.5, 's2_gndvi': 0.5, 's2_ndsi': 0.0,
                's2_ndwi': 0.0, 's2_savi': 0.3, 's2_evi': 0.3,
                's2_cloud_cover_percent': 20.0
            }
        except Exception as e:
            logger.warning(f"Sentinel-2 indices error: {e}")
            return {
                's2_ndvi': 0.5, 's2_gndvi': 0.5, 's2_ndsi': 0.0,
                's2_ndwi': 0.0, 's2_savi': 0.3, 's2_evi': 0.3,
                's2_cloud_cover_percent': 20.0
            }


    def get_firms_map_id(self, date: str = None) -> str:
        """
        Get Earth Engine Map ID for FIRMS dataset (for frontend display)
        
        Returns:
            Tile URL format string
        """
        if self.mock_mode:
            return None
            
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # FIRMS dataset
            dataset = ee.ImageCollection('FIRMS') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            # visualization - T21 is brightness temperature
            vis_params = {
                'palette': ['red', 'orange', 'yellow'],
                'min': 300.0,
                'max': 500.0,
                'bands': ['T21']
            }
            
            # Return tile URL
            map_id_dict = dataset.getMapId(vis_params)
            return map_id_dict['tile_fetcher'].url_format
        except Exception as e:
            logger.error(f"Error getting FIRMS Map ID: {e}")
            return None

    def get_active_fire_count(self, point: ee.Geometry.Point, radius_km: float = 50.0, date: str = None) -> int:
        """
        Count active fires within a radius of a point using FIRMS data.
        
        Args:
            point: Center point
            radius_km: Radius in kilometers
            date: Date string
            
        Returns:
            Number of active fires found
        """
        if self.mock_mode:
            return 0
            
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
                
            buffer = point.buffer(radius_km * 1000)
            
            # FIRMS dataset: 'T21' band usually indicates fire
            dataset = ee.ImageCollection('FIRMS') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .filterBounds(buffer)
                
            if dataset.size().getInfo() > 0:
                # Use T21 band for fire detection
                fires = dataset.select(['T21', 'confidence']).max()
                
                # Mask low confidence/low temp
                fire_mask = fires.select('T21').gt(325.0)
                
                stats = fire_mask.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=buffer,
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()
                
                return int(stats.get('T21', 0))
            return 0
            
        except Exception as e:
            logger.warning(f"Error counting active fires: {e}")
            return 0

    def get_active_fire_locations(self, region_geometry, date: str = None) -> list:
        """
        Get list of active fires as dictionaries (lat, lon, brightness, confidence).
        Args:
            region_geometry: ee.Geometry definition of the region
            date: Date string (YYYY-MM-DD)
        Returns:
            List of dicts
        """
        if self.mock_mode:
            return []
            
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # FIRMS dataset
            dataset = ee.ImageCollection('FIRMS') \
                .filterDate(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .filterBounds(region_geometry)
            
            if dataset.size().getInfo() > 0:
                # Get the max values for the day (mosaic)
                image = dataset.select(['T21', 'confidence']).max()
                
                # Create detailed fire points (vectors)
                # We threshold T21 > 325K to filter potential fires
                fire_zones = image.select('T21').gt(325.0)
                
                # Reduce to vectors to get points
                vectors = image.addBands(ee.Image.pixelLonLat()).updateMask(fire_zones).reduceToVectors(
                    geometry=region_geometry,
                    scale=1000,  # 1km resolution (MODIS)
                    geometryType='centroid',
                    eightConnected=False,
                    labelProperty='label',
                    reducer=ee.Reducer.mean()  # Get mean values for the vector
                )
                
                # Extract features
                features = vectors.getInfo()['features']
                results = []
                
                for f in features:
                    props = f['properties']
                    coords = f['geometry']['coordinates']
                    results.append({
                        'latitude': coords[1],
                        'longitude': coords[0],
                        'brightness': float(props.get('T21', 0)),
                        'confidence': float(props.get('confidence', 0)),
                        'acq_date': date,
                        'acq_time': '0000', # aggregated
                        'satellite': 'GEE_FIRMS',
                        'instrument': 'MODIS/VIIRS',
                        'frp': 0.0 # simplified
                    })
                
                return results
            return []
            
        except Exception as e:
            logger.error(f"Error getting fire locations from GEE: {e}")
            return []

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
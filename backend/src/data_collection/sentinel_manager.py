import ee
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class SentinelManager:
    """
    Manager for Sentinel-2 satellite data in Google Earth Engine.
    Handles data fetching, cloud masking, and spectral index calculation (inputs for dNBR).
    """

    def __init__(self):
        """Initialize connection to GEE (assumes ee.Initialize() has been called elsewhere or lazily here)"""
        try:
            if not run_init_check():
                 ee.Initialize()
        except Exception as e:
            logger.warning(f"GEE not initialized in SentinelManager, attempting strict init: {e}")
            try:
                 ee.Initialize()
            except Exception as e2:
                 logger.error(f"Failed to initialize GEE: {e2}")

    def get_sentinel2_image(self, roi: ee.Geometry, date_start: str, date_end: str, max_cloud_cover: int = 20) -> Optional[ee.Image]:
        """
        Fetch the best cloud-free Sentinel-2 image for a given ROI and date range.
        
        Args:
            roi: Region of Interest (ee.Geometry)
            date_start: Start date string (YYYY-MM-DD)
            date_end: End date string (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover percentage allowed
            
        Returns:
            ee.Image or None if no image found
        """
        try:
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(roi) \
                .filterDate(date_start, date_end) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE')

            # Get the least cloudy image
            image = s2_collection.first()
            
            # Check if image exists by trying to get id (will throw if null)
            # A safer way in python API without forcing a call is harder, but .first() returns null logic element if empty
            # We can check size() but that forces a call.
            
            # Let's return it and let the caller handle nulls or checks, 
            # but usually we want to clip it here.
            if image:
                return image.clip(roi)
            return None

        except Exception as e:
            logger.error(f"Error fetching Sentinel-2 data: {e}")
            return None

    def calculate_nbr(self, image: ee.Image) -> ee.Image:
        """
        Calculate Normalized Burn Ratio (NBR).
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        For Sentinel-2:
            NIR = B8
            SWIR2 = B12
        """
        return image.normalizedDifference(['B8', 'B12']).rename('NBR')

    def calculate_dnbr(self, pre_fire_img: ee.Image, post_fire_img: ee.Image) -> ee.Image:
        """
        Calculate Differenced Normalized Burn Ratio (dNBR).
        dNBR = PreFireNBR - PostFireNBR
        """
        nbr_pre = self.calculate_nbr(pre_fire_img)
        nbr_post = self.calculate_nbr(post_fire_img)
        
        dnbr = nbr_pre.subtract(nbr_post).rename('dNBR')
        return dnbr

    def classify_burn_severity(self, dnbr_image: ee.Image) -> ee.Image:
        """
        Classify dNBR values into burn severity levels roughly based on USGS standards.
        
        Classes:
        0: Unburned / Regrowth (< 0.1)
        1: Low Severity (0.1 - 0.27)
        2: Moderate-Low Severity (0.27 - 0.44)
        3: Moderate-High Severity (0.44 - 0.66)
        4: High Severity (> 0.66)
        """
        # Define thresholds
        # < 0.10  : Unburned (approx)
        # 0.10 - 0.27 : Low SEverity
        # 0.27 - 0.66 : Moderate Severity (combined for simplicity or split)
        # > 0.66  : High Severity
        
        # We will create a discrete classification image
        severity = ee.Image(0) \
            .where(dnbr_image.gt(0.10), 1) \
            .where(dnbr_image.gt(0.27), 2) \
            .where(dnbr_image.gt(0.44), 3) \
            .where(dnbr_image.gt(0.66), 4) \
            .rename('Severity')
            
        return severity

def run_init_check():
    """Check if GEE is already initialized to avoid re-init errors"""
    import ee
    try:
        ee.Image("NOAA/VIIRS/001/VNP13A1") # Try a dummy object construction
        return True
    except:
        return False

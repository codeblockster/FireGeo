"""
NASA FIRMS (Fire Information for Resource Management System) API Integration
Fetches real-time active fire data from MODIS and VIIRS satellites
"""

import requests
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

# Cache for 30 minutes (1800 seconds) to avoid hitting API limits
# Max 100 items in cache
api_cache = TTLCache(maxsize=100, ttl=1800)

# Load environment variables
from dotenv import load_dotenv
from backend.config import PROJECT_ROOT

# Explicitly load from project root to ensure it's found
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)

class NASAFirmsAPI:
    """NASA FIRMS API client for active fire detection"""
    
    @property
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode"""
        return getattr(self, 'mock_mode', False)

    def __init__(self, api_key: str = None):
        """
        Initialize NASA FIRMS API client
        
        Args:
            api_key: NASA FIRMS API key (or from environment)
        """
        self.api_key = api_key or os.getenv('NASA_FIRMS_API_KEY')
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        
        if not self.api_key or self.api_key == "your_nasa_firms_key_here":
            logger.warning("NASA FIRMS API key not found or is default. Using MOCK DATA.")
            print("\n" + "="*50)
            print(" WARNING: NASA FIRMS API KEY NOT FOUND")
            print(" SYSTEM IS RUNNING IN MOCK MODE (SIMULATED DATA)")
            print(" To use real data, set NASA_FIRMS_API_KEY in .env file")
            print("="*50 + "\n")
            logger.warning("Please set NASA_FIRMS_API_KEY in your .env file to get real fire data.")
            self.mock_mode = True
        else:
            logger.info(f"NASA FIRMS API key found: {self.api_key[:4]}...{self.api_key[-4:]}")
            print("\n" + "="*50)
            print(f" SUCCESS: NASA FIRMS API KEY FOUND ({self.api_key[:4]}...)")
            print(" SYSTEM IS RUNNING IN REAL API MODE")
            print("="*50 + "\n")
            self.mock_mode = False
    
    @cached(cache=api_cache)
    def get_active_fires(self, 
                        region: str = None,
                        bbox: tuple = None,
                        hours: int = 24,
                        source: str = 'VIIRS_SNPP_NRT') -> List[Dict]:
        """
        Get active fires from NASA FIRMS
        
        Args:
            region: Country code (e.g., 'NPL' for Nepal) or None
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) or None
            hours: Hours to look back (default 24)
            source: Data source - 'VIIRS_SNPP_NRT', 'MODIS_NRT', or 'VIIRS_NOAA20_NRT'
            
        Returns:
            List of fire dictionaries with lat, lon, confidence, brightness, etc.
        """
        if self.mock_mode:
            return self._get_mock_fires(region, bbox)
        
        try:
            # Convert hours to days (NASA FIRMS API expects DAY_RANGE 1-10)
            days = max(1, min(10, hours // 24))
            if hours < 24:
                logger.info(f"NASA FIRMS API expects days. Mapping {hours} hours to 1 day.")
                days = 1
            
            # Use bbox if provided, otherwise use region
            if bbox:
                min_lon, min_lat, max_lon, max_lat = bbox
                url = f"{self.base_url}/{self.api_key}/{source}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}"
            elif region:
                if region.lower() == 'world':
                    # Global query using the 'world' keyword in the area API
                    url = f"{self.base_url}/{self.api_key}/{source}/world/{days}"
                else:
                    # Country-based query
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{self.api_key}/{source}/{region}/{days}"
            else:
                logger.error("Either region or bbox must be provided")
                return []
            
            logger.info(f"Fetching fires from NASA FIRMS: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                fires = self._parse_csv_response(response.text)
                logger.info(f"Retrieved {len(fires)} active fires from NASA FIRMS")
                return fires
            else:
                logger.error(
                    f"NASA FIRMS API error: HTTP {response.status_code} — {response.text[:200]}"
                )
                return []
                
        except Exception as e:
            logger.error(f"Error fetching NASA FIRMS data: {e}")
            return []
    
    def get_fires_by_country(self, country_code: str, hours: int = 24) -> List[Dict]:
        """
        Get active fires for a specific country
        
        Args:
            country_code: ISO 3-letter country code (e.g., 'NPL', 'USA')
            hours: Hours to look back
            
        Returns:
            List of fire dictionaries
        """
        return self.get_active_fires(region=country_code, hours=hours)
    
    def get_fires_by_bbox(self, min_lon: float, min_lat: float, 
                          max_lon: float, max_lat: float, hours: int = 24) -> List[Dict]:
        """
        Get active fires within a bounding box
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            hours: Hours to look back
            
        Returns:
            List of fire dictionaries
        """
        return self.get_active_fires(bbox=(min_lon, min_lat, max_lon, max_lat), hours=hours)
    
    def _parse_csv_response(self, csv_text: str) -> List[Dict]:
        """Parse CSV response from NASA FIRMS"""
        fires = []
        lines = csv_text.strip().split('\n')
        
        if len(lines) < 2:
            return fires
        
        # First line is header
        headers = lines[0].split(',')
        
        # Parse each fire detection
        for line in lines[1:]:
            values = line.split(',')
            if len(values) >= len(headers):
                fire = {}
                for i, header in enumerate(headers):
                    fire[header.strip()] = values[i].strip()
                
                # Handle different brightness column names used by NASA
                brightness = (
                    fire.get('bright_ti4') or 
                    fire.get('bright_t31') or 
                    fire.get('brightness') or 
                    fire.get('bright_ti5') or
                    fire.get('bright_t32') or
                    0
                )
                
                # Convert to standard format
                fires.append({
                    'latitude': float(fire.get('latitude', 0)),
                    'longitude': float(fire.get('longitude', 0)),
                    'brightness': float(brightness),
                    'confidence': self._parse_confidence(fire.get('confidence', 'nominal')),
                    'acq_date': fire.get('acq_date', ''),
                    'acq_time': fire.get('acq_time', ''),
                    'satellite': fire.get('satellite', 'unknown'),
                    'instrument': fire.get('instrument', 'unknown'),
                    'frp': float(fire.get('frp', 0))  # Fire Radiative Power
                })
        
        return fires
    
    def _parse_confidence(self, confidence_str: str) -> float:
        """Convert confidence string to numeric value"""
        confidence_map = {
            'low': 30.0,
            'nominal': 50.0,
            'high': 90.0
        }
        
        # Try to parse as number first
        try:
            return float(confidence_str)
        except ValueError:
            return confidence_map.get(confidence_str.lower(), 50.0)
    
    def _get_mock_fires(self, region: str = None, bbox: tuple = None) -> List[Dict]:
        """Return mock fire data for testing"""
        import random
        
        # Determine center point
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
        else:
            # Default to Nepal region
            center_lat, center_lon = 27.7, 85.3
        
        # Generate 3-5 mock fires
        num_fires = random.randint(3, 5)
        fires = []
        
        for i in range(num_fires):
            fires.append({
                'latitude': center_lat + random.uniform(-0.5, 0.5),
                'longitude': center_lon + random.uniform(-0.5, 0.5),
                'brightness': round(random.uniform(300, 400), 1),
                'confidence': round(random.uniform(70, 100), 1),
                'acq_date': datetime.now().strftime('%Y-%m-%d'),
                'acq_time': datetime.now().strftime('%H%M'),
                'satellite': 'Mock',
                'instrument': 'VIIRS',
                'frp': round(random.uniform(10, 100), 2)
            })
        
        logger.info(f"Generated {len(fires)} mock fires")
        return fires

# Singleton instance
_nasa_firms_api = None

def get_nasa_firms_api() -> NASAFirmsAPI:
    """Get or create NASA FIRMS API instance"""
    global _nasa_firms_api
    if _nasa_firms_api is None:
        _nasa_firms_api = NASAFirmsAPI()
    return _nasa_firms_api

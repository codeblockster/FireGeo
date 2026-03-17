"""
Data Collection Module
Handles fetching data from various APIs:
- NASA FIRMS (fire detection)
- Open-Meteo (weather data)
- Google Earth Engine (satellite data)
- Sentinel Hub (satellite imagery)
"""

from .nasa_firms import get_nasa_firms_api, NASAFirmsAPI
from .weather_api import WeatherDataFetcher

__all__ = ['get_nasa_firms_api', 'NASAFirmsAPI', 'WeatherDataFetcher']

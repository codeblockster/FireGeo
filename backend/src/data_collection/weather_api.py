"""
Weather Data Fetcher
Fetches current and historical weather data from Open-Meteo API
Provides all weather-related features for fire risk prediction
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class WeatherDataFetcher:
    """
    Weather data fetcher using Open-Meteo API (free, no key required).
    
    Fetches:
    - Current weather: temp, humidity, dewpoint, cloud cover, wind, precipitation
    - Historical weather: 14-day history for lag features and rolling averages
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather data fetcher.
        
        Args:
            api_key: API key (not used for Open-Meteo, kept for compatibility)
        """
        self.api_key = api_key
        self.base_url = "https://api.open-meteo.com/v1"
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        logger.info("Weather Data Fetcher initialized (using Open-Meteo)")
    
    def fetch_current_weather(self, lat: float, lon: float) -> Dict:
        """
        Fetch current weather data.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with current weather:
                - temp: Temperature (°C)
                - humidity: Relative humidity (%)
                - dewpoint: Dewpoint temperature (°C)
                - cloud_cover: Cloud coverage (%)
                - wind_speed: Wind speed (m/s)
                - wind_direction: Wind direction (degrees)
                - wind_u: U wind component (m/s)
                - wind_v: V wind component (m/s)
                - precip: Current precipitation (mm)
                - skin_temp: Skin temperature (°C)
                - soil_temp: Soil temperature 0-7cm (°C)
                - soil_moisture: Soil moisture 0-7cm (m³/m³)
        """
        endpoint = f"{self.base_url}/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'dew_point_2m',
                'cloud_cover',
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_u_component_10m',
                'wind_v_component_10m',
                'precipitation',
                'surface_temperature',
                'soil_temperature_0_to_7cm',
                'soil_moisture_0_to_7cm'
            ],
            'timezone': 'auto'
        }
        
        try:
            logger.debug(f"Fetching current weather for ({lat}, {lon})")
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get('current', {})
            
            result = {
                'temp': float(data.get('temperature_2m') if data.get('temperature_2m') is not None else 25.0),
                'humidity': float(data.get('relative_humidity_2m') if data.get('relative_humidity_2m') is not None else 50.0),
                'dewpoint': float(data.get('dew_point_2m') if data.get('dew_point_2m') is not None else 15.0),
                'cloud_cover': float(data.get('cloud_cover') if data.get('cloud_cover') is not None else 20.0),
                'wind_speed': float(data.get('wind_speed_10m') if data.get('wind_speed_10m') is not None else 10.0),
                'wind_direction': float(data.get('wind_direction_10m') if data.get('wind_direction_10m') is not None else 180.0),
                'wind_u': float(data.get('wind_u_component_10m') if data.get('wind_u_component_10m') is not None else 0.0),
                'wind_v': float(data.get('wind_v_component_10m') if data.get('wind_v_component_10m') is not None else 0.0),
                'precip': float(data.get('precipitation') if data.get('precipitation') is not None else 0.0),
                'skin_temp': float(data.get('surface_temperature') if data.get('surface_temperature') is not None else 25.0),
                'soil_temp': float(data.get('soil_temperature_0_to_7cm') if data.get('soil_temperature_0_to_7cm') is not None else 20.0),
                'soil_moisture': float(data.get('soil_moisture_0_to_7cm') if data.get('soil_moisture_0_to_7cm') is not None else 0.3)
            }
            
            logger.debug(f"Current weather: temp={result['temp']}°C, humidity={result['humidity']}%")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching current weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return None

    def get_historical_weather(self, lat: float, lon: float, days_back: int = 30) -> Dict:
        """
        Fetch historical weather for lag features and rolling averages.
        
        Args:
            lat: Latitude
            lon: Longitude
            days_back: Number of days of history to fetch (default: 30)
            
        Returns:
            Dictionary with historical data:
                - humidity: List of daily mean relative humidity (%)
                - precip: List of daily precipitation sum (mm)
                - vpd: List of calculated VPD values (kPa)
                - temp_mean: List of daily mean temperature (°C)
                - temp_min: List of daily minimum temperature (°C)
                - temp_max: List of daily maximum temperature (°C)
                - skin_temp: List of daily mean skin temperature (°C)
                - soil_temp: List of daily mean soil temperature (°C)
                - soil_moisture: List of daily mean soil moisture (m³/m³)
                - wind_speed: List of daily mean wind speed (m/s)
                - wind_direction: List of daily mean wind direction (degrees)
                - wind_u: List of daily mean u wind component (m/s)
                - wind_v: List of daily mean v wind component (m/s)
                - dates: List of date strings
        """
        # Archive API has a delay of 2-5 days. Use 7 days ago as end date to be safe.
        end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back + 7)).strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': [
                'temperature_2m_mean',
                'temperature_2m_min',
                'temperature_2m_max',
                'relative_humidity_2m_mean',
                'precipitation_sum',
                'wind_speed_10m_mean',
                'wind_direction_10m_dominant',
                'soil_temperature_0_to_7cm_mean',
                'soil_moisture_0_to_7cm_mean'
            ],
            'timezone': 'auto'
        }
        
        try:
            logger.debug(f"Fetching {days_back} days of historical weather for ({lat}, {lon})")
            response = requests.get(self.archive_url, params=params, timeout=15)
            response.raise_for_status()
            daily = response.json().get('daily', {})
            
            # Extract data
            temps_mean = daily.get('temperature_2m_mean', [])
            temps_min = daily.get('temperature_2m_min', [])
            temps_max = daily.get('temperature_2m_max', [])
            hums = daily.get('relative_humidity_2m_mean', [])
            precips = daily.get('precipitation_sum', [])
            wind_speeds = daily.get('wind_speed_10m_mean', [])
            wind_directions = daily.get('wind_direction_10m_dominant', [])
            soil_temps = daily.get('soil_temperature_0_to_7cm_mean', [])
            soil_moistures = daily.get('soil_moisture_0_to_7cm_mean', [])
            dates = daily.get('time', [])
            
            # Calculate wind U/V components from speed and direction
            wind_u = []
            wind_v = []
            for speed, direction in zip(wind_speeds, wind_directions):
                if speed is not None and direction is not None:
                    dir_rad = np.radians(float(direction))
                    wind_u.append(-float(speed) * np.sin(dir_rad))
                    wind_v.append(-float(speed) * np.cos(dir_rad))
                else:
                    wind_u.append(0.0)
                    wind_v.append(0.0)

            # Calculate VPD for each day
            vpds = []
            for temp, hum in zip(temps_mean, hums):
                if temp is not None and hum is not None:
                    vpds.append(self.calculate_vpd(temp, hum))
                else:
                    vpds.append(0.0)
            
            # Fill missing values with defaults
            def safe_list(data, default, length):
                """Ensure list has correct length with defaults for None values"""
                result = []
                for i in range(length):
                    if i < len(data) and data[i] is not None:
                        result.append(float(data[i]))
                    else:
                        result.append(default)
                return result
            
            n_days = len(dates) if dates else days_back
            
            result = {
                'humidity': safe_list(hums, 50.0, n_days),
                'precip': safe_list(precips, 0.0, n_days),
                'vpd': vpds if vpds else [1.0] * n_days,
                'temp_mean': safe_list(temps_mean, 25.0, n_days),
                'temp_min': safe_list(temps_min, 15.0, n_days),
                'temp_max': safe_list(temps_max, 30.0, n_days),
                'skin_temp': safe_list(temps_mean, 25.0, n_days),  # Use air temp as proxy
                'soil_temp': safe_list(soil_temps, 20.0, n_days),
                'soil_moisture': safe_list(soil_moistures, 0.3, n_days),
                'wind_speed': safe_list(wind_speeds, 10.0, n_days),
                'wind_direction': safe_list(wind_directions, 180.0, n_days),
                'wind_u': wind_u if wind_u else [0.0] * n_days,
                'wind_v': wind_v if wind_v else [0.0] * n_days,
                'dates': dates
            }
            
            logger.debug(f"Retrieved {len(result['dates'])} days of historical data")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching historical weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return None


    def calculate_vpd(self, temp_c: float, humidity_pct: float) -> float:
        """
        Calculate Vapor Pressure Deficit (kPa).
        
        Formula:
            VPD = es - ea
            es = 0.6108 * exp(17.27 * T / (T + 237.3))  [Saturation vapor pressure]
            ea = es * (RH / 100)                         [Actual vapor pressure]
        
        Args:
            temp_c: Temperature in Celsius
            humidity_pct: Relative humidity in percentage (0-100)
            
        Returns:
            VPD in kPa
        """
        try:
            if temp_c is None or humidity_pct is None:
                return 0.0
            
            # Saturation vapor pressure (kPa)
            es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
            
            # Actual vapor pressure (kPa)
            ea = es * (humidity_pct / 100.0)
            
            # VPD (kPa)
            vpd = es - ea
            
            return max(0.0, vpd)
            
        except Exception as e:
            logger.warning(f"VPD calculation error: {e}")
            return 0.0

    def get_forecast_weather(self, lat: float, lon: float, days_ahead: int = 7) -> Dict:
        """
        Fetch weather forecast for future fire risk prediction.
        
        Args:
            lat: Latitude
            lon: Longitude
            days_ahead: Number of days to forecast (default: 7)
            
        Returns:
            Dictionary with forecast data
        """
        endpoint = f"{self.base_url}/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'relative_humidity_2m_mean',
                'precipitation_sum',
                'wind_speed_10m_max',
                'wind_direction_10m_dominant'
            ],
            'forecast_days': days_ahead,
            'timezone': 'auto'
        }
        
        try:
            logger.debug(f"Fetching {days_ahead}-day forecast for ({lat}, {lon})")
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            daily = response.json().get('daily', {})
            
            return {
                'temp_max': daily.get('temperature_2m_max', []),
                'temp_min': daily.get('temperature_2m_min', []),
                'humidity': daily.get('relative_humidity_2m_mean', []),
                'precip': daily.get('precipitation_sum', []),
                'wind_speed': daily.get('wind_speed_10m_max', []),
                'wind_direction': daily.get('wind_direction_10m_dominant', []),
                'dates': daily.get('time', [])
            }
            
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return None

    def _get_mock_current(self) -> Dict:
        """Mock current weather data for fallback"""
        logger.warning("Using mock current weather data")
        return {
            'temp': 25.0,
            'humidity': 60.0,
            'dewpoint': 18.0,
            'cloud_cover': 10.0,
            'wind_speed': 15.0,
            'wind_direction': 180.0,
            'wind_u': 0.0,
            'wind_v': 0.0,
            'precip': 0.0,
            'skin_temp': 25.0,
            'soil_temp': 20.0,
            'soil_moisture': 0.3
        }

    def _get_mock_historical(self, days: int) -> Dict:
        """Mock historical weather data for fallback"""
        logger.warning(f"Using mock historical weather data ({days} days)")
        return {
            'humidity': [60.0] * days,
            'precip': [0.0] * days,
            'vpd': [1.2] * days,
            'temp_mean': [25.0] * days,
            'temp_min': [15.0] * days,
            'temp_max': [30.0] * days,
            'skin_temp': [25.0] * days,
            'soil_temp': [20.0] * days,
            'soil_moisture': [0.3] * days,
            'wind_speed': [10.0] * days,
            'wind_direction': [180.0] * days,
            'wind_u': [0.0] * days,
            'wind_v': [0.0] * days,
            'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(days, 0, -1)]
        }

    def _get_mock_forecast(self, days: int) -> Dict:
        """Mock forecast data for fallback"""
        logger.warning(f"Using mock forecast data ({days} days)")
        return {
            'temp_max': [30.0] * days,
            'temp_min': [20.0] * days,
            'humidity': [50.0] * days,
            'precip': [0.0] * days,
            'wind_speed': [15.0] * days,
            'wind_direction': [180.0] * days,
            'dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(1, days + 1)]
        }

    def get_summary(self, lat: float, lon: float) -> Dict:
        """
        Get comprehensive weather summary for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with current, historical summary, and forecast
        """
        current = self.fetch_current_weather(lat, lon)
        historical = self.get_historical_weather(lat, lon, days_back=7)
        
        # Calculate trends
        recent_temps = historical['temp_mean'][-7:] if len(historical['temp_mean']) >= 7 else []
        recent_humidity = historical['humidity'][-7:] if len(historical['humidity']) >= 7 else []
        
        return {
            'current': current,
            'trends': {
                'avg_temp_7d': np.mean(recent_temps) if recent_temps else 25.0,
                'avg_humidity_7d': np.mean(recent_humidity) if recent_humidity else 50.0,
                'total_precip_7d': sum(historical['precip'][-7:]) if historical['precip'] else 0.0
            },
            'location': {
                'latitude': lat,
                'longitude': lon
            }
        }

# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Weather Data Fetcher Test")
    print("=" * 70)
    
    fetcher = WeatherDataFetcher()
    
    # Test location: Kathmandu, Nepal
    lat, lon = 28.3949, 84.1240
    
    print(f"\n1. Testing current weather for ({lat}, {lon})...")
    current = fetcher.fetch_current_weather(lat, lon)
    print(f"   Temperature: {current['temp']}°C")
    print(f"   Humidity: {current['humidity']}%")
    print(f"   Wind: {current['wind_speed']} m/s from {current['wind_direction']}°")
    
    print(f"\n2. Testing historical weather (15 days)...")
    historical = fetcher.get_historical_weather(lat, lon, days_back=15)
    print(f"   Retrieved {len(historical['dates'])} days")
    print(f"   Last 3 days humidity: {historical['humidity'][-3:]}")
    print(f"   Last 3 days VPD: {[round(v, 2) for v in historical['vpd'][-3:]]}")
    
    print(f"\n3. Testing weather summary...")
    summary = fetcher.get_summary(lat, lon)
    print(f"   7-day avg temp: {summary['trends']['avg_temp_7d']:.1f}°C")
    print(f"   7-day avg humidity: {summary['trends']['avg_humidity_7d']:.1f}%")
    print(f"   7-day total precip: {summary['trends']['total_precip_7d']:.1f}mm")
    
    print("\n" + "=" * 70)
    print(" All tests complete!")

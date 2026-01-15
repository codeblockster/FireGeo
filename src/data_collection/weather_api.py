import requests

class WeatherDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        # Use OpenWeatherMap or similar
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def fetch_current_weather(self, lat, lon):
        """Fetch current weather data"""
        endpoint = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {}

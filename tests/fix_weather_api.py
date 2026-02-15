"""
Test and fix Weather API key
"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()

weather_key = os.getenv('WEATHER_API_KEY')
print(f"Current WEATHER_API_KEY: {weather_key}")
print(f"Key length: {len(weather_key) if weather_key else 0}")

# Test it
lat, lon = 28.3949, 84.1240
url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_key}&units=metric"

print(f"\nTesting API call...")
response = requests.get(url, timeout=10)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")

if response.status_code == 401:
    print("\n❌ API Key is INVALID!")
    print("\n📝 To fix:")
    print("1. Go to: https://openweathermap.org/api")
    print("2. Sign up (free account)")
    print("3. Go to API Keys section")
    print("4. Copy your API key")
    print("5. Update .env file:")
    print("   WEATHER_API_KEY=your_actual_key_here")
    print("\n⚠️  Note: New keys may take 1-2 hours to activate")
elif response.status_code == 200:
    print("\n✅ API Key is VALID!")
    data = response.json()
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Humidity: {data['main']['humidity']}%")
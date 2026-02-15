"""
API Integration Tests
Tests all external API connections: Weather API, NASA FIRMS, Google Earth Engine
Run: python tests/test_api.py
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# TEST: Environment Variables
# ============================================================================

def test_env_file_exists():
    """Test that .env file exists"""
    env_path = project_root / '.env'
    assert env_path.exists(), f".env file not found at {env_path}"
    logger.info(f"✓ .env file found at: {env_path}")


def test_weather_api_key_exists():
    """Test that WEATHER_API_KEY is set"""
    weather_key = os.getenv('WEATHER_API_KEY')
    assert weather_key is not None, "WEATHER_API_KEY not found in environment"
    assert weather_key != 'your_weather_api_key_here', "WEATHER_API_KEY is still placeholder"
    logger.info(f"✓ WEATHER_API_KEY found: {weather_key[:8]}...")


def test_nasa_firms_key_exists():
    """Test that NASA_FIRMS_API_KEY is set"""
    nasa_key = os.getenv('NASA_FIRMS_API_KEY')
    assert nasa_key is not None, "NASA_FIRMS_API_KEY not found in environment"
    assert nasa_key != 'your_nasa_firms_key_here', "NASA_FIRMS_API_KEY is still placeholder"
    logger.info(f"✓ NASA_FIRMS_API_KEY found: {nasa_key[:8]}...")


# ============================================================================
# TEST: Weather API
# ============================================================================

def test_weather_api_connection():
    """Test Weather API connection with real request"""
    import requests
    
    weather_key = os.getenv('WEATHER_API_KEY')
    if not weather_key:
        raise Exception("WEATHER_API_KEY not configured - SKIPPED")
    
    # Test location: Kathmandu, Nepal
    lat = 28.3949
    lon = 84.1240
    
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
    
    response = requests.get(url, timeout=10)
    
    assert response.status_code == 200, f"Weather API failed with status {response.status_code}"
    
    data = response.json()
    assert 'main' in data, "Weather API response missing 'main' field"
    assert 'temp' in data['main'], "Weather API response missing temperature"
    assert 'humidity' in data['main'], "Weather API response missing humidity"
    
    temp = data['main']['temp']
    humidity = data['main']['humidity']
    
    logger.info(f"✓ Weather API working - Temp: {temp}°C, Humidity: {humidity}%")


def test_weather_api_invalid_key():
    """Test Weather API fails gracefully with invalid key"""
    import requests
    
    lat = 28.3949
    lon = 84.1240
    fake_key = "invalid_key_12345"
    
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={fake_key}&units=metric"
    
    response = requests.get(url, timeout=10)
    
    assert response.status_code == 401, "Expected 401 for invalid API key"
    logger.info("✓ Weather API correctly rejects invalid key")


# ============================================================================
# TEST: NASA FIRMS API
# ============================================================================

def test_nasa_firms_api_connection():
    """Test NASA FIRMS API connection"""
    import requests
    
    nasa_key = os.getenv('NASA_FIRMS_API_KEY')
    if not nasa_key:
        raise Exception("NASA_FIRMS_API_KEY not configured - SKIPPED")
    
    # Test with Nepal country code
    country_code = 'NPL'
    source = 'VIIRS_SNPP_NRT'
    days = 1
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{nasa_key}/{source}/{country_code}/{days}"
    
    logger.info(f"Testing NASA FIRMS API...")
    response = requests.get(url, timeout=30)
    
    assert response.status_code == 200, f"NASA FIRMS API failed with status {response.status_code}: {response.text}"
    
    # Parse CSV response
    lines = response.text.strip().split('\n')
    assert len(lines) >= 1, "NASA FIRMS returned empty response"
    
    # Check header
    header = lines[0]
    assert 'latitude' in header.lower(), "NASA FIRMS response missing latitude column"
    assert 'longitude' in header.lower(), "NASA FIRMS response missing longitude column"
    
    fire_count = len(lines) - 1  # Subtract header
    logger.info(f"✓ NASA FIRMS API working - Found {fire_count} fire detections in {country_code}")


def test_nasa_firms_bbox_query():
    """Test NASA FIRMS API with bounding box query"""
    import requests
    
    nasa_key = os.getenv('NASA_FIRMS_API_KEY')
    if not nasa_key:
        raise Exception("NASA_FIRMS_API_KEY not configured - SKIPPED")
    
    # Bounding box for Nepal
    min_lon, min_lat, max_lon, max_lat = 80.0, 26.0, 89.0, 31.0
    source = 'VIIRS_SNPP_NRT'
    days = 1
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{nasa_key}/{source}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}"
    
    logger.info(f"Testing NASA FIRMS BBox query...")
    response = requests.get(url, timeout=30)
    
    assert response.status_code == 200, f"NASA FIRMS BBox query failed: {response.status_code}"
    
    lines = response.text.strip().split('\n')
    fire_count = max(0, len(lines) - 1)
    
    logger.info(f"✓ NASA FIRMS BBox query working - Found {fire_count} fires in bbox")


def test_nasa_firms_invalid_key():
    """Test NASA FIRMS API fails gracefully with invalid key"""
    import requests
    
    fake_key = "invalid_nasa_key_12345"
    country_code = 'NPL'
    source = 'VIIRS_SNPP_NRT'
    days = 1
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{fake_key}/{source}/{country_code}/{days}"
    
    response = requests.get(url, timeout=10)
    
    # NASA FIRMS returns various error codes for invalid keys
    assert response.status_code != 200, "Expected error for invalid NASA FIRMS key"
    logger.info(f"✓ NASA FIRMS correctly rejects invalid key (status: {response.status_code})")


def test_nasa_firms_data_fields():
    """Test that NASA FIRMS returns all required data fields"""
    import requests
    
    nasa_key = os.getenv('NASA_FIRMS_API_KEY')
    if not nasa_key:
        raise Exception("NASA_FIRMS_API_KEY not configured - SKIPPED")
    
    # Nepal bounding box
    min_lon, min_lat, max_lon, max_lat = 80.0, 26.0, 89.0, 31.0
    source = 'VIIRS_SNPP_NRT'
    days = 7  # Use 7 days to increase chance of finding fires
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{nasa_key}/{source}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}"
    
    response = requests.get(url, timeout=30)
    assert response.status_code == 200, "NASA FIRMS API request failed"
    
    lines = response.text.strip().split('\n')
    
    if len(lines) > 1:  # If we have fire data
        header = lines[0].lower()
        
        # Check for required fields
        required_fields = ['latitude', 'longitude', 'confidence', 'acq_date']
        for field in required_fields:
            assert field in header, f"NASA FIRMS response missing required field: {field}"
        
        logger.info(f"✓ NASA FIRMS returns all required fields")
    else:
        logger.info("⚠️  No fires detected in test region (this is okay)")


# ============================================================================
# TEST: Google Earth Engine
# ============================================================================

def test_earth_engine_library():
    """Test that Earth Engine library is installed"""
    try:
        import ee
        logger.info("✓ Earth Engine library installed")
    except ImportError:
        raise Exception("Earth Engine library not installed - SKIPPED")


def test_earth_engine_authentication():
    """Test Earth Engine authentication"""
    try:
        import ee
        ee.Initialize()
        logger.info("✓ Earth Engine authenticated")
    except ImportError:
        raise Exception("Earth Engine library not installed - SKIPPED")
    except Exception as e:
        raise Exception(f"Earth Engine not authenticated - SKIPPED: {e}")


def test_earth_engine_basic_query():
    """Test basic Earth Engine data query"""
    try:
        import ee
        ee.Initialize()
        
        # Test with Nepal coordinates
        point = ee.Geometry.Point([84.1240, 28.3949])
        
        # Get elevation from SRTM
        elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().get('elevation').getInfo()
        
        assert elevation is not None, "Earth Engine query returned None"
        assert isinstance(elevation, (int, float)), "Earth Engine elevation should be numeric"
        assert 0 < elevation < 9000, f"Unexpected elevation value: {elevation}"
        
        logger.info(f"✓ Earth Engine query working - Elevation: {elevation}m")
        
    except ImportError:
        raise Exception("Earth Engine library not installed - SKIPPED")
    except Exception as e:
        if 'not authenticated' in str(e).lower():
            raise Exception("Earth Engine not authenticated - SKIPPED")
        else:
            raise


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    """Run tests directly"""
    
    print("=" * 70)
    print("API INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Environment Variables", [
            test_env_file_exists,
            test_weather_api_key_exists,
            test_nasa_firms_key_exists
        ]),
        ("Weather API", [
            test_weather_api_connection,
            test_weather_api_invalid_key
        ]),
        ("NASA FIRMS API", [
            test_nasa_firms_api_connection,
            test_nasa_firms_bbox_query,
            test_nasa_firms_invalid_key,
            test_nasa_firms_data_fields
        ]),
        ("Google Earth Engine", [
            test_earth_engine_library,
            test_earth_engine_authentication,
            test_earth_engine_basic_query
        ])
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for category, test_funcs in tests:
        print(f"\n{category}")
        print("-" * 70)
        
        for test_func in test_funcs:
            total_tests += 1
            test_name = test_func.__doc__ or test_func.__name__
            
            try:
                test_func()
                print(f"  ✓ PASS: {test_name}")
                passed_tests += 1
            except Exception as e:
                error_msg = str(e)
                if "SKIPPED" in error_msg:
                    print(f"  ⊘ SKIP: {test_name}")
                    print(f"         {error_msg}")
                    skipped_tests += 1
                else:
                    print(f"  ✗ FAIL: {test_name}")
                    print(f"         {error_msg}")
                    failed_tests += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:     {total_tests}")
    print(f"✓ Passed:  {passed_tests}")
    print(f"✗ Failed:  {failed_tests}")
    print(f"⊘ Skipped: {skipped_tests}")
    
    if failed_tests == 0 and passed_tests > 0:
        print("\n✅ All configured tests passed!")
    elif failed_tests > 0:
        print(f"\n⚠️  {failed_tests} test(s) failed")
    
    print("=" * 70)
import ee
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Google Earth Engine Authentication")
print("=" * 70)

print("\nStep 1: Starting authentication...")
print("This will open your browser for Google login.\n")

try:
    # Authenticate
    ee.Authenticate()
    print("\n Authentication successful!")
    
    # Initialize with project
    print("\nStep 2: Initializing Earth Engine...")
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'cellular-smoke-486811-u8')
    print(f"Using project: {project_id}")
    
    ee.Initialize(project=project_id) 
    
    # Test it
    print("\nStep 3: Testing connection...")
    point = ee.Geometry.Point([84.1240, 28.3949])
    elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().get('elevation').getInfo()
    
    print(f" SUCCESS! Elevation: {elevation}m")
    print("\nYour project is now working with Earth Engine!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have a Google account")
    print("2. Make sure you're signed up for Earth Engine at:")
    print("   https://earthengine.google.com/signup/")
    print("3. Make sure GOOGLE_CLOUD_PROJECT is set in .env")

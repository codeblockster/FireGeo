"""
Test Google Earth Engine with project from .env
"""
import os
import ee
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("Testing Google Earth Engine with Project from .env")
print("=" * 70)

# Get project ID from environment
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

print(f"\nGOOGLE_CLOUD_PROJECT from .env: {project_id}")

if not project_id:
    print("\n❌ GOOGLE_CLOUD_PROJECT not found in .env file!")
    print("\nAdd this to your .env file:")
    print("GOOGLE_CLOUD_PROJECT=your-project-id-here")
    exit(1)

print(f"\nInitializing Earth Engine with project: {project_id}")

try:
    # Initialize with project from env
    ee.Initialize(project=project_id)
    print("✅ Initialization successful!")
    
    # Test query
    print("\nTesting query...")
    point = ee.Geometry.Point([84.1240, 28.3949])
    elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().get('elevation').getInfo()
    
    print(f"✅ Query successful!")
    print(f"   Elevation at Kathmandu: {elevation}m")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Google Earth Engine is working with your project.")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your project ID is correct")
    print("2. Check it at: https://console.cloud.google.com/")
    print("3. Make sure Earth Engine API is enabled for your project:")
    print("   https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
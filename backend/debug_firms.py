import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.firedetect.fire_detector import FireDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_firms():
    print("Initializing FireDetector...")
    detector = FireDetector()
    
    print("\nTesting 'Nepal' region (Country Code: NPL)...")
    try:
        result = detector.detect_fires("Nepal", hours=24)
        print(f"Result count: {result.get('count')}")
        if result.get('count') == 0:
             print("No fires found. Checking error...")
             if 'error' in result:
                 print(f"Error: {result['error']}")
        else:
             print("Fires found!")
             print(result['fires'][0] if result['fires'] else "No fire list")
    except Exception as e:
        print(f"Exception during detection: {e}")

    print("\nTesting 'Whole World' region...")
    try:
        result = detector.detect_fires("Whole World", hours=24)
        print(f"Result count: {result.get('count')}")
        if result.get('count') == 0:
             print("No fires found. Checking error...")
             if 'error' in result:
                 print(f"Error: {result['error']}")
        else:
             print("Fires found!")
             print(result['fires'][0] if result['fires'] else "No fire list")
    except Exception as e:
        print(f"Exception during detection: {e}")

if __name__ == "__main__":
    test_firms()

try:
    import catboost
    print("CatBoost Installed")
except ImportError:
    print("CatBoost Missing")

try:
    import ee
    print("Earth Engine Installed")
except ImportError:
    print("Earth Engine Missing")

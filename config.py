"""
Central Configuration for Wildfire Management System
Defines all paths and settings used across the application.
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"

# Model subdirectories
PRE_FIRE_MODELS_DIR = MODELS_DIR / "pre_fire"
POST_FIRE_MODELS_DIR = MODELS_DIR / "post_fire"
SPREAD_MODELS_DIR = MODELS_DIR / "spread"

# Specific model paths
ENSEMBLE_MODELS = {
    'catboost': PRE_FIRE_MODELS_DIR / "catboost_s_tier_model.pkl",
    'lightgbm': PRE_FIRE_MODELS_DIR / "lightgbm_best_model.pkl",
    'xgboost': PRE_FIRE_MODELS_DIR / "xgboost_enhanced_model.joblib"
}

XGBOOST_MODEL = PRE_FIRE_MODELS_DIR / "xgboost_fire_risk.pkl"
UNET_MODEL = SPREAD_MODELS_DIR / "unet_fire_spread.pth"

# Scripts directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAINING_SCRIPTS_DIR = SCRIPTS_DIR / "training"
DATA_COLLECTION_SCRIPTS_DIR = SCRIPTS_DIR / "data_collection"
ANALYSIS_SCRIPTS_DIR = SCRIPTS_DIR / "analysis"

# Source code directories
SRC_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TESTS_DIR = PROJECT_ROOT / "tests"

# Environment variables
GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT = os.getenv('GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT')
GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH = os.getenv('GOOGLE_EARTH_ENGINE_PRIVATE_KEY_PATH')
NASA_FIRMS_API_KEY = os.getenv('NASA_FIRMS_API_KEY')


# Model Configuration
DEFAULT_ENSEMBLE_WEIGHTS = {
    'catboost': 0.5,
    'lightgbm': 0.3,
    'xgboost': 0.2
}

# Risk Map Configuration
DEFAULT_GRID_RESOLUTION = 5
DEFAULT_SIZE_KM = 10

# Create directories if they don't exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAINING_DATA_DIR,
        PRE_FIRE_MODELS_DIR,
        POST_FIRE_MODELS_DIR,
        SPREAD_MODELS_DIR,
        SCRIPTS_DIR,
        TRAINING_SCRIPTS_DIR,
        DATA_COLLECTION_SCRIPTS_DIR,
        ANALYSIS_SCRIPTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # When run directly, create all directories and print configuration
    ensure_directories()
    print("✅ Directory structure created successfully!")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Scripts Directory: {SCRIPTS_DIR}")

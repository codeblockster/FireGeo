
import pickle
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def check_model_features():
    model_path = Path("f:/v4 cleanup/backend/prefire/models/catboost_s_tier_model.pkl")
    if not model_path.exists():
        print("Model file not found")
        return

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'feature_names_'):
            print(f"Features ({len(model.feature_names_)}):")
            for i, name in enumerate(model.feature_names_):
                print(f"{i+1}: {name}")
        else:
            print("Model has no feature_names_ attribute")
            # Try to see total features if it's a CatBoost model
            try:
                print(f"Total features: {model.feature_count_}")
            except:
                pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model_features()

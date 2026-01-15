import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.models.unet_fire import build_unet_model
from src.analysis.spread_prediction import SpreadPredictor

def check_deps():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    try:
        model = build_unet_model()
        print("Successfully built U-Net model.")
    except Exception as e:
        print(f"Failed to build model: {e}")
        return

    try:
        predictor = SpreadPredictor()
        print("Successfully initialized SpreadPredictor.")
    except Exception as e:
        print(f"Failed to init predictor: {e}")

if __name__ == "__main__":
    check_deps()

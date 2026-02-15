"""
Simple test to list all generated features
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.prefire.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
features = engineer._get_mock_features()

print(f"Total features: {len(features)}\n")
print("All features:")
for i, name in enumerate(sorted(features.keys()), 1):
    print(f"{i:3d}. {name}")

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection.sentinel_manager import SentinelManager
from src.analysis.post_fire import PostFireAnalyzer
from src.analysis.pre_fire import PreFireAnalyzer
from src.analysis.spread_prediction import SpreadPredictor

def test_sentinel_manager_init():
    with patch('src.data_collection.sentinel_manager.ee') as mock_ee:
        sm = SentinelManager()
        assert sm is not None

def test_post_fire_analyzer_init():
    with patch('src.data_collection.sentinel_manager.ee') as mock_ee:
        pfa = PostFireAnalyzer()
        assert pfa is not None

def test_pre_fire_analyzer_init():
    # Mock GEE extractor
    with patch('src.analysis.pre_fire.get_gee_extractor') as mock_get_gee:
        mock_get_gee.return_value = MagicMock()
        pfa = PreFireAnalyzer(model_path="dummy_path")
        assert pfa is not None

def test_spread_predictor_init():
    sp = SpreadPredictor(model_path="dummy_path")
    assert sp is not None

def test_mock_spread_prediction():
    sp = SpreadPredictor(model_path="dummy_path")
    # Should use simulation mode
    result = sp.predict_spread(28.0, 84.0, "2023-01-01")
    assert result["type"] == "Feature"
    assert result["geometry"]["type"] == "Polygon"

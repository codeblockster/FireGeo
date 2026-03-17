import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_vpd(temp_c, humidity_pct):
    """
    Calculate Vapor Pressure Deficit (kPa)
    VPD = es - ea
    """
    try:
        if temp_c is None or humidity_pct is None:
            return 0.0
        
        # Saturation Vapor Pressure (es)
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        
        # Actual Vapor Pressure (ea)
        ea = es * (humidity_pct / 100.0)
        
        return max(0.0, es - ea)
    except Exception as e:
        logger.error(f"VPD calculation error: {e}")
        return 0.0

def calculate_rolling_mean(values):
    """Calculate mean of a list of values"""
    if not values:
        return 0.0
    return float(np.mean(values))

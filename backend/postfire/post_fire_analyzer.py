try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("Earth Engine (ee) not available. Post-fire analysis will use mock data.")
from typing import Dict, Any
from datetime import datetime, timedelta
import logging
from backend.src.data_collection.sentinel_manager import SentinelManager

logger = logging.getLogger(__name__)

class PostFireAnalyzer:
    """
    Analyzer for Post-Fire Assessment using dNBR.
    """
    
    def __init__(self):
        self.sentinel_manager = SentinelManager()
        
    def analyze_burn_severity(self, lat: float, lon: float, 
                            pre_date: str = None, 
                            post_date: str = None, 
                            buffer_km: int = 10) -> Dict[str, Any]:
        """
        Perform dNBR analysis and return map layers for visualization.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            pre_date: Date before fire (YYYY-MM-DD). Defaults to 1 year ago.
            post_date: Date after fire (YYYY-MM-DD). Defaults to today.
            buffer_km: Radius of analysis in km.
            
        Returns:
            Dictionary containing map IDs and stats.
        """
        try:
            if not EE_AVAILABLE:
                return {
                    "error": "Earth Engine API not available. Cannot perform burn severity analysis.",
                    "status": "failed",
                    "mock": True
                }

            # 1. Define ROI
            point = ee.Geometry.Point([lon, lat])
            roi = point.buffer(buffer_km * 1000)
            
            # 2. Determine Dates if not provided
            if not post_date:
                post_date = datetime.now().strftime('%Y-%m-%d')
            if not pre_date:
                # Default to same season previous year for better comparison
                post_dt = datetime.strptime(post_date, '%Y-%m-%d')
                pre_date = (post_dt - timedelta(days=365)).strftime('%Y-%m-%d')
                
            # Define time windows (e.g., 1 month window to find cloud-free images)
            pre_start = (datetime.strptime(pre_date, '%Y-%m-%d') - timedelta(days=15)).strftime('%Y-%m-%d')
            pre_end = (datetime.strptime(pre_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')
            
            post_start = (datetime.strptime(post_date, '%Y-%m-%d') - timedelta(days=15)).strftime('%Y-%m-%d')
            post_end = (datetime.strptime(post_date, '%Y-%m-%d') + timedelta(days=15)).strftime('%Y-%m-%d')

            # 3. Fetch Images
            pre_img = self.sentinel_manager.get_sentinel2_image(roi, pre_start, pre_end)
            post_img = self.sentinel_manager.get_sentinel2_image(roi, post_start, post_end)
            
            if not pre_img or not post_img:
                return {
                    "error": "Could not find cloud-free Sentinel-2 imagery for the specified dates.",
                    "status": "failed"
                }

            # 4. Calculate dNBR
            dnbr = self.sentinel_manager.calculate_dnbr(pre_img, post_img)
            severity = self.sentinel_manager.classify_burn_severity(dnbr)
            
            # 5. Generate Vis Parameters
            dnbr_vis = {'min': -0.5, 'max': 1.3, 'palette': ['00b700', 'ffff00', 'ffaa00', 'ff0000', '880022']}
            severity_vis = {'min': 0, 'max': 4, 'palette': ['00b700', 'ffff00', 'ffaa00', 'ff0000', '880022']}
            
            # 6. Get Tile Layers (MapID)
            # MapID allows the frontend to load tiles directly from Google servers
            dnbr_mapid = dnbr.getMapId(dnbr_vis)
            severity_mapid = severity.getMapId(severity_vis)
            
            # 7. Calculate Statistics (Burned Area in Hectares)
            # Count pixels in each severity class
            stats = severity.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=roi,
                scale=20, # Sentinel-2 resolution
                maxPixels=1e9
            ).get('Severity').getInfo()
            
            return {
                "status": "success",
                "center": [lat, lon],
                "roi_geojson": roi.getInfo(),
                "layers": {
                    "dnbr": {
                        "mapid": dnbr_mapid['mapid'],
                        "token": dnbr_mapid['token'],
                        "vis_params": dnbr_vis
                    },
                    "severity": {
                        "mapid": severity_mapid['mapid'],
                        "token": severity_mapid['token'],
                        "vis_params": severity_vis
                    }
                },
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Post-fire analysis failed: {e}")
            return {"error": str(e), "status": "failed"}

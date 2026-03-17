"""
Fire Detection Module
Handles active fire detection using NASA FIRMS satellite data.
"""

from backend.src.data_collection.nasa_firms import get_nasa_firms_api
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FireDetector:
    
    
    def __init__(self):
        """Initialize the fire detector with NASA FIRMS API."""
        self.firms_api = get_nasa_firms_api()
        
        
        self.region_map = {
            "Whole World": ("world", [0, 0], 2),
            
            # Nepal - Using bounding box for reliable API queries
            # (min_lon, min_lat, max_lon, max_lat) covers entire Nepal
            "Nepal": ((80.0, 26.0, 88.0, 30.5), [28.3949, 84.1240], 7),
            
            # Other regions with bounding boxes
            "California": ((-124.5, 32.5, -114.1, 42.0), [36.7, -119.4], 6),
            "Australia": ("AUS", [-25.2744, 133.7751], 4),
            "Indonesia": ("IDN", [-2.5, 118.0], 5),
            "India": ("IND", [20.5937, 78.9629], 5),
            
        }
    
    def get_region_map(self) -> Dict[str, Tuple]:
        
        return self.region_map
    
    def add_custom_region(self, 
                         name: str, 
                         region_code, 
                         center: List[float], 
                         zoom: int = 6) -> None:
        
        self.region_map[name] = (region_code, center, zoom)
        logger.info(f"Added custom region: {name}")
    
    def detect_fires(self, 
                    region: str, 
                    hours: int = 24,
                    source: str = 'VIIRS_SNPP_NRT') -> Dict:
        
        if region not in self.region_map:
            available = ", ".join(self.region_map.keys())
            raise ValueError(f"Unknown region: '{region}'. Available regions: {available}")
        
        region_code, center, zoom = self.region_map[region]
        
        try:
            # USE NASA FIRMS API DIRECTLY (with caching)
            # NASA FIRMS API uses days, not hours
            # But we accept hours for user convenience
            days = max(1, min(10, hours // 24))
            if hours < 24:
                days = 1
            
            logger.info(f"Detecting fires in {region} (last {hours}h / {days}d) using {source}")
            
            # Determine API call based on region_code type
            if isinstance(region_code, tuple):
                # BBox query
                fires = self.firms_api.get_active_fires(
                    bbox=region_code, 
                    hours=hours,
                    source=source
                )
            elif region_code == "world":
                # Global query
                fires = self.firms_api.get_active_fires(
                    region="world", 
                    hours=hours,
                    source=source
                )
            else:
                # Country code query (e.g., 'NPL' for Nepal)
                fires = self.firms_api.get_active_fires(
                    region=region_code, 
                    hours=hours,
                    source=source
                )
            
            logger.info(f"Successfully detected {len(fires)} fires in {region}")
            
            return {
                "count": len(fires),
                "fires": fires,
                "source": f"NASA FIRMS ({source})",
                "hours": hours,
                "days": days,
                "center": center,
                "zoom": zoom,
                "region": region,
                "region_code": region_code,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error detecting fires in {region}: {e}")
            return {
                "count": 0,
                "fires": [],
                "source": source,
                "hours": hours,
                "center": center,
                "zoom": zoom,
                "region": region,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False
            }
    
    def get_fires_dataframe(self, fires_data: Dict) -> pd.DataFrame:
        
        if fires_data['count'] == 0:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(fires_data['fires'])
            
            # Standard columns that should always be present
            required_cols = ['latitude', 'longitude', 'brightness', 'confidence', 'acq_date']
            
            # Optional columns that add value if available
            optional_cols = [
                'acq_time',      # Acquisition time
                'frp',           # Fire Radiative Power (MW)
                'daynight',      # D=day, N=night
                'satellite',     # Satellite name
                'instrument',    # VIIRS or MODIS
                'type',          # Fire type
                'scan',          # Scan value
                'track',         # Track value
                'version'        # Data version
            ]
            
            # Select available columns
            available_cols = [col for col in required_cols + optional_cols if col in df.columns]
            result_df = df[available_cols].copy()
            
            # Parse datetime
            if 'acq_date' in result_df.columns:
                result_df['acq_datetime'] = pd.to_datetime(result_df['acq_date'], errors='coerce')
                
                # Add time if available
                if 'acq_time' in result_df.columns:
                    result_df['acq_time_str'] = result_df['acq_time'].astype(str).str.zfill(4)
                    result_df['acq_datetime'] = pd.to_datetime(
                        result_df['acq_date'] + ' ' + 
                        result_df['acq_time_str'].str[:2] + ':' + 
                        result_df['acq_time_str'].str[2:],
                        errors='coerce'
                    )
            
            # Convert numeric columns
            numeric_cols = ['latitude', 'longitude', 'brightness', 'confidence', 'frp']
            for col in numeric_cols:
                if col in result_df.columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # Sort by acquisition time (most recent first)
            if 'acq_datetime' in result_df.columns:
                result_df = result_df.sort_values('acq_datetime', ascending=False)
            
            # Add confidence category
            if 'confidence' in result_df.columns:
                result_df['confidence_level'] = result_df['confidence'].apply(
                    self._categorize_confidence
                )
            
            logger.info(f"Created DataFrame with {len(result_df)} fires and {len(result_df.columns)} columns")
            return result_df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def _categorize_confidence(self, confidence: float) -> str:
        
        try:
            conf_val = float(confidence)
            if conf_val < 30:
                return 'Low'
            elif conf_val < 80:
                return 'Nominal'
            else:
                return 'High'
        except (ValueError, TypeError):
            # Handle string values like 'l', 'n', 'h'
            conf_str = str(confidence).lower()
            if conf_str in ['l', 'low']:
                return 'Low'
            elif conf_str in ['h', 'high']:
                return 'High'
            else:
                return 'Nominal'
    
    def filter_by_confidence(self, 
                            fires_data: Dict, 
                            min_confidence: str = 'nominal') -> Dict:
        
        if fires_data['count'] == 0:
            return fires_data
        
        # Confidence thresholds
        thresholds = {
            'low': 0,
            'nominal': 30,
            'high': 80
        }
        
        threshold = thresholds.get(min_confidence.lower(), 30)
        
        filtered_fires = []
        for fire in fires_data['fires']:
            try:
                conf = float(fire.get('confidence', 0))
                if conf >= threshold:
                    filtered_fires.append(fire)
            except (ValueError, TypeError):
                # Handle string confidence values
                conf_str = str(fire.get('confidence', '')).lower()
                if min_confidence.lower() == 'low' or \
                   (min_confidence.lower() == 'nominal' and conf_str in ['n', 'h', 'nominal', 'high']) or \
                   (min_confidence.lower() == 'high' and conf_str in ['h', 'high']):
                    filtered_fires.append(fire)
        
        result = fires_data.copy()
        result['count'] = len(filtered_fires)
        result['fires'] = filtered_fires
        result['filtered_by'] = f'confidence >= {min_confidence}'
        
        logger.info(f"Filtered to {len(filtered_fires)} fires with confidence >= {min_confidence}")
        return result
    
    def get_statistics(self, fires_data: Dict) -> Dict:
        
        if fires_data['count'] == 0:
            return {
                'total_fires': 0,
                'region': fires_data.get('region', 'Unknown'),
                'message': 'No fires detected in this region'
            }
        
        df = pd.DataFrame(fires_data['fires'])
        
        stats = {
            'total_fires': fires_data['count'],
            'region': fires_data.get('region', 'Unknown'),
            'time_range_hours': fires_data.get('hours', 24),
            'data_source': fires_data.get('source', 'NASA FIRMS'),
            'timestamp': fires_data.get('timestamp', datetime.utcnow().isoformat())
        }
        
        # Brightness statistics
        if 'brightness' in df.columns:
            df['brightness'] = pd.to_numeric(df['brightness'], errors='coerce')
            stats['brightness'] = {
                'mean': round(df['brightness'].mean(), 2),
                'min': round(df['brightness'].min(), 2),
                'max': round(df['brightness'].max(), 2),
                'median': round(df['brightness'].median(), 2)
            }
        
        # Confidence statistics
        if 'confidence' in df.columns:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            stats['confidence'] = {
                'mean': round(df['confidence'].mean(), 2),
                'high_confidence_count': int((df['confidence'] >= 80).sum()),
                'nominal_confidence_count': int(((df['confidence'] >= 30) & (df['confidence'] < 80)).sum()),
                'low_confidence_count': int((df['confidence'] < 30).sum())
            }
        
        # Fire Radiative Power (FRP) statistics
        if 'frp' in df.columns:
            df['frp'] = pd.to_numeric(df['frp'], errors='coerce')
            stats['frp'] = {
                'total_MW': round(df['frp'].sum(), 2),
                'mean_MW': round(df['frp'].mean(), 2),
                'max_MW': round(df['frp'].max(), 2)
            }
        
        # Satellite breakdown
        if 'satellite' in df.columns:
            stats['by_satellite'] = df['satellite'].value_counts().to_dict()
        
        # Day/Night breakdown
        if 'daynight' in df.columns:
            stats['by_time_of_day'] = df['daynight'].value_counts().to_dict()
        
        # Instrument breakdown
        if 'instrument' in df.columns:
            stats['by_instrument'] = df['instrument'].value_counts().to_dict()
        
        logger.info(f"Generated statistics for {stats['total_fires']} fires")
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FireDetector()
    
    # Detect fires in Nepal (using proper ISO country code)
    print("Detecting fires in Nepal...")
    nepal_fires = detector.detect_fires("Nepal", hours=48)
    
    print(f"\nFound {nepal_fires['count']} fires in Nepal")
    print(f"Data source: {nepal_fires['source']}")
    print(f"Region code: {nepal_fires['region_code']}")
    
    # Get statistics
    stats = detector.get_statistics(nepal_fires)
    print(f"\nStatistics:")
    print(f"  Total fires: {stats['total_fires']}")
    if 'confidence' in stats:
        print(f"  High confidence: {stats['confidence']['high_confidence_count']}")
        print(f"  Average confidence: {stats['confidence']['mean']}%")
    
    # Convert to DataFrame
    df = detector.get_fires_dataframe(nepal_fires)
    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print(df.head())

import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import time

# ------------------------------
# Configuration
# ------------------------------
# Paths
SHAPEFILE_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
GEE_KEY_PATH = Path("/Users/prabhatrawal/Minor_project_code/keys/gee_project_id.txt")
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/sentinel2_data")

# Date range - Sentinel-2 started in 2015
START_YEAR = 2015  # Sentinel-2A launch year
END_YEAR = 2025

# Cloud masking threshold
CLOUD_PROBABILITY_THRESHOLD = 20  # Pixels with >20% cloud probability will be masked

# Create output directory
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Sentinel-2 L2A Data Extraction from Google Earth Engine")
print("=" * 60)
print("OPTIMIZED VERSION - Processing year by year")
print(f"Date range: {START_YEAR} to {END_YEAR}")

# ------------------------------
# Initialize Google Earth Engine
# ------------------------------
print("\n[1/6] Initializing Google Earth Engine...")
try:
    # Read project ID from file
    with open(GEE_KEY_PATH, 'r') as f:
        project_id = f.read().strip()
    
    # Initialize Earth Engine
    ee.Initialize(project=project_id)
    print(f"✓ GEE initialized with project: {project_id}")
except Exception as e:
    print(f"✗ Error initializing GEE: {e}")
    print("Please run 'earthengine authenticate' in terminal first")
    exit(1)

# ------------------------------
# Load and Process Shapefile
# ------------------------------
print("\n[2/6] Loading shapefile and creating zones...")
gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"✓ Loaded shapefile with {len(gdf)} districts")

# Import zone creation code
import math
from shapely.geometry import box
from shapely.ops import unary_union

union_poly = gdf.geometry.union_all()
minx, miny, maxx, maxy = union_poly.bounds
width = maxx - minx
height = maxy - miny
total_area = gdf.geometry.area.sum()

n_districts = len(gdf)
target_n_total = 10 * n_districts
cell_size = math.sqrt(total_area / target_n_total)

n_cols = math.ceil(width / cell_size)
n_rows = math.ceil(height / cell_size)
dx = width / n_cols
dy = height / n_rows

grid_boxes = []
for i in range(n_cols):
    for j in range(n_rows):
        xmin = minx + i * dx
        xmax = xmin + dx
        ymin = miny + j * dy
        ymax = ymin + dy
        grid_boxes.append(box(xmin, ymin, xmax, ymax))

def assign_and_merge_zones(district_poly, grid_boxes, target_n=10, merge_threshold=0.25):
    zones = []
    for gb in grid_boxes:
        inter = gb.intersection(district_poly)
        if not inter.is_empty:
            zones.append({'geometry': inter})
    
    if not zones:
        return None
    
    zones_gdf = gpd.GeoDataFrame(zones, crs=gdf.crs)
    zones_gdf['area'] = zones_gdf.geometry.area
    
    district_area = district_poly.area
    target_area = district_area / target_n
    threshold_area = merge_threshold * target_area
    
    merged = True
    while merged:
        merged = False
        small_zones = zones_gdf[zones_gdf['area'] < threshold_area].copy()
        if small_zones.empty:
            break
        
        for idx, small_row in small_zones.iterrows():
            if idx not in zones_gdf.index:
                continue
            
            touches = zones_gdf.geometry.touches(small_row.geometry)
            neighbors = zones_gdf[touches & (zones_gdf.index != idx)]
            
            if neighbors.empty:
                continue
            
            smallest_neighbor = neighbors.loc[neighbors['area'].idxmin()]
            smallest_idx = smallest_neighbor.name
            
            merged_geom = unary_union([small_row.geometry, smallest_neighbor.geometry])
            
            zones_gdf.at[smallest_idx, 'geometry'] = merged_geom
            zones_gdf.at[smallest_idx, 'area'] = merged_geom.area
            
            zones_gdf = zones_gdf.drop(idx)
            
            merged = True
            break
    
    zones_gdf = zones_gdf.reset_index(drop=True)
    zones_gdf['zone'] = zones_gdf.index + 1
    
    return zones_gdf

# Process each district
district_zones = {}
for idx, row in gdf.iterrows():
    district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
    polygon = row['geometry']
    zones_gdf = assign_and_merge_zones(polygon, grid_boxes)
    if zones_gdf is not None:
        zones_gdf['district'] = district_name
        district_zones[district_name] = zones_gdf
        print(f"  ✓ {district_name}: {len(zones_gdf)} zones")

# ------------------------------
# Convert geometries to GEE format
# ------------------------------
print("\n[3/6] Converting geometries to GEE format...")

def shapely_to_ee_geometry(shapely_geom, source_crs):
    """Convert Shapely geometry to Earth Engine geometry"""
    # Reproject to WGS84 (EPSG:4326) for GEE
    gdf_temp = gpd.GeoDataFrame([1], geometry=[shapely_geom], crs=source_crs)
    gdf_wgs84 = gdf_temp.to_crs('EPSG:4326')
    geom_wgs84 = gdf_wgs84.geometry.iloc[0]
    
    # Convert to GEE format
    if geom_wgs84.geom_type == 'Polygon':
        coords = [list(geom_wgs84.exterior.coords)]
        return ee.Geometry.Polygon(coords)
    elif geom_wgs84.geom_type == 'MultiPolygon':
        all_coords = []
        for poly in geom_wgs84.geoms:
            all_coords.append(list(poly.exterior.coords))
        return ee.Geometry.MultiPolygon(all_coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_wgs84.geom_type}")

# ------------------------------
# Cloud masking and index calculation functions
# ------------------------------
print("\n[4/6] Setting up Sentinel-2 processing functions...")

def mask_clouds(image):
    """Mask clouds using SCL band and cloud probability"""
    # SCL (Scene Classification Layer) values:
    # 3 = cloud shadows, 8 = cloud medium probability, 9 = cloud high probability, 10 = thin cirrus
    scl = image.select('SCL')
    
    # Create cloud mask
    cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    
    # Also use cloud probability if available
    if 'MSK_CLDPRB' in image.bandNames().getInfo():
        cloud_prob = image.select('MSK_CLDPRB')
        cloud_mask = cloud_mask.And(cloud_prob.lt(CLOUD_PROBABILITY_THRESHOLD))
    
    return image.updateMask(cloud_mask)

def calculate_indices(image):
    """Calculate vegetation and burn indices"""
    # Get bands
    b2 = image.select('B2').divide(10000)  # Blue
    b3 = image.select('B3').divide(10000)  # Green
    b4 = image.select('B4').divide(10000)  # Red
    b8 = image.select('B8').divide(10000)  # NIR
    b11 = image.select('B11').divide(10000)  # SWIR1
    b12 = image.select('B12').divide(10000)  # SWIR2
    
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')
    
    # GNDVI = (NIR - Green) / (NIR + Green)
    gndvi = b8.subtract(b3).divide(b8.add(b3)).rename('GNDVI')
    
    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = b8.subtract(b12).divide(b8.add(b12)).rename('NBR')
    
    # NDWI = (NIR - SWIR1) / (NIR + SWIR1)
    ndwi = b8.subtract(b11).divide(b8.add(b11)).rename('NDWI')
    
    # NDSI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    ndsi = b11.subtract(b12).divide(b11.add(b12)).rename('NDSI')
    
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = b8.subtract(b4).divide(
        b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
    ).multiply(2.5).rename('EVI')
    
    # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L), where L = 0.5
    L = 0.5
    savi = b8.subtract(b4).divide(
        b8.add(b4).add(L)
    ).multiply(1 + L).rename('SAVI')
    
    # Add all indices to the image
    return image.addBands([ndvi, gndvi, nbr, ndwi, ndsi, evi, savi])

print("✓ Cloud masking function ready")
print("✓ Index calculation functions ready:")
print("  - NDVI (Normalized Difference Vegetation Index)")
print("  - GNDVI (Green NDVI)")
print("  - NBR (Normalized Burn Ratio)")
print("  - NDWI (Normalized Difference Water Index)")
print("  - NDSI (Normalized Difference SWIR Index)")
print("  - EVI (Enhanced Vegetation Index)")
print("  - SAVI (Soil-Adjusted Vegetation Index)")

# ------------------------------
# COMPLETELY REWRITTEN extraction function - NO MAPPED FUNCTIONS
# ------------------------------
def extract_zone_data_optimized(zone_geometry, zone_id, district_name, year):
    """Extract Sentinel-2 data - REWRITTEN to avoid .map() issues"""
    try:
        # Convert zone geometry to EE
        ee_geom = shapely_to_ee_geometry(zone_geometry, gdf.crs)
        
        # Date range for this year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31' if year < 2025 else '2025-01-19'
        
        # Load Sentinel-2 L2A collection
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start_date, end_date) \
            .filterBounds(ee_geom) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        
        # Get image count
        count = s2.size().getInfo()
        
        if count == 0:
            return pd.DataFrame()
        
        # Get the list of images (client-side)
        image_list = s2.toList(count).getInfo()
        
        # Process each image individually (client-side loop)
        records = []
        for img_info in image_list:
            try:
                # Reconstruct the image from its ID
                img_id = img_info['id']
                img = ee.Image(img_id)
                
                # Apply cloud masking
                scl = img.select('SCL')
                cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                img_masked = img.updateMask(cloud_mask)
                
                # Calculate indices
                b2 = img_masked.select('B2').divide(10000)
                b3 = img_masked.select('B3').divide(10000)
                b4 = img_masked.select('B4').divide(10000)
                b8 = img_masked.select('B8').divide(10000)
                b11 = img_masked.select('B11').divide(10000)
                b12 = img_masked.select('B12').divide(10000)
                
                ndvi = b8.subtract(b4).divide(b8.add(b4))
                gndvi = b8.subtract(b3).divide(b8.add(b3))
                nbr = b8.subtract(b12).divide(b8.add(b12))
                ndwi = b8.subtract(b11).divide(b8.add(b11))
                ndsi = b11.subtract(b12).divide(b11.add(b12))
                evi = b8.subtract(b4).divide(
                    b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
                ).multiply(2.5)
                L = 0.5
                savi = b8.subtract(b4).divide(b8.add(b4).add(L)).multiply(1 + L)
                
                # Combine all bands
                combined = ee.Image.cat([
                    b2.rename('B2'),
                    b3.rename('B3'),
                    b4.rename('B4'),
                    b8.rename('B8'),
                    b11.rename('B11'),
                    b12.rename('B12'),
                    ndvi.rename('NDVI'),
                    gndvi.rename('GNDVI'),
                    nbr.rename('NBR'),
                    ndwi.rename('NDWI'),
                    ndsi.rename('NDSI'),
                    evi.rename('EVI'),
                    savi.rename('SAVI')
                ])
                
                # Reduce region to get mean values
                stats = combined.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geom,
                    scale=20,
                    maxPixels=1e9
                ).getInfo()
                
                # Get date and cloud cover
                date_millis = img.get('system:time_start').getInfo()
                date_obj = datetime.fromtimestamp(date_millis / 1000)
                date_str = date_obj.strftime('%Y-%m-%d')
                
                cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                
                # Skip if all values are None (completely masked)
                if stats.get('NDVI') is None:
                    continue
                
                # Build record
                records.append({
                    'date': date_str,
                    'zone': zone_id,
                    'district': district_name,
                    'B2_blue_mean': stats.get('B2'),
                    'B3_green_mean': stats.get('B3'),
                    'B4_red_mean': stats.get('B4'),
                    'B8_nir_mean': stats.get('B8'),
                    'B11_swir1_mean': stats.get('B11'),
                    'B12_swir2_mean': stats.get('B12'),
                    'NDVI': round(stats.get('NDVI', 0), 4),
                    'GNDVI': round(stats.get('GNDVI', 0), 4),
                    'NBR': round(stats.get('NBR', 0), 4),
                    'NDWI': round(stats.get('NDWI', 0), 4),
                    'NDSI': round(stats.get('NDSI', 0), 4),
                    'EVI': round(stats.get('EVI', 0), 4),
                    'SAVI': round(stats.get('SAVI', 0), 4),
                    'cloud_cover_percent': round(cloud_cover, 2) if cloud_cover else 0
                })
                
            except Exception as e:
                # Skip this image if there's an error
                continue
        
        if records:
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ✗ Error extracting data for zone {zone_id}, year {year}: {e}")
        return pd.DataFrame()

# ------------------------------
# Process all districts and zones - BY YEAR
# ------------------------------
print("\n[5/6] Extracting Sentinel-2 data for all zones (year by year)...")

for district_name, zones_gdf in district_zones.items():
    print(f"\n{'='*60}")
    print(f"Processing District: {district_name}")
    print(f"{'='*60}")
    
    # Create district folder
    district_folder = OUTPUT_BASE / district_name.replace(' ', '_')
    district_folder.mkdir(exist_ok=True)
    
    # Store all data for this district
    all_district_data = []
    
    # Process each zone
    for idx, zone_row in zones_gdf.iterrows():
        zone_id = zone_row['zone']
        zone_geom = zone_row['geometry']
        
        print(f"\n  Zone {zone_id}/{len(zones_gdf)}:")
        
        zone_all_years_data = []
        
        # Process year by year
        for year in range(START_YEAR, END_YEAR + 1):
            try:
                print(f"    Year {year}...", end=' ')
                year_df = extract_zone_data_optimized(zone_geom, zone_id, district_name, year)
                
                if not year_df.empty:
                    zone_all_years_data.append(year_df)
                    avg_ndvi = year_df['NDVI'].mean()
                    print(f"✓ {len(year_df)} images (avg NDVI: {avg_ndvi:.3f})")
                else:
                    print("✓ No clear images")
                
                time.sleep(0.3)  # Small delay between years
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Combine all years for this zone
        if zone_all_years_data:
            zone_complete_df = pd.concat(zone_all_years_data, ignore_index=True)
            all_district_data.append(zone_complete_df)
            
            # Save individual zone file
            zone_file = district_folder / f"zone_{zone_id}_data.csv"
            zone_complete_df.to_csv(zone_file, index=False)
            
            avg_ndvi = zone_complete_df['NDVI'].mean()
            print(f"  ✓ Zone {zone_id} complete: {len(zone_complete_df)} total images (avg NDVI: {avg_ndvi:.3f})")
    
    # Combine all zones for this district
    if all_district_data:
        district_df = pd.concat(all_district_data, ignore_index=True)
        
        # Save combined district file
        output_file = district_folder / f"{district_name.replace(' ', '_')}_sentinel2_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ District complete: {len(district_df)} total records saved to {output_file}")
        
        # Create summary statistics
        summary = district_df.groupby('zone').agg({
            'NDVI': ['mean', 'min', 'max', 'std'],
            'GNDVI': ['mean', 'min', 'max'],
            'NBR': ['mean', 'min', 'max'],
            'NDWI': ['mean', 'min', 'max'],
            'EVI': ['mean', 'min', 'max'],
            'SAVI': ['mean', 'min', 'max'],
            'cloud_cover_percent': 'mean',
            'date': 'count'
        }).round(4)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_file = district_folder / f"{district_name.replace(' ', '_')}_summary.csv"
        summary.to_csv(summary_file)
        print(f"✓ Summary statistics saved to {summary_file}")
    else:
        print(f"\n⚠ No data extracted for {district_name}")

# ------------------------------
# Summary
# ------------------------------
print("\n" + "="*60)
print("[6/6] EXTRACTION COMPLETE!")
print("="*60)
print(f"\nData saved to: {OUTPUT_BASE}")
print("\nFolder structure:")
print("  sentinel2_data/")
for district_name in district_zones.keys():
    print(f"    ├── {district_name.replace(' ', '_')}/")
    print(f"    │   ├── {district_name.replace(' ', '_')}_sentinel2_data.csv")
    print(f"    │   ├── {district_name.replace(' ', '_')}_summary.csv")
    print(f"    │   └── zone_*.csv")

print("\n" + "="*60)
print("Data Description:")
print("="*60)
print("Bands (mean values):")
print("  - B2_blue_mean, B3_green_mean, B4_red_mean")
print("  - B8_nir_mean, B11_swir1_mean, B12_swir2_mean")
print("\nCalculated Indices:")
print("  - NDVI: Normalized Difference Vegetation Index")
print("  - GNDVI: Green NDVI")
print("  - NBR: Normalized Burn Ratio")
print("  - NDWI: Normalized Difference Water Index")
print("  - NDSI: Normalized Difference SWIR Index")
print("  - EVI: Enhanced Vegetation Index")
print("  - SAVI: Soil-Adjusted Vegetation Index")
print("\nOther:")
print("  - cloud_cover_percent: Cloud coverage of the image")
print("\n✓ All done!")
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import math

# ------------------------------
# Load shapefile
# ------------------------------
shapefile_path = r"C:\Users\NCE\OneDrive\Desktop\forest-fire-detection-Prabhat\polygon_file\actual_timezone_designated_district_using_EPSG_32644.shp"
gdf = gpd.read_file(shapefile_path)

# ------------------------------
# Function to split polygon into approximately n zones in a grid (chessboard style with clipping to boundary)
# ------------------------------
def split_polygon_into_grid(polygon, n_zones=10):
    """
    Splits a polygon into a grid of approximately square boxes, clipped to the polygon boundary
    to ensure zones match the district outline (like terraces). Generates roughly n_zones, but
    may be slightly more or less to cover the entire area with square-like cells.
    """
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    
    # Approximate cell size for square cells (in CRS units, which are meters for EPSG:32644)
    cell_size = math.sqrt(width * height / n_zones)
    
    # Number of columns and rows to fit square cells
    n_cols = math.ceil(width / cell_size)
    n_rows = math.ceil(height / cell_size)
    
    dx = width / n_cols
    dy = height / n_rows
    
    zones = []
    for i in range(n_cols):
        for j in range(n_rows):
            xmin = minx + i * dx
            xmax = xmin + dx
            ymin = miny + j * dy
            ymax = ymin + dy
            z = box(xmin, ymin, xmax, ymax)
            zone_poly = z.intersection(polygon)
            if not zone_poly.is_empty:
                zones.append(zone_poly)
    
    return zones

# ------------------------------
# Create GeoDataFrame for zones
# ------------------------------
zones_list = []
for idx, row in gdf.iterrows():
    district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
    polygon = row['geometry']
    zones = split_polygon_into_grid(polygon, 10)
    for zone_idx, z in enumerate(zones):
        zones_list.append({'district': district_name, 'zone': zone_idx + 1, 'geometry': z})

zones_gdf = gpd.GeoDataFrame(zones_list, crs=gdf.crs)

# ------------------------------
# Compute centroids (centers) of zones
# ------------------------------
zones_gdf['centroid'] = zones_gdf.geometry.centroid

# Print center coordinates (these can be used directly in GEE for pulling satellite data)
print("Center coordinates for each zone (format: district - zone: (x, y)):")
for idx, row in zones_gdf.iterrows():
    cent = row['centroid']
    print(f"{row['district']} - Zone {row['zone']}: ({cent.x}, {cent.y})")

# ------------------------------
# Plot for all districts combined (with zones and centers)
# ------------------------------
fig, ax = plt.subplots(figsize=(12, 12))
gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, label='District Boundaries')
zones_gdf.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
zones_gdf.centroid.plot(ax=ax, color='red', markersize=50, label='Zone Centers')
plt.title("All Districts with Clipped Zones and Centers")
plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.legend()
plt.show()

# ------------------------------
# Plot individual districts (one plot per district with zones and centers)
# ------------------------------
if 'NAME' in gdf.columns:
    for district in gdf['NAME'].unique():
        district_poly = gdf[gdf['NAME'] == district]
        district_zones = zones_gdf[zones_gdf['district'] == district]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        district_poly.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)
        district_zones.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
        district_zones.centroid.plot(ax=ax, color='red', markersize=50)
        plt.title(f"{district} with Clipped Zones and Centers")
        plt.xlabel("Easting (meters)")
        plt.ylabel("Northing (meters)")
        plt.show()
else:
    print("No 'NAME' column in shapefile; skipping individual district plots.")

# Optional: Save zones and centers to a new shapefile for use in GEE or other tools
# zones_gdf.to_file("district_zones_with_centers.shp")
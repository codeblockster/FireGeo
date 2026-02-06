import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import math
from shapely.ops import unary_union

# ------------------------------
# Load shapefile
# ------------------------------
shapefile_path = r"C:\Users\NCE\OneDrive\Desktop\forest-fire-detection-Prabhat\polygon_file\actual_timezone_designated_district_using_EPSG_32644.shp"
gdf = gpd.read_file(shapefile_path)

# ------------------------------
# Function to split polygon into grid zones and merge small ones
# ------------------------------
def split_and_merge_zones(polygon, n_zones=10, merge_threshold=0.25):
    """
    Splits a polygon into a grid of approximately square boxes, clipped to the boundary.
    Then merges zones smaller than merge_threshold * (target_area) into the smallest neighboring zone.
    Iterates until no small zones remain or no merges possible.
    """
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    
    # Approximate cell size for square cells
    cell_size = math.sqrt(width * height / n_zones)
    
    # Number of columns and rows
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
                zones.append({'geometry': zone_poly, 'original_idx': len(zones) + 1})
    
    # Create initial GeoDataFrame for zones
    zones_gdf = gpd.GeoDataFrame(zones, crs=gdf.crs)
    zones_gdf['area'] = zones_gdf.geometry.area
    
    target_area = polygon.area / n_zones
    threshold_area = merge_threshold * target_area
    
    # Merge loop
    merged = True
    while merged:
        merged = False
        small_zones = zones_gdf[zones_gdf['area'] < threshold_area].copy()
        if small_zones.empty:
            break
        
        for idx, small_row in small_zones.iterrows():
            if idx not in zones_gdf.index:
                continue  # Already merged
            
            # Find touching neighbors
            touches = zones_gdf.geometry.touches(small_row.geometry)
            neighbors = zones_gdf[touches & (zones_gdf.index != idx)]
            
            if neighbors.empty:
                continue  # No neighbors, can't merge
            
            # Select the neighbor with the smallest area
            smallest_neighbor = neighbors.loc[neighbors['area'].idxmin()]
            smallest_idx = smallest_neighbor.name
            
            # Merge geometries
            merged_geom = unary_union([small_row.geometry, smallest_neighbor.geometry])
            
            # Update the smallest neighbor
            zones_gdf.at[smallest_idx, 'geometry'] = merged_geom
            zones_gdf.at[smallest_idx, 'area'] = merged_geom.area
            
            # Remove the small zone
            zones_gdf = zones_gdf.drop(idx)
            
            merged = True
            break  # Restart loop after each merge to recalculate
    
    # Renumber zones
    zones_gdf = zones_gdf.reset_index(drop=True)
    zones_gdf['zone'] = zones_gdf.index + 1
    
    return zones_gdf

# ------------------------------
# Process each district
# ------------------------------
district_zones = {}
print("Center coordinates for each zone (format: district - zone: (x, y)):")
for idx, row in gdf.iterrows():
    district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
    polygon = row['geometry']
    zones_gdf = split_and_merge_zones(polygon, 10, merge_threshold=0.25)
    zones_gdf['district'] = district_name
    district_zones[district_name] = zones_gdf
    
    # Compute centroids
    zones_gdf['centroid'] = zones_gdf.geometry.centroid

    # # Print centers
    # for z_idx, z_row in zones_gdf.iterrows():
    #     cent = z_row['centroid']
    #     print(f"{district_name} - Zone {z_row['zone']}: ({cent.x}, {cent.y})")

# ------------------------------
# Combine all zones for plotting
# ------------------------------
all_zones_gdf = gpd.pd.concat(district_zones.values(), ignore_index=True)

# Plot all districts combined
fig, ax = plt.subplots(figsize=(12, 12))
gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, label='District Boundaries')
all_zones_gdf.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
#all_zones_gdf.centroid.plot(ax=ax, color='red', markersize=50, label='Zone Centers')
plt.title("All Districts with Merged Clipped Zones and Centers")
plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.legend()
plt.show()

# Plot individual districts
if 'NAME' in gdf.columns:
    for district, zones_gdf in district_zones.items():
        district_poly = gdf[gdf['NAME'] == district]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        district_poly.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)
        zones_gdf.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
       #zones_gdf.centroid.plot(ax=ax, color='red', markersize=50)
        plt.title(f"{district} with Merged Clipped Zones and Centers")
        plt.xlabel("Easting (meters)")
        plt.ylabel("Northing (meters)")
        plt.show()
else:
    print("No 'NAME' column in shapefile; skipping individual district plots.")

# Optional: Save to shapefile
# all_zones_gdf.to_file("district_zones_merged_with_centers.shp")
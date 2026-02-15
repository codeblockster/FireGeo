1. Modis 
Key Features of MODIS LST Extraction Code
 a.⁠ ⁠Intelligent Zone-Based Processing

Automatically divides each district into ~10 optimal zones using a grid-based approach
Implements adaptive zone merging to handle small zones (25% threshold of target area)
Ensures balanced spatial coverage across irregular district boundaries

 b.⁠ ⁠Parallel Processing Architecture

Concurrent zone processing using ThreadPoolExecutor with 4 workefirs
Simultaneous data extraction across multiple zones
Real-time progress tracking with completion counters

 c.⁠ ⁠Optimized Batch Processing

Processes 5-year batches instead of individual years
Reduces API calls and improves memory efficiency
Handles both Terra and Aqua satellites in single operations

 d.⁠ ⁠Dual-Satellite Data Integration

Extracts data from both MODIS Terra (MOD11A1) and Aqua (MYD11A1)
Combines observations from both satellites for comprehensive coverage
Maintains satellite attribution in output data

 e.⁠ ⁠Advanced Quality Control

Automatic quality masking using QC_Day and QC_Night bands
Filters for highest quality observations (QC bits 0-1 = 0)
Includes clear sky coverage metrics for day and night

 f.⁠ ⁠Temperature Processing

Converts Kelvin to Celsius (scale factor 0.02, then -273.15)
Separate processing for daytime and nighttime LST
Returns mean temperature statistics per zone

 g.⁠ ⁠Robust Error Handling

Try-catch blocks at multiple levels (zone, batch, district)
Graceful handling of missing data or API failures
Continues processing even if individual zones fail

 h.⁠ ⁠Comprehensive Output Structure

Individual CSV files per zone for granular analysis
Combined district-level datasets
Statistical summaries (mean, min, max, std) by zone and satellite
Metadata including clear coverage percentages

 i.⁠ ⁠Efficient Geometry Management

Automatic CRS conversion (local projection to WGS84 for GEE)
Handles both Polygon and MultiPolygon geometries
Uses Shapely for efficient spatial operations

j.⁠ ⁠Scalable Configuration

Customizable date ranges (2000-2025)
Adjustable batch sizes and worker counts
Organized output directory structure per district

k.⁠ ⁠Memory Optimization

Single API calls to fetch all features per batch
Incremental data aggregation instead of loading everything at once
Minimal rate limiting (50ms) while respecting GEE quotas

l.⁠ ⁠Data Quality Metrics

Records count per zone and satellite
Average day/night temperatures displayed during processing
Summary statistics for validation and quality assessment




---------------------------------------------------------------------------------------------------------------------------------------




2. MOD14A1
Key Features of MOD14A1 Fire Mask Extraction Code
 1.⁠ ⁠Fire Detection Data Extraction

Extracts MODIS MOD14A1 Thermal Anomalies/Fire product (1km resolution)
Processes FireMask band with three confidence levels:

Low confidence (value 7)
Nominal confidence (value 8)
High confidence (value 9)



 2.⁠ ⁠Year-by-Year Processing Strategy

Processes data incrementally by individual years (2000-2025)
Prevents GEE timeout errors on large date ranges
Allows recovery from failures without losing all progress
Real-time feedback per year showing fire detection results

 3.⁠ ⁠Multi-Confidence Fire Classification

Separate pixel counts for each confidence level
Aggregates total fire pixels across all confidence categories
Enables filtering by fire detection certainty in downstream analysis

 4.⁠ ⁠Server-Side Optimization

Uses GEE's server-side mapping (map() function) to process all images at once
Single reduceRegion() call per image instead of multiple operations
Batch retrieval of all features using toList().getInfo()
Minimizes client-server communication overhead

 5.⁠ ⁠Comprehensive Fire Metrics

Pixel counts: Low, nominal, and high confidence fire pixels
Total fire pixels: Sum of all confidence levels
Total pixels: Complete pixel count in zone for normalization
Fire percentage: Proportion of zone affected by fire

 6.⁠ ⁠Zone-Based Spatial Analysis

Same intelligent zone division system (~10 zones per district)
Adaptive zone merging for spatial balance
Enables fine-grained spatial patterns in fire occurrence

 7.⁠ ⁠Temporal Coverage

Processes 25+ years of fire data (2000-2025)
Daily temporal resolution
Handles leap years and partial year (2025 up to Jan 19)

 8.⁠ ⁠Progressive Data Storage

Saves data incrementally after each year completes
Zone-level CSV files for granular analysis
District-level combined datasets
Summary statistics with aggregated fire metrics

 9.⁠ ⁠Fire Event Counting

Tracks total observation days per zone/year
Identifies days with detected fire activity
Provides fire frequency statistics in console output

10.⁠ ⁠Error Resilience

Try-catch blocks at year and zone levels
Continues processing if individual years fail
Small delays (0.2s) between years to respect API limits
Returns empty DataFrames on errors rather than crashing

11.⁠ ⁠Statistical Summaries

Per-zone aggregations: sum, mean, and max fire pixels
Breakdown by confidence level
Observation count per zone
Easily identifies fire hotspots within districts

12.⁠ ⁠Output Organization
mod14a1_data/
├── District_Name/
│   ├── District_Name_mod14a1_data.csv      # All zones combined
│   ├── District_Name_summary.csv            # Statistical summary
│   └── zone_1_data.csv, zone_2_data.csv... # Individual zones
13.⁠ ⁠Fire Percentage Calculation

Normalizes fire pixels by total pixels in zone
Accounts for varying zone sizes
Enables direct comparison between zones
Rounded to 4 decimal places for precision

14.⁠ ⁠Data Quality Tracking

Records total pixels analyzed per observation
Distinguishes between "no fire" and "no data"
Maintains data provenance with date and zone IDs

15.⁠ ⁠Scalable Architecture

Configurable year range via START_YEAR/END_YEAR
Works with any shapefile with district geometries
Adaptable zone count (target_n parameter)
Platform-agnostic path handling with pathlib

Performance Advantages Over Sequential Processing:

Server-side batch processing: 10-20x faster than image-by-image iteration
Year-based chunking: Prevents memory/timeout issues
Single API calls: Retrieves all features per year at once
Incremental saves: Data preserved even if script interrupts






---------------------------------------------------------------------------------------------------------------------------------------





3. Sentinel - La2

# Key Features of Sentinel-2 L2A Data Extraction Code

## 1. *Multi-Spectral Band Extraction*
•⁠  ⁠Extracts 6 key spectral bands from Sentinel-2 L2A (Surface Reflectance):
  - *B2* (Blue, 490nm)
  - *B3* (Green, 560nm)
  - *B4* (Red, 665nm)
  - *B8* (NIR, 842nm)
  - *B11* (SWIR1, 1610nm)
  - *B12* (SWIR2, 2190nm)
•⁠  ⁠All bands scaled to reflectance values (0-1) by dividing by 10,000

## 2. *Comprehensive Vegetation Indices*
•⁠  ⁠*NDVI* (Normalized Difference Vegetation Index): General vegetation health
•⁠  ⁠*GNDVI* (Green NDVI): Chlorophyll-sensitive vegetation index
•⁠  ⁠*EVI* (Enhanced Vegetation Index): Improved sensitivity in high biomass regions
•⁠  ⁠*SAVI* (Soil-Adjusted Vegetation Index): Minimizes soil brightness influence (L=0.5)

## 3. *Specialized Environmental Indices*
•⁠  ⁠*NBR* (Normalized Burn Ratio): Fire damage and burn severity detection
•⁠  ⁠*NDWI* (Normalized Difference Water Index): Water content and moisture stress
•⁠  ⁠*NDSI* (Normalized Difference SWIR Index): Additional moisture/vegetation analysis

## 4. *Advanced Cloud Masking*
•⁠  ⁠Uses *SCL* (Scene Classification Layer) band for intelligent cloud filtering
•⁠  ⁠Masks cloud shadows (value 3), medium probability clouds (8), high probability clouds (9), and thin cirrus (10)
•⁠  ⁠Optional integration with cloud probability band (MSK_CLDPRB) with 20% threshold
•⁠  ⁠Pre-filters images with >80% cloud coverage to reduce processing

## 5. *Client-Side Processing Architecture*
•⁠  ⁠Completely rewritten to avoid GEE server-side ⁠ .map() ⁠ limitations
•⁠  ⁠Processes images individually in Python loop rather than server-side batch
•⁠  ⁠Reconstructs images from IDs for granular control
•⁠  ⁠Better error handling and debugging capabilities

## 6. *Year-by-Year Progressive Processing*
•⁠  ⁠Processes data from 2015 (Sentinel-2A launch) through 2025
•⁠  ⁠Individual year processing prevents timeout errors
•⁠  ⁠Incremental data saves preserve progress
•⁠  ⁠Real-time feedback showing image counts and NDVI averages per year

## 7. *High Spatial Resolution*
•⁠  ⁠Uses 20-meter scale for reduceRegion operations
•⁠  ⁠Balances detail with processing speed
•⁠  ⁠Appropriate for agricultural and land use monitoring

## 8. *Quality Control Metrics*
•⁠  ⁠Records cloud cover percentage for each observation
•⁠  ⁠Filters completely masked images (where all pixels are clouds)
•⁠  ⁠Enables post-processing quality filtering based on cloud coverage
•⁠  ⁠Skips images with no valid pixels after masking

## 9. *Robust Error Handling*
•⁠  ⁠Try-catch at multiple levels: image, year, and zone
•⁠  ⁠Gracefully skips problematic individual images
•⁠  ⁠Continues processing even if some images fail
•⁠  ⁠Returns empty DataFrames rather than crashing

## 10. *Comprehensive Output Data*
⁠ csv
date, zone, district, 
B2_blue_mean, B3_green_mean, B4_red_mean, 
B8_nir_mean, B11_swir1_mean, B12_swir2_mean,
NDVI, GNDVI, NBR, NDWI, NDSI, EVI, SAVI,
cloud_cover_percent
 ⁠

## 11. *Statistical Summaries*
•⁠  ⁠Per-zone aggregations for all 7 indices
•⁠  ⁠Mean, min, max, and standard deviation
•⁠  ⁠Average cloud cover per zone
•⁠  ⁠Observation count for temporal coverage assessment

## 12. *Zone-Based Spatial Analysis*
•⁠  ⁠Same intelligent ~10 zones per district system
•⁠  ⁠Enables detection of spatial heterogeneity in vegetation
•⁠  ⁠Adaptive zone merging for balanced areas
•⁠  ⁠Individual zone files for focused analysis

## 13. *Temporal Coverage Tracking*
•⁠  ⁠Counts clear images per year per zone
•⁠  ⁠Displays average NDVI during processing for quick quality check
•⁠  ⁠Helps identify data gaps due to persistent cloud cover
•⁠  ⁠Date sorting ensures chronological order

## 14. *Harmonized Data Processing*
•⁠  ⁠Uses *S2_SR_HARMONIZED* collection for consistency
•⁠  ⁠Ensures compatibility between Sentinel-2A and 2B data
•⁠  ⁠Corrects for radiometric differences between sensors

## 15. *Progressive Data Storage*

sentinel2_data/
├── District_Name/
│   ├── District_Name_sentinel2_data.csv    # All zones combined
│   ├── District_Name_summary.csv           # Statistical summary
│   └── zone_1_data.csv, zone_2_data.csv... # Individual zones


## 16. *Index Value Precision*
•⁠  ⁠All indices rounded to 4 decimal places
•⁠  ⁠Balances precision with file size
•⁠  ⁠Sufficient accuracy for most applications
•⁠  ⁠Spectral bands preserved at full precision

## 17. *Delay Management*
•⁠  ⁠0.3 second delay between years
•⁠  ⁠Respects GEE rate limits
•⁠  ⁠Prevents quota exhaustion
•⁠  ⁠Allows long-running extractions to complete successfully

## Application-Specific Advantages:

### *Agricultural Monitoring*
•⁠  ⁠NDVI, GNDVI, EVI for crop health tracking
•⁠  ⁠SAVI reduces soil background interference
•⁠  ⁠High temporal resolution (5-day revisit)

### *Fire Analysis*
•⁠  ⁠NBR for pre/post-fire comparison
•⁠  ⁠Burn severity assessment
•⁠  ⁠Recovery monitoring over time

### *Water Resource Management*
•⁠  ⁠NDWI for irrigation assessment
•⁠  ⁠Drought monitoring
•⁠  ⁠Wetland mapping

### *Land Use Change Detection*
•⁠  ⁠Multiple indices enable classification
•⁠  ⁠Temporal series reveals trends
•⁠  ⁠Cloud-filtered ensures reliable baselines

## Performance Characteristics:
•⁠  ⁠*Client-side control*: Better debugging and error recovery
•⁠  ⁠*Incremental processing*: Data preserved despite interruptions
•⁠  ⁠*Quality filtering*: Cloud masking ensures usable observations
•⁠  ⁠*Multi-index output*: Comprehensive analysis from single extraction







---------------------------------------------------------------------------------------------------------------------------------------





4. Srtm

# Key Features of SRTM Elevation Data Extraction Code

## 1. **Static Terrain Dataset**
- Extracts from SRTM (Shuttle Radar Topography Mission) - **single-time acquisition** (year 2000)
- No temporal dimension - elevation is static/unchanging
- **No year-by-year processing needed** - one extraction per zone covers all time periods
- Efficient single-pass data collection

## 2. **Dual SRTM Dataset Integration**
- **USGS/SRTMGL1_003**: Standard elevation model (30m resolution)
- **CSP/ERGo SRTM_mTPI**: Multi-scale Topographic Position Index
- Combines absolute elevation with relative topographic context

## 3. **Comprehensive Elevation Statistics**
- **Mean elevation**: Average height above sea level
- **Min/Max elevation**: Absolute range of terrain
- **Median elevation**: Central tendency (robust to outliers)
- **Standard deviation**: Terrain variability measure
- **Elevation range**: Total relief (max - min)

## 4. **Elevation Percentiles**
- **P10, P25, P75, P90**: Quartile and decile analysis
- Enables understanding of elevation distribution
- Identifies lowlands vs highlands proportions
- More robust than simple min/max for terrain characterization

## 5. **Multi-scale Topographic Position Index (mTPI)**
- Quantifies relative position in landscape
- **Positive values**: Ridges, hilltops, peaks
- **Negative values**: Valleys, depressions, lowlands
- **Near zero**: Flat areas or mid-slopes
- Mean, min, max, and standard deviation provided

## 6. **Advanced Terrain Derivatives**
- **Slope**: Steepness of terrain in degrees
  - Mean, min, max, standard deviation
  - Critical for erosion, agriculture, development analysis
- **Aspect**: Compass direction of slope (0-360°)
  - Mean and standard deviation
  - Important for solar exposure, microclimate

## 7. **Server-Side Combined Reducers**
- Uses `ee.Reducer.combine()` to calculate multiple statistics in one call
- Single API request per metric type (elevation, mTPI, terrain)
- Significantly faster than multiple separate requests
- Efficient chaining of mean, min, max, stdDev, median reducers

## 8. **High-Resolution Processing**
- 30-meter native SRTM resolution maintained
- Appropriate scale for regional topographic analysis
- Balances detail with processing efficiency
- Industry-standard DEM resolution

## 9. **Zone-Based Topographic Profiling**
- Same ~10 zones per district system
- Captures intra-district topographic variation
- Enables identification of topographic sub-regions
- Each zone gets complete statistical profile

## 10. **Hierarchical Data Organization**
```
srtm_data/
├── all_districts_srtm_data.csv          # All zones combined
├── district_summary.csv                  # District-level aggregations
└── District_Name/
    └── District_Name_srtm_data.csv      # All zones for district
```

## 11. **Rich Metadata Output**
Single row per zone contains:
- 6 basic elevation statistics
- 4 elevation percentiles  
- 4 mTPI metrics
- 8 slope/aspect statistics
- **Total: 22+ terrain characteristics per zone**

## 12. **District-Level Summaries**
- Aggregates elevation across all zones per district
- Mean elevation, range statistics
- Average slope characteristics
- Zone count for validation
- Quick overview of district topography

## 13. **Terrain Variability Metrics**
- Standard deviation of elevation = terrain roughness
- Standard deviation of slope = terrain complexity
- Standard deviation of aspect = directional variability
- Quantifies landscape heterogeneity

## 14. **Error Handling Without Temporal Complexity**
- Simple try-catch per zone
- No year loops = fewer failure points
- Missing zones handled gracefully
- Small 0.1s delay for rate limiting

## 15. **Interpretive Console Output**
```
Zone 1/10... ✓ Elev: 245.3m (range: 180-320m), Slope: 12.5°
```
- Immediate visual feedback on terrain characteristics
- Range shows topographic relief
- Slope indicates terrain difficulty

## 16. **Static Reference Dataset**
- Can be joined with temporal datasets (LST, MODIS, Sentinel-2)
- Elevation as explanatory variable for temperature patterns
- Slope/aspect influence on vegetation indices
- Topography controls fire spread patterns

## Applications in Multi-Dataset Analysis:

### **Temperature Modeling**
- Elevation correlates with LST (lapse rate ~6.5°C/km)
- Aspect affects solar heating
- Slope influences local climate

### **Fire Risk Assessment**
- Slope affects fire spread rate
- Aspect determines fuel moisture
- Topographic position influences wind patterns
- Combine with MOD14A1 fire data

### **Vegetation Analysis**
- Elevation zones = ecological boundaries
- Aspect affects vegetation types
- Slope limits agricultural use
- Link with Sentinel-2 NDVI patterns

### **Hydrological Context**
- Negative mTPI = water accumulation areas
- Slope affects runoff and erosion
- Aspect influences snowmelt timing

## Performance Advantages:

- **No temporal iteration**: 100x faster than temporal datasets
- **Single extraction per zone**: Complete terrain profile in one API call
- **Combined reducers**: 3 API calls instead of 15+
- **Static data**: Results cacheable and reusable

## Data Completeness:
- **Global coverage**: SRTM covers 80% of Earth's land
- **No missing dates**: Static dataset always available
- **No cloud issues**: Radar data unaffected by weather
- **Consistent quality**: Single acquisition campaign












---------------------------------------------------------------------------------------------------------------------------------------










6. Usgs - landsat 8 and 9

# Key Features of Landsat 8/9 Data Extraction Code

## 1. **Dual-Satellite Integration**
- **Landsat 8 (LC08)**: Historical data (April 2013 - 2021)
- **Landsat 9 (LC09)**: Real-time data (October 2021 - present)
- Intelligent satellite selection based on date range
- Seamless merging during transition period (2021)
- Maintains continuity across 12+ years

## 2. **4-Batch Parallel Processing System**
- Divides 13-year period into **4 batches** of 3 years each:
  - Batch 1: 2013-2015
  - Batch 2: 2016-2018
  - Batch 3: 2019-2021
  - Batch 4: 2022-2025
- **MAX_WORKERS = 4**: All batches process simultaneously
- **ZONE_BATCH_SIZE = 4**: Processes 4 zones in parallel
- Dramatic speed increase over sequential processing

## 3. **Resume Capability**
- **RESUME_MODE = True**: Skip already-processed zones
- Checks for existing CSV files before processing
- Enables interruption and restart without data loss
- Critical for long-running extractions
- Validates existing files before skipping

## 4. **Collection 2 Surface Reflectance**
- Uses **LANDSAT/LC08/C02/T1_L2** (Tier 1, Level 2)
- Uses **LANDSAT/LC09/C02/T1_L2** (Tier 1, Level 2)
- Atmospherically corrected surface reflectance
- Higher quality than raw Top-of-Atmosphere (TOA) data
- Consistent preprocessing across all images

## 5. **Advanced Cloud Masking**
- Uses **QA_PIXEL** band for quality assessment
- Masks cloud shadows (bit 3) and clouds (bit 4)
- Bitwise operations for precise masking
- Pre-filters images with >20% cloud cover
- Reduces processing time and improves data quality

## 6. **Landsat-Specific Band Mapping**
Translates Landsat bands to Sentinel-2 equivalents:
- **SR_B2** → Blue (B2)
- **SR_B3** → Green (B3)
- **SR_B4** → Red (B4)
- **SR_B5** → NIR (B8 equivalent)
- **SR_B6** → SWIR1 (B11 equivalent)
- **SR_B7** → SWIR2 (B12 equivalent)

## 7. **Collection 2 Reflectance Scaling**
- Applies **scale factor: 0.0000275**
- Applies **offset: -0.2**
- Converts DN (Digital Numbers) to surface reflectance (0-1)
- Matches Sentinel-2 reflectance format
- Essential for accurate index calculation

## 8. **Seven Vegetation/Environmental Indices**
- **NDVI**: (NIR - Red) / (NIR + Red) - vegetation health
- **GNDVI**: (NIR - Green) / (NIR + Green) - chlorophyll content
- **NBR**: (NIR - SWIR1) / (NIR + SWIR1) - burn severity
- **NDWI**: (Green - NIR) / (Green + NIR) - water content
- **NDSI**: (SWIR1 - SWIR2) / (SWIR1 + SWIR2) - moisture index
- **EVI**: 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
- **SAVI**: ((NIR - Red) / (NIR + Red + 0.5)) × 1.5 - soil-adjusted

## 9. **30-Meter Resolution Processing**
- Landsat's native 30m resolution (vs Sentinel-2's 10-20m)
- **scale=30** in reduceRegion operations
- Appropriate for regional and landscape-scale analysis
- Longer temporal record compensates for lower spatial resolution

## 10. **Comprehensive Output Structure**
```csv
date, zone, district, satellite, cloud_cover,
B2_Blue, B3_Green, B4_Red, B5_NIR, B6_SWIR1, B7_SWIR2,
NDVI, GNDVI, NBR, NDWI, NDSI, EVI, SAVI
```
- Raw spectral bands preserved
- All 7 indices included
- Satellite attribution for data provenance
- Cloud cover for quality filtering

## 11. **Satellite Transition Logic**
```python
if batch_end < 2021:      # Landsat 8 only
if batch_start >= 2021:   # Landsat 8/9 merged
else:                      # Transition period
```
- Handles satellite changeover intelligently
- Ensures no data gaps during transition
- Maintains consistent data format

## 12. **Parallel Batch Processing Architecture**
```python
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Process 4 batches × 4 zones = 16 parallel operations
```
- **4 year-batches** processed simultaneously
- **4 zones** processed per batch
- **16 concurrent API calls** at peak
- Maximizes GEE API throughput

## 13. **Cloud Cover Filtering**
- Pre-filter: `CLOUD_COVER < 20%`
- Records cloud cover percentage in output
- Enables post-processing quality control
- Reduces wasted processing on cloudy images
- Configurable threshold (CLOUD_COVER_MAX)

## 14. **Error Handling at Multiple Levels**
- Batch-level try-catch (continues if one batch fails)
- Zone-level try-catch (processes other zones)
- Feature-level validation (skips invalid data)
- Graceful degradation without crashing

## 15. **Progress Tracking and Feedback**
```
✓ Zone 3 [12/40]: 127 images (avg NDVI: 0.654)
⏭ Zone 5 [13/40]: Skipped (exists)
⚠ Zone 7 [14/40], years 2016-2018: Timeout
```
- Real-time progress with completion ratio
- Skip indicators for resumed zones
- Error messages with specific batch info
- NDVI averages for quick quality checks

## 16. **Future Warning Suppression**
```python
warnings.filterwarnings('ignore', category=FutureWarning)
```
- Suppresses pandas concat warnings
- Cleaner console output
- Focuses on actionable messages

## 17. **Hierarchical Data Organization**
```
landsat_data/
├── District_Name/
│   ├── zone_1_landsat_data.csv
│   ├── zone_2_landsat_data.csv
│   └── District_Name_landsat_data.csv (combined)
└── District_Name_summary.csv
```

## 18. **Statistical Summary Generation**
- Zone-level aggregations
- Mean, min, max, std for all indices
- Satellite-specific statistics
- Observation counts for coverage assessment

## 19. **Best Effort Processing**
- `bestEffort=True` in reduceRegion
- Handles large geometries gracefully
- Prevents timeout on complex zones
- Adapts scale if needed

## 20. **Temporal Coverage**
- **13 years** of continuous data (2013-2025)
- **16-day revisit** time (8 days with both satellites)
- Significantly longer record than Sentinel-2 (2015+)
- Enables long-term trend analysis

## Advantages Over Sentinel-2:

### **Temporal Depth**
- **3 extra years** of historical data (2013-2015)
- Critical for establishing baselines
- Better for climate trend analysis

### **Consistency**
- Single sensor family (OLI/OLI-2)
- Collection 2 harmonization across satellites
- Minimal calibration differences

### **Complementarity**
- Can fill Sentinel-2 cloud gaps
- Different orbit paths reduce revisit time
- Combined use improves temporal resolution

## Performance Characteristics:

- **4×3-year batches**: Reduces API calls by 75% vs year-by-year
- **Parallel zones**: 4× speedup within district
- **Resume mode**: Zero wasted computation on reruns
- **Cloud pre-filter**: ~50% reduction in processed images

## Application Advantages:

### **Agricultural Monitoring**
- 13-year crop history
- Inter-annual variability analysis
- Long-term productivity trends

### **Land Use Change**
- Detect gradual transitions (forest to agriculture)
- Quantify urbanization rates
- Track ecosystem degradation

### **Fire Recovery**
- Pre/post-fire comparisons using NBR
- Multi-year recovery trajectories
- Burn severity classification

### **Climate Impact Studies**
- Drought progression (NDWI, NDVI trends)
- Vegetation response to temperature (link with LST)
- Phenology shifts over decade

This code provides **13 years of harmonized multispectral satellite data** with sophisticated processing, parallel execution, and resume capability—optimized for large-scale, long-term environmental monitoring and analysis.









--------------------------------------------------------------------------------------------------------------------------------------





6. Era-5 
# Key Features of ERA5-Land Climate Reanalysis Data Extraction Code

## 1. **Global Climate Reanalysis Dataset**
- **ERA5-Land**: ECMWF's high-resolution land surface reanalysis
- **Daily aggregated data** from 1950 to near-real-time
- Combines modeling with observations for gap-free coverage
- **No cloud contamination** - model-based data always available
- Processes 2000-2025 (25+ years of climate data)

## 2. **Comprehensive Meteorological Variables**
### **Temperature Suite (4 variables)**
- **temperature_2m**: Air temperature at 2 meters height
- **dewpoint_temperature_2m**: Moisture saturation point
- **skin_temperature**: Land surface temperature (LST equivalent)
- **soil_temperature_level_1**: Top soil layer temperature (~0-7cm)

### **Moisture & Precipitation**
- **volumetric_soil_water_layer_1**: Soil moisture content (m³/m³)
- **total_precipitation_sum**: Daily accumulated precipitation

### **Wind Components**
- **u_component_of_wind_10m**: Eastward wind at 10m
- **v_component_of_wind_10m**: Northward wind at 10m

### **Atmospheric Pressure**
- **surface_pressure**: Barometric pressure at surface

## 3. **Derived Meteorological Variables**
Calculates four critical fire risk and ecological indicators:

### **Wind Speed & Direction**
```python
wind_speed = sqrt(u² + v²)
wind_direction = atan2(u, v) × 180/π + 180
```
- Magnitude and azimuth from vector components
- Direction in meteorological convention (0-360°)

### **Relative Humidity**
```python
RH = 100 × exp(17.625×Td / (243.04 + Td)) / exp(17.625×T / (243.04 + T))
```
- Magnus formula approximation
- Requires temperature and dewpoint in Celsius
- Critical for fire behavior modeling

### **Vapor Pressure Deficit (VPD)**
```python
VPD = es - ea
```
- Difference between saturation and actual vapor pressure
- **Key fire risk indicator**: High VPD = dry, fire-prone conditions
- Important for plant stress assessment
- Measured in kilopascals (kPa)

## 4. **Intelligent Unit Conversions**
- **Temperature**: Kelvin → Celsius (-273.15)
- **Precipitation**: Meters → Millimeters (×1000)
- **Pressure**: Pascals → Hectopascals (÷100)
- All conversions applied automatically
- Output in user-friendly units

## 5. **Year-by-Year Processing Strategy**
- Processes 25 years individually (2000-2025)
- Prevents timeout on large date ranges
- Enables progress monitoring per year
- Same proven architecture as MODIS/Sentinel-2 codes

## 6. **~11km Spatial Resolution**
- **scale=11132**: Native ERA5-Land resolution
- Coarser than satellite data but consistent globally
- Appropriate for regional climate analysis
- No gaps from clouds or sensor failures

## 7. **Server-Side Batch Processing**
- Maps derived variable calculation over entire collection
- Single `reduceRegion()` call per image
- Retrieves all features in one `toList().getInfo()` operation
- Minimizes client-server round trips

## 8. **Daily Temporal Resolution**
- One value per day per zone
- Aggregated from hourly ERA5-Land data
- Consistent temporal coverage (no missing days)
- 365-366 records per year per zone

## 9. **Comprehensive Output Structure**
```csv
date, zone, district,
temperature_2m_celsius, dewpoint_2m_celsius, 
skin_temperature_celsius, soil_temperature_celsius,
soil_moisture_m3m3, precipitation_mm,
u_wind_component_ms, v_wind_component_ms,
wind_speed_ms, wind_direction_deg,
surface_pressure_hpa,
relative_humidity_pct, vapor_pressure_deficit_kpa
```
**15 climate variables** per observation

## 10. **Real-Time Climate Feedback**
```
Year 2023... ✓ 365 days (avg temp: 24.3°C, precip: 1247.5mm)
```
- Instant quality checks during processing
- Annual temperature and precipitation totals
- Helps identify anomalous years

## 11. **Statistical Summaries**
Per-zone aggregations:
- **Temperature**: mean, min, max, standard deviation
- **Precipitation**: sum (total), mean (daily avg), max (wettest day)
- **Soil moisture**: mean, min, max
- **Wind**: mean speed, max gust
- **Humidity & VPD**: mean values

## 12. **Zone-Based Climate Profiling**
- Same ~10 zones per district system
- Captures microclimatic variation
- Temperature gradients across elevation
- Precipitation variability in complex terrain

## 13. **Fire Risk Integration**
Critical variables for fire modeling:
- **VPD**: Primary fire weather indicator
- **Relative humidity**: Fuel moisture proxy
- **Wind speed/direction**: Fire spread vectors
- **Precipitation**: Fuel moisture recharge
- **Temperature**: Evaporative demand

## 14. **Agricultural Applications**
- **Soil moisture**: Irrigation scheduling
- **Precipitation**: Water balance calculations
- **Temperature**: Growing degree days
- **VPD**: Crop water stress
- **Wind**: Evapotranspiration

## 15. **Hydrological Variables**
- **Precipitation**: Rainfall-runoff modeling
- **Soil moisture**: Infiltration capacity
- **Temperature**: Snowmelt estimation
- **Pressure**: Storm system tracking

## 16. **Data Quality & Completeness**
- **No missing dates**: Model-based reanalysis fills all gaps
- **No cloud issues**: Not observational satellite data
- **Consistent quality**: Uniform global coverage
- **Validated**: Extensively compared with ground stations

## 17. **Error Handling**
- Year-level try-catch blocks
- Zone-level error recovery
- Continues processing despite failures
- 0.3s delay between years for rate limiting

## 18. **Hierarchical Organization**
```
era5_data/
├── District_Name/
│   ├── District_Name_era5_data.csv      # All zones combined
│   ├── District_Name_summary.csv         # Statistical summary
│   └── zone_1_data.csv, zone_2_data.csv # Individual zones
```

## Integration with Other Datasets:

### **With MODIS LST**
- Compare skin_temperature (ERA5) vs LST (satellite)
- Validate reanalysis against observations
- Gap-fill satellite data with reanalysis

### **With Fire Data (MOD14A1)**
- **VPD + RH + Wind** = Fire Weather Index
- Pre-fire meteorological conditions
- Post-fire precipitation for recovery

### **With Sentinel-2/Landsat**
- **Precipitation** explains vegetation greenness
- **Temperature** affects phenology timing
- **Soil moisture** correlates with NDVI/NDWI
- **VPD** influences vegetation stress (low NDVI)

### **With Topography (SRTM)**
- Elevation affects temperature (lapse rate)
- Aspect influences wind patterns
- Slope affects precipitation distribution

## Performance Characteristics:

- **Daily resolution**: ~9,125 days over 25 years
- **Guaranteed coverage**: Every day has data
- **Fast processing**: Model data, no cloud filtering needed
- **Year-by-year**: Prevents timeouts on large requests

## Climate Analysis Capabilities:

### **Trend Detection**
- 25-year temperature trends
- Precipitation pattern changes
- Increasing VPD (drought intensification)
- Wind pattern shifts

### **Extreme Events**
- Heat waves (max temperature)
- Droughts (low precipitation, high VPD)
- Heavy rainfall days (max precipitation)
- High wind events (max wind_speed)

### **Seasonal Analysis**
- Monsoon strength (summer precipitation)
- Winter temperatures
- Seasonal VPD cycles
- Wind direction seasonality

### **Inter-Annual Variability**
- ENSO impacts (El Niño/La Niña years)
- Year-to-year precipitation variability
- Temperature anomalies
- Drought frequency

## Fire Risk Modeling Inputs:

**Fire Weather Index Components:**
1. **Temperature**: Evaporative demand
2. **Relative Humidity**: Fuel moisture
3. **Wind Speed**: Rate of spread
4. **Precipitation**: Fuel moisture recharge
5. **VPD**: Atmospheric drying power

**Combined with satellite data:**
- Fire occurrence (MOD14A1) + meteorology = fire climatology
- Vegetation state (NDVI) + VPD = fire hazard
- NBR (burn index) + pre-fire weather = severity prediction

This code provides **gap-free, long-term climate data** essential for understanding environmental drivers of vegetation dynamics, fire regimes, and land surface changes observed in satellite imagery. The ~11km resolution is ideal for regional analysis while maintaining daily temporal detail.







--------------------------------------------------------------------------------------------------------------------------------------


Features DOne : 


Dataset	     Cleaning/Processing Status	      Notes
MODIS LST	      Done	                  QC bits, clear sky filtering, Kelvin → Celsius, day/night separation
MOD14A1	          Done	                  Fire masks, confidence levels, pixel counts, per-zone summaries
Sentinel-2	      Done	                  Cloud/shadow masking, vegetation & burn indices, per-zone statistics
Landsat 8/9	      Done	                  Cloud masking, reflectance scaling, vegetation & burn indices, multi-year harmonization
SRTM	          Done	                  Static elevation, slope, aspect, mTPI — no temporal processing required
ERA5-Land	      Done	                  Derived variables (VPD, RH, wind speed/direction), units converted, per-zone daily values


--------------------------------------------------------------------------------------------------------------------------------------

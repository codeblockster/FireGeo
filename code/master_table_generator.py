import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
from sklearn.impute import KNNImputer
warnings.filterwarnings('ignore')

# ============================================================
# MASTER FEATURE TABLE INTEGRATION FOR FFOPM - PRODUCTION READY
# ============================================================
# COMPLETE VERSION WITH ALL FIXES:
# 1. SRTM terrain data loading (district-level files)
# 2. MODIS LST column name handling (LST_Day_C_mean → lst_day_mean_c)
# 3. MODIS LST missing date column handling (static features)
# 4. Extended interpolation for vegetation (14-30 days)
# 5. Advanced missing value imputation (KNN, median)
# 6. Feature quality filtering (>70% missing)
# 7. Enhanced metadata tracking
# ============================================================

print("=" * 80)
print("FOREST FIRE OCCURRENCE PREDICTION MODEL (FFOPM)")
print("Step A: Integration & Master Feature Table (PRODUCTION READY + FIXES)")
print("=" * 80)

# ------------------------------
# Configuration
# ------------------------------
DATA_BASE = Path("/Users/prabhatrawal/Minor_project_code/data")
OUTPUT_DIR = DATA_BASE / "integrated_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "Master_FFOPM_Table.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "Master_FFOPM_Table.parquet"
METADATA_FILE = OUTPUT_DIR / "Master_FFOPM_Metadata.json"

# Quality thresholds
MISSING_THRESHOLD = 0.70  # Drop features with >70% missing
INTERPOLATION_LIMIT_SHORT = 7   # Short-term gaps (weather)
INTERPOLATION_LIMIT_MEDIUM = 14  # Medium-term gaps (vegetation)
INTERPOLATION_LIMIT_LONG = 30    # Long-term gaps (sparse vegetation)

# District mapping
DISTRICTS = {
    'District_0': 'Banke',
    'District_1': 'Bardiya',
    'District_2': 'Surkhet',
    'District_3': 'Dang',
    'District_4': 'Salyan'
}

# ------------------------------
# Helper Functions
# ------------------------------
def load_zone_data(pattern, date_col='date', zone_col='zone', verbose=True):
    """Generic loader for zone-based CSV files with error handling"""
    files = list(DATA_BASE.glob(pattern))
    if verbose:
        print(f"  Found {len(files)} files matching pattern: {pattern}")
    
    data = []
    errors = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if zone_col and zone_col in df.columns:
                df[zone_col] = df[zone_col].astype(int)
            data.append(df)
        except Exception as e:
            errors.append(f"{file.name}: {str(e)}")
    
    if errors and verbose:
        print(f"  ⚠ Errors in {len(errors)} files:")
        for err in errors[:5]:
            print(f"    - {err}")
    
    if not data:
        return None
    
    result = pd.concat(data, ignore_index=True)
    
    # Drop duplicate rows if any
    if date_col and date_col in result.columns and zone_col and zone_col in result.columns:
        if 'district' in result.columns:
            initial_rows = len(result)
            result = result.drop_duplicates(subset=[date_col, 'district', zone_col])
            if len(result) < initial_rows and verbose:
                print(f"  ⚠ Removed {initial_rows - len(result)} duplicate rows")
    
    return result

def validate_range(df, col, min_val, max_val, name=None):
    """Validate numeric column is within expected range"""
    if col not in df.columns or df[col].isna().all():
        return 0
    
    name = name or col
    invalid = ((df[col] < min_val) | (df[col] > max_val)) & df[col].notna()
    invalid_count = invalid.sum()
    
    if invalid_count > 0:
        print(f"  ⚠ {invalid_count} {name} values out of range [{min_val}, {max_val}] - clipping")
        df[col] = df[col].clip(min_val, max_val)
    
    return invalid_count

def optimize_dtypes(df):
    """Optimize memory usage by downcasting dtypes"""
    if 'district' in df.columns:
        df['district'] = df['district'].astype('category')
    if 'zone' in df.columns:
        df['zone'] = df['zone'].astype('category')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        if col not in ['zone']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

def flexible_column_match(df, keywords, exclude_keywords=None):
    """Find columns matching keywords (case-insensitive)"""
    exclude_keywords = exclude_keywords or []
    matched = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            if not any(ex in col_lower for ex in exclude_keywords):
                matched.append(col)
    return matched

def apply_knn_imputation(df, columns, n_neighbors=5, group_cols=['district', 'zone']):
    """Apply KNN imputation within district-zone groups"""
    imputed_count = 0
    
    for group_vals, group_df in df.groupby(group_cols):
        group_indices = group_df.index
        
        # Select only numeric columns that need imputation
        impute_cols = [col for col in columns if col in group_df.columns and group_df[col].isna().any()]
        
        if not impute_cols:
            continue
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=min(n_neighbors, len(group_df) - 1))
        try:
            imputed_data = imputer.fit_transform(group_df[impute_cols])
            df.loc[group_indices, impute_cols] = imputed_data
            imputed_count += group_df[impute_cols].isna().sum().sum()
        except:
            # Fallback to median if KNN fails
            for col in impute_cols:
                median_val = group_df[col].median()
                if not pd.isna(median_val):
                    df.loc[group_indices, col] = group_df[col].fillna(median_val)
    
    return imputed_count

# ------------------------------
# Step A0: Define Prediction Unit
# ------------------------------
print("\n[A0] Prediction Unit Definition")
print("-" * 80)
print("Unit of Prediction: (District, Zone, Date) → Fire or No Fire")
print("Primary Key: [date, district, zone]")
print("✓ Confirmed\n")

# ------------------------------
# Step A1: Load BASE TABLE (MOD14A1)
# ------------------------------
print("\n[A1] Loading BASE TABLE: MOD14A1 (Fire Detection)")
print("-" * 80)

base_table = load_zone_data("mod14a1_data/*/zone_*_data.csv")

if base_table is None:
    raise FileNotFoundError("No MOD14A1 data found! Check data directory.")

base_table.columns = base_table.columns.str.lower().str.replace(' ', '_')

print(f"✓ Loaded MOD14A1: {len(base_table):,} rows")
print(f"  Date range: {base_table['date'].min()} to {base_table['date'].max()}")
print(f"  Districts: {base_table['district'].nunique()}")
print(f"  Zones: {base_table['zone'].nunique()}")

# ------------------------------
# Step A3: Create TARGET VARIABLE
# ------------------------------
print("\n[A3] Creating Target Variable: Fire_Label")
print("-" * 80)

base_table['fire_label'] = (base_table['total_fire_pixels'] > 0).astype(int)

fire_days = base_table['fire_label'].sum()
no_fire_days = len(base_table) - fire_days
fire_percentage = (fire_days / len(base_table)) * 100

print(f"Fire Days: {fire_days:,} ({fire_percentage:.2f}%)")
print(f"No-Fire Days: {no_fire_days:,} ({100-fire_percentage:.2f}%)")
print(f"Imbalance Ratio: 1:{int(no_fire_days/fire_days)}")
print("✓ Target variable created")
print(f"⚠ CRITICAL: Severe class imbalance - must use SMOTE/class weights in ML")

master_table = base_table.copy()

# ------------------------------
# Step A5.1: Load MODIS LST (FIXED!)
# ------------------------------
print("\n[A5.1] Loading MODIS LST (Temperature) - FIXED")
print("-" * 80)

lst_patterns = [
    "modis_lst_data/*/zone_*_data.csv",
    "modis_lst_data/*/zone_*.csv",
    "lst_data/*/zone_*_data.csv"
]

lst_table = None
for pattern in lst_patterns:
    # Load without date conversion initially to check structure
    files = list(DATA_BASE.glob(pattern))
    if files:
        try:
            # Read first file to check columns
            sample_df = pd.read_csv(files[0], nrows=5)
            has_date_col = 'date' in [col.lower() for col in sample_df.columns]
            
            # Now load all files with proper date handling
            if has_date_col:
                lst_table = load_zone_data(pattern, date_col='date', verbose=False)
            else:
                lst_table = load_zone_data(pattern, date_col=None, verbose=False)
            
            if lst_table is not None:
                print(f"  ✓ Found LST data with pattern: {pattern}")
                break
        except Exception as e:
            print(f"  ⚠ Error checking pattern {pattern}: {e}")
            continue

if lst_table is not None:
    # Don't convert to lowercase immediately - check structure first
    print(f"  Original LST columns: {list(lst_table.columns)[:5]}...")
    
    # Check if date column exists
    has_date = 'date' in [col.lower() for col in lst_table.columns]
    
    if not has_date:
        print(f"  ⚠ WARNING: LST data missing 'date' column!")
        print(f"  This data appears to be zone-level averages without temporal dimension")
        print(f"  Merging as STATIC features on [district, zone] only...")
        
        # Standardize column names but keep structure
        lst_table.columns = lst_table.columns.str.lower()
        
        # Find LST columns (handle both LST_Day_C_mean and lst_day_c_mean patterns)
        lst_feature_cols = flexible_column_match(lst_table, 
                                                 keywords=['lst', 'temperature', 'clear', 'coverage'],
                                                 exclude_keywords=['qc', 'quality', 'satellite'])
        
        # Rename to standardized format
        rename_map = {}
        for col in lst_feature_cols:
            # Handle various LST column naming patterns
            col_parts = col.replace('_', ' ').lower()
            if 'day' in col_parts and 'mean' in col_parts:
                rename_map[col] = 'lst_day_mean_c'
            elif 'day' in col_parts and 'min' in col_parts:
                rename_map[col] = 'lst_day_min_c'
            elif 'day' in col_parts and 'max' in col_parts:
                rename_map[col] = 'lst_day_max_c'
            elif 'day' in col_parts and ('std' in col_parts or 'stddev' in col_parts):
                rename_map[col] = 'lst_day_std_c'
            elif 'day' in col_parts and 'count' in col_parts:
                rename_map[col] = 'lst_day_count'
            elif 'night' in col_parts and 'mean' in col_parts:
                rename_map[col] = 'lst_night_mean_c'
            elif 'night' in col_parts and 'min' in col_parts:
                rename_map[col] = 'lst_night_min_c'
            elif 'night' in col_parts and 'max' in col_parts:
                rename_map[col] = 'lst_night_max_c'
            elif 'night' in col_parts and ('std' in col_parts or 'stddev' in col_parts):
                rename_map[col] = 'lst_night_std_c'
            elif 'night' in col_parts and 'count' in col_parts:
                rename_map[col] = 'lst_night_count'
            elif 'clear' in col_parts and 'day' in col_parts:
                rename_map[col] = 'clear_day_coverage'
            elif 'clear' in col_parts and 'night' in col_parts:
                rename_map[col] = 'clear_night_coverage'
        
        if rename_map:
            lst_table = lst_table.rename(columns=rename_map)
            print(f"  ✓ Standardized {len(rename_map)} LST column names")
            print(f"    Sample mappings: {list(rename_map.items())[:3]}")
        
        # Keep only relevant columns
        keep_cols = ['district', 'zone'] + list(rename_map.values())
        available_cols = [col for col in keep_cols if col in lst_table.columns]
        lst_table = lst_table[available_cols]
        
        # Remove duplicates
        lst_table = lst_table.drop_duplicates(subset=['district', 'zone'])
        
        # Merge as static features (no date dimension)
        master_table = master_table.merge(lst_table, on=['district', 'zone'], how='left')
        
        lst_cols_in_master = list(rename_map.values())
        print(f"✓ Merged LST as STATIC features: {len(lst_cols_in_master)} columns")
        print(f"  Features: {lst_cols_in_master}")
        
        # Validate ranges
        for col in ['lst_day_mean_c', 'lst_night_mean_c']:
            if col in master_table.columns:
                validate_range(master_table, col, -50, 70, f'LST {col}')
        
        missing_lst = master_table[lst_cols_in_master[0]].isna().sum() if lst_cols_in_master else 0
        print(f"  Missing LST: {missing_lst:,} ({missing_lst/len(master_table)*100:.1f}%)")
        
    else:
        # Original temporal LST handling
        lst_table.columns = lst_table.columns.str.lower().str.replace(' ', '_')
        
        lst_feature_cols = flexible_column_match(lst_table, 
                                                 keywords=['lst', 'temperature', 'temp', 'clear', 'coverage'],
                                                 exclude_keywords=['qc', 'quality'])
        
        required_cols = ['date', 'district', 'zone']
        for col in required_cols:
            if col not in lst_feature_cols:
                lst_feature_cols.insert(0, col)
        
        lst_feature_cols = [col for col in lst_feature_cols if col in lst_table.columns]
        lst_table = lst_table[lst_feature_cols]
        
        # Standardize column names for temporal data
        rename_map = {}
        for col in lst_table.columns:
            if 'lst' in col.lower() and 'day' in col.lower() and col not in required_cols:
                if 'mean' in col.lower() or 'avg' in col.lower():
                    rename_map[col] = 'lst_day_mean_c'
            elif 'lst' in col.lower() and 'night' in col.lower() and col not in required_cols:
                if 'mean' in col.lower() or 'avg' in col.lower():
                    rename_map[col] = 'lst_night_mean_c'
        
        if rename_map:
            lst_table = lst_table.rename(columns=rename_map)
            print(f"  ✓ Standardized {len(rename_map)} LST column names")
        
        master_table = master_table.merge(lst_table, on=['date', 'district', 'zone'], how='left')
        
        # Enhanced interpolation for LST
        lst_cols_in_master = [col for col in master_table.columns 
                             if 'lst' in col.lower() and col not in ['date', 'district', 'zone']]
        
        if len(lst_cols_in_master) > 0:
            master_table = master_table.sort_values(['district', 'zone', 'date'])
            for col in lst_cols_in_master:
                if master_table[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    before_missing = master_table[col].isna().sum()
                    master_table[col] = master_table.groupby(['district', 'zone'])[col].transform(
                        lambda x: x.interpolate(method='linear', limit=INTERPOLATION_LIMIT_SHORT)
                    )
                    after_missing = master_table[col].isna().sum()
                    filled = before_missing - after_missing
                    if filled > 0:
                        print(f"  ✓ Interpolated {filled:,} values for {col}")
            
            master_table['lst_missing_flag'] = master_table[lst_cols_in_master[0]].isna().astype(int)
            
            print(f"✓ Merged LST: {len(master_table):,} rows")
            missing_lst = master_table[lst_cols_in_master[0]].isna().sum()
            print(f"  Missing LST after interpolation: {missing_lst:,} ({missing_lst/len(master_table)*100:.1f}%)")
else:
    print("⚠ No LST data found")
    lst_cols_in_master = []

# ------------------------------
# Step A5.2: Load Sentinel-2
# ------------------------------
print("\n[A5.2] Loading Sentinel-2 (Vegetation Indices)")
print("-" * 80)

s2_table = load_zone_data("sentinel2_data/*/zone_*_data.csv")

if s2_table is not None:
    s2_table.columns = s2_table.columns.str.lower().str.replace(' ', '_')
    
    s2_feature_cols = flexible_column_match(s2_table, 
                                           keywords=['ndvi', 'gndvi', 'nbr', 'ndwi', 'ndsi', 'evi', 'savi', 'cloud'],
                                           exclude_keywords=['qc', 'quality'])
    
    for col in ['date', 'district', 'zone']:
        if col not in s2_feature_cols:
            s2_feature_cols.insert(0, col)
    
    s2_feature_cols = [col for col in s2_feature_cols if col in s2_table.columns]
    s2_table = s2_table[s2_feature_cols]
    
    rename_map = {col: f's2_{col}' for col in s2_table.columns if col not in ['date', 'district', 'zone']}
    s2_table = s2_table.rename(columns=rename_map)
    
    print(f"  Selected {len(s2_feature_cols)-3} Sentinel-2 features")
    
    s2_table = s2_table.sort_values(['district', 'zone', 'date'])
    master_table = master_table.merge(s2_table, on=['date', 'district', 'zone'], how='left')
    
    # Forward fill + extended interpolation
    master_table = master_table.sort_values(['district', 'zone', 'date'])
    s2_index_cols = [col for col in master_table.columns if col.startswith('s2_')]
    
    for col in s2_index_cols:
        before_missing = master_table[col].isna().sum()
        # Step 1: Forward fill
        master_table[col] = master_table.groupby(['district', 'zone'])[col].ffill()
        # Step 2: Interpolate remaining gaps (up to 30 days for sparse satellite data)
        master_table[col] = master_table.groupby(['district', 'zone'])[col].transform(
            lambda x: x.interpolate(method='linear', limit=INTERPOLATION_LIMIT_LONG)
        )
        after_missing = master_table[col].isna().sum()
        filled = before_missing - after_missing
        if filled > 0:
            print(f"  ✓ Filled {filled:,} values for {col} (ffill + interpolation)")
    
    print(f"✓ Merged Sentinel-2: {len(master_table):,} rows")
    if 's2_ndvi' in master_table.columns:
        missing_ndvi = master_table['s2_ndvi'].isna().sum()
        print(f"  Missing S2 NDVI: {missing_ndvi:,} ({missing_ndvi/len(master_table)*100:.1f}%)")
        validate_range(master_table, 's2_ndvi', -1, 1, 'Sentinel-2 NDVI')
else:
    print("⚠ No Sentinel-2 data found")

# ------------------------------
# Step A5.3: Load Landsat 8/9
# ------------------------------
print("\n[A5.3] Loading Landsat 8/9 (Additional Vegetation Data)")
print("-" * 80)

landsat_table = load_zone_data("landsat_data/*/zone_*_landsat_data.csv")
if landsat_table is None:
    landsat_table = load_zone_data("landsat_data/*/zone_*_data.csv")

if landsat_table is not None:
    landsat_table.columns = landsat_table.columns.str.lower().str.replace(' ', '_')
    
    veg_cols = flexible_column_match(landsat_table,
                                    keywords=['ndvi', 'gndvi', 'nbr', 'ndwi', 'evi', 'savi'],
                                    exclude_keywords=['qc', 'quality', 'satellite', 'cloud'])
    
    landsat_rename = {col: f'landsat_{col}' for col in veg_cols}
    landsat_table = landsat_table.rename(columns=landsat_rename)
    
    keep_cols = ['date', 'district', 'zone'] + list(landsat_rename.values())
    available_cols = [col for col in keep_cols if col in landsat_table.columns]
    landsat_table = landsat_table[available_cols]
    
    print(f"  Selected {len(available_cols)-3} Landsat features")
    
    landsat_table = landsat_table.sort_values(['district', 'zone', 'date'])
    master_table = master_table.merge(landsat_table, on=['date', 'district', 'zone'], how='left')
    
    # Forward fill + extended interpolation
    landsat_cols = [col for col in master_table.columns if col.startswith('landsat_')]
    for col in landsat_cols:
        before_missing = master_table[col].isna().sum()
        master_table[col] = master_table.groupby(['district', 'zone'])[col].ffill()
        master_table[col] = master_table.groupby(['district', 'zone'])[col].transform(
            lambda x: x.interpolate(method='linear', limit=INTERPOLATION_LIMIT_LONG)
        )
        after_missing = master_table[col].isna().sum()
        filled = before_missing - after_missing
        if filled > 0:
            print(f"  ✓ Filled {filled:,} values for {col}")
    
    print(f"✓ Merged Landsat: {len(master_table):,} rows")
    if 'landsat_ndvi' in master_table.columns:
        missing_landsat = master_table['landsat_ndvi'].isna().sum()
        print(f"  Missing Landsat NDVI: {missing_landsat:,} ({missing_landsat/len(master_table)*100:.1f}%)")
        validate_range(master_table, 'landsat_ndvi', -1, 1, 'Landsat NDVI')
else:
    print("⚠ No Landsat data found")

# ------------------------------
# Step A5.4: Load ERA5-Land
# ------------------------------
print("\n[A5.4] Loading ERA5-Land (Climate/Weather)")
print("-" * 80)

era5_table = load_zone_data("era5_data/*/zone_*_data.csv")

if era5_table is not None:
    era5_table.columns = era5_table.columns.str.lower().str.replace(' ', '_')
    
    weather_cols = flexible_column_match(era5_table,
                                        keywords=['temperature', 'temp', 'humidity', 'vapor', 'vpd', 
                                                 'wind', 'precipitation', 'precip', 'rain', 'soil', 'moisture',
                                                 'dewpoint', 'pressure'],
                                        exclude_keywords=['qc', 'quality'])
    
    era5_cols = ['date', 'district', 'zone'] + weather_cols
    available_cols = [col for col in era5_cols if col in era5_table.columns]
    era5_table = era5_table[available_cols]
    
    print(f"  Selected {len(available_cols)-3} ERA5 weather features")
    
    master_table = master_table.merge(era5_table, on=['date', 'district', 'zone'], how='left')
    
    print(f"✓ Merged ERA5: {len(master_table):,} rows")
    
    weather_cols_in_master = [col for col in master_table.columns if col in weather_cols]
    if len(weather_cols_in_master) > 0:
        missing_era5 = master_table[weather_cols_in_master[0]].isna().sum()
        print(f"  Missing ERA5 values: {missing_era5:,} ({missing_era5/len(master_table)*100:.1f}%)")
else:
    print("⚠ No ERA5 data found")

# ------------------------------
# Step A5.5: Load SRTM (FIXED - District Level Files)
# ------------------------------
print("\n[A5.5] Loading SRTM (Topography - FIXED)")
print("-" * 80)

# Load district-level SRTM files
srtm_files = [
    "srtm_data/District_0/District_0_srtm_data.csv",
    "srtm_data/District_1/District_1_srtm_data.csv",
    "srtm_data/District_2/District_2_srtm_data.csv",
    "srtm_data/District_3/District_3_srtm_data.csv",
    "srtm_data/District_4/District_4_srtm_data.csv"
]

srtm_data_list = []
for file_path in srtm_files:
    full_path = DATA_BASE / file_path
    if full_path.exists():
        try:
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            if 'zone' in df.columns:
                df['zone'] = df['zone'].astype(int)
            srtm_data_list.append(df)
            print(f"  ✓ Loaded {full_path.name}")
        except Exception as e:
            print(f"  ⚠ Error loading {full_path.name}: {e}")

if srtm_data_list:
    srtm_table = pd.concat(srtm_data_list, ignore_index=True)
    
    # Find terrain columns
    terrain_cols = flexible_column_match(srtm_table,
                                        keywords=['elevation', 'elev', 'slope', 'aspect', 'mtpi', 'tpi', 'roughness'],
                                        exclude_keywords=['qc', 'quality'])
    
    srtm_cols = ['district', 'zone'] + terrain_cols
    available_cols = [col for col in srtm_cols if col in srtm_table.columns]
    srtm_table = srtm_table[available_cols]
    
    # Remove duplicates
    srtm_table = srtm_table.drop_duplicates(subset=['district', 'zone'])
    
    print(f"  ✓ Combined SRTM data: {len(srtm_table)} zones")
    print(f"  ✓ Terrain features: {len(terrain_cols)}")
    print(f"    Features: {terrain_cols[:10]}...")
    
    # Merge with master table
    master_table = master_table.merge(srtm_table, on=['district', 'zone'], how='left')
    
    print(f"✓ Merged SRTM: {len(master_table):,} rows")
    
    # Check coverage
    if len(terrain_cols) > 0:
        first_terrain_col = terrain_cols[0]
        missing_terrain = master_table[first_terrain_col].isna().sum()
        if missing_terrain > 0:
            print(f"  ⚠ {missing_terrain:,} rows without terrain data")
        else:
            print(f"  ✓ Complete terrain coverage")
else:
    print("⚠ CRITICAL: No SRTM data loaded - terrain features missing!")
    print("  This will significantly impact model performance.")
    terrain_cols = []

# ------------------------------
# Step A8: Create LAG FEATURES
# ------------------------------
print("\n[A8] Creating Lag Features (Temporal Context)")
print("-" * 80)

master_table = master_table.sort_values(['district', 'zone', 'date'])

lag_features = {}

if 's2_ndvi' in master_table.columns:
    lag_features['s2_ndvi'] = [1, 3, 7, 14]
if 'landsat_ndvi' in master_table.columns:
    lag_features['landsat_ndvi'] = [1, 3, 7, 14]

lst_temp_cols = [col for col in master_table.columns 
                if 'lst' in col.lower() and 'mean' in col.lower() and 'flag' not in col.lower()]
for col in lst_temp_cols:
    lag_features[col] = [1, 3, 7]

precip_cols = flexible_column_match(master_table, keywords=['precipitation', 'precip', 'rain'])
for col in precip_cols:
    lag_features[col] = [1, 5, 10, 30]

vpd_cols = flexible_column_match(master_table, keywords=['vapor', 'vpd'])
for col in vpd_cols:
    lag_features[col] = [1, 3, 7, 14]

soil_cols = flexible_column_match(master_table, keywords=['soil', 'moisture'])
for col in soil_cols:
    lag_features[col] = [1, 3, 7, 14]

temp_cols = flexible_column_match(master_table, keywords=['temperature', 'temp'], exclude_keywords=['lst'])
for col in temp_cols:
    lag_features[col] = [1, 3, 7]

humid_cols = flexible_column_match(master_table, keywords=['humidity'])
for col in humid_cols:
    lag_features[col] = [1, 3, 7]

lag_count = 0
for feature, lags in lag_features.items():
    if feature in master_table.columns:
        for lag in lags:
            col_name = f'{feature}_lag{lag}'
            master_table[col_name] = master_table.groupby(['district', 'zone'])[feature].shift(lag)
            lag_count += 1

print(f"✓ Created {lag_count} lag features")

# ------------------------------
# Step A9: Create ROLLING STATISTICS
# ------------------------------
print("\n[A9] Creating Rolling Statistics (Temporal Trends)")
print("-" * 80)

rolling_features = {}

precip_cols = flexible_column_match(master_table, keywords=['precipitation', 'precip', 'rain'], exclude_keywords=['lag', 'roll'])
for col in precip_cols:
    rolling_features[col] = {'windows': [7, 14, 30], 'agg': 'sum'}

if 's2_ndvi' in master_table.columns:
    rolling_features['s2_ndvi'] = {'windows': [7, 14], 'agg': 'mean'}
if 'landsat_ndvi' in master_table.columns:
    rolling_features['landsat_ndvi'] = {'windows': [7, 14], 'agg': 'mean'}

temp_cols = flexible_column_match(master_table, keywords=['temperature', 'temp'], exclude_keywords=['lag', 'roll', 'lst'])
for col in temp_cols[:1]:
    rolling_features[col] = {'windows': [7, 14], 'agg': 'mean'}

vpd_cols = flexible_column_match(master_table, keywords=['vapor', 'vpd'], exclude_keywords=['lag', 'roll'])
for col in vpd_cols:
    rolling_features[col] = {'windows': [7, 14], 'agg': 'mean'}

rolling_count = 0
for feature, config in rolling_features.items():
    if feature in master_table.columns:
        for window in config['windows']:
            agg = config['agg']
            col_name = f'{feature}_roll{window}_{agg}'
            master_table[col_name] = master_table.groupby(['district', 'zone'])[feature].transform(
                lambda x: x.rolling(window, min_periods=1).agg(agg)
            )
            rolling_count += 1

print(f"✓ Created {rolling_count} rolling features")

# ------------------------------
# Step A15: ENHANCED POST-INTEGRATION
# ------------------------------
print("\n[A15] Enhanced Post-Integration Processing")
print("-" * 80)

# 1. Composite NDVI
if 's2_ndvi' in master_table.columns and 'landsat_ndvi' in master_table.columns:
    master_table['ndvi_composite'] = master_table['s2_ndvi'].fillna(master_table['landsat_ndvi'])
    missing_composite = master_table['ndvi_composite'].isna().sum()
    print(f"✓ Created composite NDVI")
    print(f"  Missing: {missing_composite:,} ({missing_composite/len(master_table)*100:.1f}%)")
elif 's2_ndvi' in master_table.columns:
    master_table['ndvi_composite'] = master_table['s2_ndvi']
    print("✓ Created composite NDVI (Sentinel-2 only)")
elif 'landsat_ndvi' in master_table.columns:
    master_table['ndvi_composite'] = master_table['landsat_ndvi']
    print("✓ Created composite NDVI (Landsat only)")

# 2. Vegetation quality flag
if 's2_ndvi' in master_table.columns and 'landsat_ndvi' in master_table.columns:
    master_table['veg_data_quality'] = (
        (~master_table['s2_ndvi'].isna()).astype(int) * 2 +
        (~master_table['landsat_ndvi'].isna()).astype(int)
    )
    quality_counts = master_table['veg_data_quality'].value_counts().sort_index()
    print("✓ Created vegetation data quality flag (0-3):")
    for val in [0, 1, 2, 3]:
        count = quality_counts.get(val, 0)
        pct = count / len(master_table) * 100
        label = {0: 'no data', 1: 'Landsat only', 2: 'S2 only', 3: 'both'}[val]
        print(f"  {val} ({label}): {count:,} ({pct:.1f}%)")

# 3. KNN Imputation for LST
print("\n✓ Applying KNN imputation for remaining LST gaps...")
lst_cols_to_impute = [col for col in master_table.columns 
                      if 'lst' in col.lower() and 'mean' in col.lower() and 'flag' not in col.lower()]
if lst_cols_to_impute:
    before_total = sum(master_table[col].isna().sum() for col in lst_cols_to_impute)
    imputed = apply_knn_imputation(master_table, lst_cols_to_impute, n_neighbors=5)
    after_total = sum(master_table[col].isna().sum() for col in lst_cols_to_impute)
    print(f"  Imputed {before_total - after_total:,} LST values using KNN")

# 4. Median imputation for weather features
print("\n✓ Median imputation for weather features...")
weather_cols_base = flexible_column_match(master_table, 
                                         keywords=['temperature', 'humidity', 'wind', 'soil'],
                                         exclude_keywords=['lag', 'roll', 'lst'])
for col in weather_cols_base:
    if master_table[col].isna().any():
        before = master_table[col].isna().sum()
        master_table[col] = master_table.groupby(['district', 'zone'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        after = master_table[col].isna().sum()
        if before > after:
            print(f"  {col}: filled {before - after:,} values with group median")

# ------------------------------
# Step A16: FEATURE QUALITY FILTERING
# ------------------------------
print("\n[A16] Feature Quality Filtering (Drop >70% Missing)")
print("-" * 80)

# Calculate missing percentages
missing_pct = master_table.isnull().mean() * 100
high_missing = missing_pct[missing_pct > MISSING_THRESHOLD * 100].sort_values(ascending=False)

if len(high_missing) > 0:
    print(f"Found {len(high_missing)} features with >{MISSING_THRESHOLD*100}% missing:")
    
    # Identify columns to drop (exclude key columns)
    protected_cols = ['date', 'district', 'zone', 'fire_label', 'ndvi_composite', 'veg_data_quality']
    cols_to_drop = [col for col in high_missing.index if col not in protected_cols]
    
    for col in cols_to_drop[:10]:  # Show first 10
        print(f"  {col}: {missing_pct[col]:.1f}% missing")
    
    if len(cols_to_drop) > 10:
        print(f"  ... and {len(cols_to_drop) - 10} more")
    
    # Drop low-quality features
    master_table = master_table.drop(columns=cols_to_drop)
    print(f"\n✓ Dropped {len(cols_to_drop)} low-quality features")
    print(f"  Remaining columns: {len(master_table.columns)}")
else:
    print("✓ No features exceed 70% missing threshold")
    cols_to_drop = []

# ------------------------------
# Step A10: Data Validation
# ------------------------------
print("\n[A10] Data Validation & Quality Checks")
print("-" * 80)

duplicates = master_table.duplicated(subset=['date', 'district', 'zone']).sum()
if duplicates > 0:
    print(f"⚠ WARNING: {duplicates} duplicate rows found - removing...")
    master_table = master_table.drop_duplicates(subset=['date', 'district', 'zone'])
else:
    print("✓ No duplicate rows")

assert master_table['fire_label'].isin([0, 1]).all(), "Invalid fire_label values"
print("✓ Target variable validated")

# Validate ranges
validation_results = {}
if 's2_ndvi' in master_table.columns:
    validation_results['s2_ndvi'] = validate_range(master_table, 's2_ndvi', -1, 1, 'Sentinel-2 NDVI')
if 'landsat_ndvi' in master_table.columns:
    validation_results['landsat_ndvi'] = validate_range(master_table, 'landsat_ndvi', -1, 1, 'Landsat NDVI')
if 'ndvi_composite' in master_table.columns:
    validation_results['ndvi_composite'] = validate_range(master_table, 'ndvi_composite', -1, 1, 'Composite NDVI')

humid_cols = flexible_column_match(master_table, keywords=['humidity'], exclude_keywords=['lag', 'roll'])
for col in humid_cols:
    validation_results[col] = validate_range(master_table, col, 0, 100, f'Humidity')

print("✓ Range validation complete")

# ------------------------------
# Step A11: Optimize Memory
# ------------------------------
print("\n[A11] Memory Optimization")
print("-" * 80)

initial_memory = master_table.memory_usage(deep=True).sum() / 1024**2
print(f"Initial memory usage: {initial_memory:.2f} MB")

master_table = optimize_dtypes(master_table)

final_memory = master_table.memory_usage(deep=True).sum() / 1024**2
print(f"Optimized memory usage: {final_memory:.2f} MB")
print(f"Memory saved: {initial_memory - final_memory:.2f} MB ({(1 - final_memory/initial_memory)*100:.1f}%)")

# ------------------------------
# Step A12: Column Organization
# ------------------------------
print("\n[A12] Organizing Columns")
print("-" * 80)

id_cols = ['date', 'district', 'zone']
target_col = ['fire_label']
fire_cols = [col for col in master_table.columns if 'fire' in col.lower() and col != 'fire_label']

lst_cols = [col for col in master_table.columns if 'lst' in col.lower() and 'lag' not in col and 'roll' not in col]

s2_veg_cols = [col for col in master_table.columns if col.startswith('s2_') and 'lag' not in col and 'roll' not in col]
landsat_veg_cols = [col for col in master_table.columns if col.startswith('landsat_') and 'lag' not in col and 'roll' not in col]
composite_veg_cols = [col for col in master_table.columns if 'composite' in col.lower() or 'veg_data_quality' in col]

weather_keywords = ['temperature', 'humidity', 'vpd', 'vapor', 'wind', 'precipitation', 'rain', 'soil', 'dewpoint', 'pressure']
weather_cols = [col for col in master_table.columns 
                if any(w in col.lower() for w in weather_keywords) 
                and 'lag' not in col and 'roll' not in col and 'lst' not in col.lower()]

terrain_keywords = ['elevation', 'slope', 'aspect', 'mtpi', 'tpi', 'roughness']
terrain_cols = [col for col in master_table.columns if any(t in col.lower() for t in terrain_keywords)]

lag_cols = [col for col in master_table.columns if 'lag' in col]
roll_cols = [col for col in master_table.columns if 'roll' in col]

column_order = (id_cols + target_col + fire_cols + lst_cols + 
                s2_veg_cols + landsat_veg_cols + composite_veg_cols +
                weather_cols + terrain_cols + lag_cols + roll_cols)

seen = set()
column_order_unique = []
for col in column_order:
    if col in master_table.columns and col not in seen:
        column_order_unique.append(col)
        seen.add(col)

for col in master_table.columns:
    if col not in seen:
        column_order_unique.append(col)

master_table = master_table[column_order_unique]
print(f"✓ Columns organized: {len(master_table.columns)} total")

# ------------------------------
# Step A13: Final Summary
# ------------------------------
print("\n[A13] Final Master Table Summary")
print("-" * 80)

print(f"\n📊 Dataset Overview:")
print(f"  Total rows: {len(master_table):,}")
print(f"  Total columns: {len(master_table.columns)}")
print(f"  Date range: {master_table['date'].min()} to {master_table['date'].max()}")
print(f"  Total days: {(master_table['date'].max() - master_table['date'].min()).days}")
print(f"  Districts: {master_table['district'].nunique()}")
print(f"  Zones: {master_table['zone'].nunique()}")

print(f"\n🎯 Target Variable:")
print(f"  Fire days: {fire_days:,} ({fire_percentage:.2f}%)")
print(f"  No-fire days: {no_fire_days:,} ({100-fire_percentage:.2f}%)")
print(f"  Imbalance ratio: 1:{int(no_fire_days/fire_days)}")
print(f"  ⚠ CRITICAL: Use SMOTE/class weights in ML pipeline")

print(f"\n📈 Feature Categories:")
print(f"  Fire metrics: {len(fire_cols)}")
print(f"  LST features: {len(lst_cols)}")
print(f"  Sentinel-2 vegetation: {len(s2_veg_cols)}")
print(f"  Landsat vegetation: {len(landsat_veg_cols)}")
print(f"  Composite vegetation: {len(composite_veg_cols)}")
print(f"  Weather features: {len(weather_cols)}")
print(f"  Terrain features: {len(terrain_cols)}")
print(f"  Lag features: {len(lag_cols)}")
print(f"  Rolling statistics: {len(roll_cols)}")
print(f"  Total features: {len(master_table.columns) - len(id_cols) - len(target_col)}")

print(f"\n📊 Missing Value Summary:")
missing_summary = master_table.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(missing_summary) > 0:
    print(f"  Columns with missing values: {len(missing_summary)}")
    total_missing = missing_summary.sum()
    total_cells = len(master_table) * len(master_table.columns)
    print(f"  Total missing cells: {total_missing:,} ({total_missing/total_cells*100:.2f}%)")
    print(f"\n  Top 10 columns with missing values:")
    for col, count in missing_summary.head(10).items():
        pct = (count / len(master_table)) * 100
        print(f"    {col}: {count:,} ({pct:.1f}%)")
else:
    print("  ✓ No missing values!")

# ------------------------------
# Step A14: Save Master Table
# ------------------------------
print("\n[A14] Saving Master Feature Table")
print("-" * 80)

master_table.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved CSV: {OUTPUT_FILE}")
print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

master_table.to_parquet(OUTPUT_PARQUET, index=False, compression='snappy')
print(f"✓ Saved Parquet: {OUTPUT_PARQUET}")
print(f"  File size: {OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.2f} MB")

# Comprehensive metadata
metadata = {
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script_version': 'FFOPM Step A - Production Ready v5.0 (Column Fixes)',
    
    'dataset_info': {
        'total_rows': int(len(master_table)),
        'total_columns': int(len(master_table.columns)),
        'date_range_start': str(master_table['date'].min()),
        'date_range_end': str(master_table['date'].max()),
        'total_days': int((master_table['date'].max() - master_table['date'].min()).days),
        'districts': master_table['district'].unique().tolist(),
        'zones_per_district': {str(k): int(v) for k, v in master_table.groupby('district')['zone'].nunique().items()},
        'total_zones': int(master_table['zone'].nunique())
    },
    
    'target_variable': {
        'name': 'fire_label',
        'fire_days': int(fire_days),
        'no_fire_days': int(no_fire_days),
        'fire_percentage': float(fire_percentage),
        'imbalance_ratio': f"1:{int(no_fire_days/fire_days)}",
        'ml_strategy_required': 'SMOTE, class_weight, or focal_loss'
    },
    
    'feature_groups': {
        'identifiers': id_cols,
        'target': target_col,
        'fire_metrics': fire_cols,
        'temperature_lst': lst_cols,
        'sentinel2_vegetation': s2_veg_cols,
        'landsat_vegetation': landsat_veg_cols,
        'composite_vegetation': composite_veg_cols,
        'weather': weather_cols,
        'terrain': terrain_cols,
        'lag_features': lag_cols,
        'rolling_statistics': roll_cols
    },
    
    'feature_counts': {
        'fire_metrics': len(fire_cols),
        'temperature_lst': len(lst_cols),
        'sentinel2_vegetation': len(s2_veg_cols),
        'landsat_vegetation': len(landsat_veg_cols),
        'composite_vegetation': len(composite_veg_cols),
        'weather': len(weather_cols),
        'terrain': len(terrain_cols),
        'lag_features': len(lag_cols),
        'rolling_statistics': len(roll_cols),
        'total_features': len(master_table.columns) - len(id_cols) - len(target_col)
    },
    
    'data_quality': {
        'columns_with_missing': int(len(missing_summary)),
        'total_missing_cells': int(missing_summary.sum()) if len(missing_summary) > 0 else 0,
        'missing_percentage': float((missing_summary.sum() / (len(master_table) * len(master_table.columns))) * 100) if len(missing_summary) > 0 else 0.0,
        'dropped_low_quality_features': len(cols_to_drop),
        'quality_threshold': f">{MISSING_THRESHOLD*100}% missing",
        'validation_issues': {k: int(v) for k, v in validation_results.items() if v > 0}
    },
    
    'enhancements_applied': {
        'composite_ndvi': 'ndvi_composite' in master_table.columns,
        'vegetation_quality_flag': 'veg_data_quality' in master_table.columns,
        'lst_knn_imputation': True,
        'weather_median_imputation': True,
        'extended_interpolation_vegetation': f'{INTERPOLATION_LIMIT_LONG} days',
        'short_interpolation_weather': f'{INTERPOLATION_LIMIT_SHORT} days',
        'terrain_features_loaded': len(terrain_cols) > 0,
        'feature_quality_filtering': True,
        'modis_lst_column_fixes': 'LST_Day_C_mean → lst_day_mean_c',
        'modis_lst_static_merge': 'Merged as static if no date column'
    },
    
    'memory_usage': {
        'initial_mb': float(initial_memory),
        'optimized_mb': float(final_memory),
        'saved_mb': float(initial_memory - final_memory),
        'reduction_percentage': float((1 - final_memory/initial_memory) * 100)
    },
    
    'ml_readiness': {
        'terrain_data': 'present' if len(terrain_cols) > 0 else 'missing',
        'lst_data': 'present' if len(lst_cols) > 0 else 'missing',
        'composite_ndvi': 'present' if 'ndvi_composite' in master_table.columns else 'missing',
        'class_imbalance_warning': True,
        'recommended_preprocessing': [
            'StandardScaler or RobustScaler for features',
            'SMOTE or class_weight for imbalance',
            'Temporal train/test split (2000-2020 train, 2021-2025 test)',
            'Cross-validation by year to avoid data leakage'
        ]
    },
    
    'column_names': master_table.columns.tolist()
}

with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: {METADATA_FILE}")

# ------------------------------
# Final Summary
# ------------------------------
print("\n" + "=" * 80)
print("STEP (A) COMPLETE: Production-Ready Master Feature Table")
print("=" * 80)

print(f"\n📁 Output Files:")
print(f"  • CSV: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"  • Parquet: {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"  • Metadata: {METADATA_FILE}")

print(f"\n✅ All Fixes Applied:")
print(f"  • SRTM terrain data loaded ({len(terrain_cols)} features)")
print(f"  • MODIS LST column name standardization (LST_Day_C_mean → lst_day_mean_c)")
print(f"  • MODIS LST static merge (no date column handling)")
print(f"  • Extended interpolation (vegetation: {INTERPOLATION_LIMIT_LONG} days)")
print(f"  • KNN imputation for LST gaps")
print(f"  • Median imputation for weather")
print(f"  • Composite NDVI (S2 + Landsat fusion)")
print(f"  • Vegetation quality flag (0-3 scale)")
print(f"  • Quality filtering (dropped {len(cols_to_drop)} features >{MISSING_THRESHOLD*100}% missing)")
print(f"  • Memory optimization ({(1 - final_memory/initial_memory)*100:.1f}% reduction)")

print(f"\n🎯 Data Quality Status:")
has_lst = len(lst_cols) > 0
has_terrain = len(terrain_cols) > 0
has_composite = 'ndvi_composite' in master_table.columns
has_quality = 'veg_data_quality' in master_table.columns

status_lst = '✓ Present' if has_lst else '✗ Missing'
status_terrain = '✓ Present' if has_terrain else '✗ Missing - CRITICAL'
status_composite = '✓ Created' if has_composite else '✗ Not created'
status_quality = '✓ Created' if has_quality else '✗ Not created'

print(f"  • LST Features: {status_lst}")
print(f"  • Terrain Features: {status_terrain}")
print(f"  • Composite NDVI: {status_composite}")
print(f"  • Vegetation Quality Flag: {status_quality}")

if len(missing_summary) > 0:
    print(f"\n⚠️  Remaining Missing Values: {len(missing_summary)} columns")
    worst_missing = missing_summary.head(3)
    for col, count in worst_missing.items():
        pct = count / len(master_table) * 100
        print(f"    • {col}: {pct:.1f}%")

print(f"\n🚀 Next Steps for ML Pipeline:")
print("  1. Load: df = pd.read_parquet('Master_FFOPM_Table.parquet')")
print("  2. Train/Test Split: Temporal split (2000-2020 train, 2021-2025 test)")
print("  3. Handle Imbalance: SMOTE or class_weight (1:141 ratio)")
print("  4. Feature Scaling: StandardScaler or RobustScaler")
print("  5. Model Selection: XGBoost, LightGBM, or Random Forest")
print("  6. Evaluation: Precision, Recall, F1-Score (focus on fire class)")

print("\n" + "=" * 80)
print("✓ Integration complete! Production-ready for ML with all column fixes.")
print("=" * 80)
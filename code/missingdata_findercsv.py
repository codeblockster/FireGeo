import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# --- 1. SETUP PATHS ---
# Base directory for District 4
BASE_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/sentinel2_data/District_0")

# Specific File paths
DISTRICT_FILE = BASE_DIR / "District_0_sentinel2_data.csv"
SUMMARY_FILE = BASE_DIR / "District_0_summary.csv"

# Configuration
NUM_ZONES = 14  # You mentioned District 4 has 14 zones

# Set visual style
plt.style.use('ggplot')

def check_missing_data(df, name):
    """
    Analyzes a dataframe for missing values and prints a summary.
    """
    print(f"\n{'='*20} ANALYZING: {name} {'='*20}")
    
    # Check if empty
    if df.empty:
        print("❌ Dataframe is EMPTY.")
        return

    # Basic Info
    total_rows = len(df)
    print(f"Total Rows: {total_rows}")
    
    # Check for nulls/NaNs
    missing_counts = df.isnull().sum()
    
    # Filter to show only columns with missing data
    missing_only = missing_counts[missing_counts > 0]
    
    if not missing_only.empty:
        print("\n⚠️  MISSING VALUES FOUND (NaN):")
        print(missing_only)
    else:
        print("\n✅  No empty cells (NaN) found.")

    # Check Date Gaps
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        unique_dates = df['date'].dt.date.unique()
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"Date Range: {min_date.date()} to {max_date.date()}")
        
        # Calculate expected days
        full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        if len(unique_dates) == len(full_range):
            print(f"✅  Date sequence is continuous ({len(unique_dates)} days).")
        else:
            print(f"⚠️  DATE GAPS DETECTED!")
            print(f"   Expected days: {len(full_range)}")
            print(f"   Actual unique days: {len(unique_dates)}")
            print(f"   Missing days count: {len(full_range) - len(unique_dates)}")

def plot_visuals(df, name, expected_count):
    """
    Creates the Matplotlib visualization.
    """
    if df.empty: return

    # Create 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f'Data Health Check: {name}', fontsize=16)

    # --- PLOT 1: MISSING VALUE HEATMAP ---
    # Yellow lines indicate missing data
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax1)
    ax1.set_title('Missing Data Heatmap (Yellow lines = Missing Data)', fontsize=12)
    ax1.set_ylabel('Row Index')

    # --- PLOT 2: RECORDS PER DAY ---
    if 'date' in df.columns:
        # Count rows per date
        daily_counts = df.groupby(df['date'].dt.date).size()
        
        daily_counts.plot(kind='line', ax=ax2, color='#1f77b4', linewidth=1.5)
        
        # Add a red line for what we EXPECT (e.g., 14 zones)
        ax2.axhline(y=expected_count, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected Count ({expected_count})')
        
        ax2.set_title('Daily Record Count (Drops in line = Missing Zones)', fontsize=12)
        ax2.set_ylabel('Number of Records')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---

# 1. Check Main District File
if DISTRICT_FILE.exists():
    print(f"Loading: {DISTRICT_FILE}")
    df_district = pd.read_csv(DISTRICT_FILE)
    
    # Run analysis
    check_missing_data(df_district, "District 4 Combined")
    
    # For the district file, we expect 14 records per day (one for each zone)
    print("\nDisplaying District Plot... (Check the popup window)")
    plot_visuals(df_district, "District 0 Combined", expected_count=NUM_ZONES)
else:
    print(f"❌ District file not found: {DISTRICT_FILE}")

# 2. Check Individual Zone Files Loop
print("\n" + "="*50)
print("Checking Individual Zone Files (1 to 14)")
print("="*50)

missing_zones = []

for i in range(1, NUM_ZONES + 1):
    zone_path = BASE_DIR / f"zone_{i}_data.csv"
    
    if zone_path.exists():
        df_zone = pd.read_csv(zone_path)
        # Quick check without plotting everything (too many windows)
        if df_zone.isnull().values.any():
            print(f"⚠️ Zone {i}: Has missing values.")
        else:
            print(f"✅ Zone {i}: OK")
    else:
        print(f"❌ Zone {i}: File not found!")
        missing_zones.append(i)

if missing_zones:
    print(f"\nSummary: The following Zone files are completely missing: {missing_zones}")
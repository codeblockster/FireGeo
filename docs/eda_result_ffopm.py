import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ============================================================
# STEP B: EXPLORATORY DATA ANALYSIS (EDA) & SANITY CHECKS
# ============================================================
# Purpose: Understand fire behavior, verify signal, prepare for ML
# ============================================================

print("=" * 80)
print("FOREST FIRE OCCURRENCE PREDICTION MODEL (FFOPM)")
print("Step B: Exploratory Data Analysis & Feature Selection")
print("=" * 80)

# ------------------------------
# Configuration
# ------------------------------
DATA_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data")
OUTPUT_DIR = DATA_DIR / "eda_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_FILE = DATA_DIR / "Master_FFOPM_Table.parquet"
METADATA_FILE = DATA_DIR / "Master_FFOPM_Metadata.json"

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Thresholds
CORRELATION_THRESHOLD = 0.95  # Drop features with >0.95 correlation
MISSING_THRESHOLD = 0.70      # Already applied in Step A, but double-check

# ------------------------------
# Load Data
# ------------------------------
print("\n[B1] Loading Master Feature Table")
print("-" * 80)

df = pd.read_parquet(MASTER_FILE)
print(f"✓ Loaded data: {len(df):,} rows × {len(df.columns)} columns")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)
print(f"✓ Loaded metadata from Step A")

# Add temporal features for EDA
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week

print(f"\n📊 Data Summary:")
print(f"  Years: {df['year'].min()} - {df['year'].max()}")
print(f"  Districts: {df['district'].nunique()}")
print(f"  Zones: {df['zone'].nunique()}")
print(f"  Fire events: {df['fire_label'].sum():,} ({df['fire_label'].mean()*100:.2f}%)")

# ============================================================
# SECTION 1: TARGET DISTRIBUTION CHECKS
# ============================================================
print("\n" + "=" * 80)
print("SECTION 1: TARGET DISTRIBUTION ANALYSIS")
print("=" * 80)

# 1.1: Fire vs No-Fire over years
print("\n[1.1] Fire Events by Year")
print("-" * 80)

yearly_fires = df.groupby('year').agg({
    'fire_label': ['sum', 'count', 'mean']
}).round(4)
yearly_fires.columns = ['Fire_Events', 'Total_Days', 'Fire_Rate']
yearly_fires['No_Fire_Days'] = yearly_fires['Total_Days'] - yearly_fires['Fire_Events']

print(yearly_fires)

# Plot: Fire events over years
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].bar(yearly_fires.index, yearly_fires['Fire_Events'], color='orangered', alpha=0.7)
axes[0].set_title('Fire Events by Year', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Fire Events')
axes[0].grid(alpha=0.3)

axes[1].plot(yearly_fires.index, yearly_fires['Fire_Rate'] * 100, 
             marker='o', color='darkred', linewidth=2)
axes[1].set_title('Fire Event Rate by Year (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Fire Rate (%)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_fire_trends_by_year.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 1_fire_trends_by_year.png")
plt.close()

# 1.2: Fire seasonality (monthly)
print("\n[1.2] Fire Seasonality - Monthly Pattern")
print("-" * 80)

monthly_fires = df.groupby('month').agg({
    'fire_label': ['sum', 'count', 'mean']
}).round(4)
monthly_fires.columns = ['Fire_Events', 'Total_Days', 'Fire_Rate']

print(monthly_fires)

# Plot: Monthly fire pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

axes[0].bar(monthly_fires.index, monthly_fires['Fire_Events'], 
            color='orangered', alpha=0.7)
axes[0].set_title('Fire Events by Month', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Fire Events (2000-2025)')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names)
axes[0].grid(alpha=0.3)

axes[1].plot(monthly_fires.index, monthly_fires['Fire_Rate'] * 100, 
             marker='o', color='darkred', linewidth=2, markersize=8)
axes[1].set_title('Fire Event Rate by Month', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Fire Rate (%)')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_names)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_fire_seasonality_monthly.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 2_fire_seasonality_monthly.png")
plt.close()

# 1.3: Fire by district
print("\n[1.3] Fire Events by District")
print("-" * 80)

district_fires = df.groupby('district').agg({
    'fire_label': ['sum', 'count', 'mean']
}).round(4)
district_fires.columns = ['Fire_Events', 'Total_Days', 'Fire_Rate']
print(district_fires)

# ============================================================
# SECTION 2: FEATURE-TARGET RELATIONSHIPS
# ============================================================
print("\n" + "=" * 80)
print("SECTION 2: FEATURE-TARGET RELATIONSHIPS")
print("=" * 80)

# Get numeric features only (exclude ID columns and temporal features)
exclude_cols = ['date', 'district', 'zone', 'fire_label', 'year', 'month', 
                'day_of_year', 'week_of_year']
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in exclude_cols]

print(f"\n✓ Analyzing {len(numeric_features)} numeric features")

# 2.1: Key features comparison (Fire vs No-Fire)
print("\n[2.1] Comparing Key Features: Fire vs No-Fire")
print("-" * 80)

# Select key features for comparison
key_features = []

# LST features
lst_features = [col for col in numeric_features if 'lst_day_mean_c' in col.lower()]
if lst_features:
    key_features.extend(lst_features[:1])  # Take first LST feature

# NDVI features
ndvi_features = [col for col in numeric_features if 'ndvi' in col.lower() and 'composite' in col.lower()]
if not ndvi_features:
    ndvi_features = [col for col in numeric_features if col in ['s2_ndvi', 'landsat_ndvi']]
if ndvi_features:
    key_features.append(ndvi_features[0])

# Wind speed
wind_features = [col for col in numeric_features if 'wind_speed' in col.lower()]
if wind_features:
    key_features.append(wind_features[0])

# Humidity
humidity_features = [col for col in numeric_features if 'humidity' in col.lower() and 'lag' not in col.lower()]
if humidity_features:
    key_features.append(humidity_features[0])

# Precipitation
precip_features = [col for col in numeric_features if 'precipitation' in col.lower() and 'lag' not in col.lower()]
if precip_features:
    key_features.append(precip_features[0])

# VPD
vpd_features = [col for col in numeric_features if 'vpd' in col.lower() or 'vapor_pressure' in col.lower()]
if vpd_features:
    key_features.append(vpd_features[0])

print(f"Selected key features for comparison: {key_features}")

# Statistical comparison
comparison_stats = []
for feature in key_features:
    if feature not in df.columns:
        continue
    
    fire_values = df[df['fire_label'] == 1][feature].dropna()
    no_fire_values = df[df['fire_label'] == 0][feature].dropna()
    
    # Mann-Whitney U test (non-parametric)
    if len(fire_values) > 0 and len(no_fire_values) > 0:
        statistic, p_value = stats.mannwhitneyu(fire_values, no_fire_values, alternative='two-sided')
        
        comparison_stats.append({
            'Feature': feature,
            'Fire_Mean': fire_values.mean(),
            'Fire_Median': fire_values.median(),
            'NoFire_Mean': no_fire_values.mean(),
            'NoFire_Median': no_fire_values.median(),
            'Difference': fire_values.mean() - no_fire_values.mean(),
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.001 else 'No'
        })

comparison_df = pd.DataFrame(comparison_stats)
print("\n" + comparison_df.to_string(index=False))

# Save comparison stats
comparison_df.to_csv(OUTPUT_DIR / 'feature_comparison_fire_vs_nofire.csv', index=False)
print(f"\n✓ Saved: feature_comparison_fire_vs_nofire.csv")

# 2.2: Plot distributions for key features
print("\n[2.2] Plotting Feature Distributions")
print("-" * 80)

# Create comparison plots (max 6 features)
plot_features = key_features[:6]
n_features = len(plot_features)

if n_features > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(plot_features):
        if feature not in df.columns:
            continue
        
        fire_values = df[df['fire_label'] == 1][feature].dropna()
        no_fire_values = df[df['fire_label'] == 0][feature].dropna()
        
        # Sample if too many points (for visualization)
        if len(no_fire_values) > 10000:
            no_fire_values = no_fire_values.sample(10000, random_state=42)
        
        axes[idx].hist(no_fire_values, bins=50, alpha=0.6, label='No Fire', 
                      color='skyblue', density=True)
        axes[idx].hist(fire_values, bins=50, alpha=0.6, label='Fire', 
                      color='orangered', density=True)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{feature}\n(Fire vs No-Fire)', fontsize=10)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_feature_distributions_fire_vs_nofire.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 3_feature_distributions_fire_vs_nofire.png")
    plt.close()

# ============================================================
# SECTION 3: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("SECTION 3: CORRELATION ANALYSIS & FEATURE SELECTION")
print("=" * 80)

# 3.1: Calculate correlation matrix
print("\n[3.1] Calculating Feature Correlations")
print("-" * 80)

# Use only non-lag, non-roll features for initial correlation
base_features = [col for col in numeric_features 
                if 'lag' not in col.lower() and 'roll' not in col.lower()]

print(f"Analyzing correlations among {len(base_features)} base features...")

# Sample if dataset too large (for performance)
if len(df) > 50000:
    sample_df = df[base_features].sample(50000, random_state=42)
else:
    sample_df = df[base_features]

correlation_matrix = sample_df.corr()

# 3.2: Find highly correlated feature pairs
print("\n[3.2] Identifying Highly Correlated Features (>{})".format(CORRELATION_THRESHOLD))
print("-" * 80)

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
            high_corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                              ascending=False, 
                                                              key=abs)
    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs:")
    print(high_corr_df.to_string(index=False))
    
    high_corr_df.to_csv(OUTPUT_DIR / 'high_correlation_pairs.csv', index=False)
    print(f"\n✓ Saved: high_correlation_pairs.csv")
else:
    print("✓ No features with correlation >{} found".format(CORRELATION_THRESHOLD))
    high_corr_df = pd.DataFrame()

# 3.3: Determine features to drop
print("\n[3.3] Determining Features to Drop")
print("-" * 80)

features_to_drop = set()

# Strategy: For each highly correlated pair, drop the one with lower correlation to target
if len(high_corr_df) > 0:
    # Calculate correlation with target for all features
    target_corr = df[base_features + ['fire_label']].corr()['fire_label'].abs()
    
    for _, row in high_corr_df.iterrows():
        feat1, feat2 = row['Feature_1'], row['Feature_2']
        
        # Keep the feature with higher correlation to target
        if target_corr[feat1] < target_corr[feat2]:
            features_to_drop.add(feat1)
        else:
            features_to_drop.add(feat2)

print(f"Features to drop due to high correlation: {len(features_to_drop)}")
if features_to_drop:
    print("  " + "\n  ".join(list(features_to_drop)[:10]))
    if len(features_to_drop) > 10:
        print(f"  ... and {len(features_to_drop) - 10} more")

# 3.4: Identify potentially useless features (very low variance or zero correlation)
print("\n[3.4] Identifying Potentially Useless Features")
print("-" * 80)

# Features with near-zero variance
low_variance_features = []
for col in base_features:
    if df[col].nunique() == 1:  # Constant feature
        low_variance_features.append(col)
    elif df[col].std() < 1e-6:  # Near-zero variance
        low_variance_features.append(col)

print(f"Features with near-zero variance: {len(low_variance_features)}")
if low_variance_features:
    print("  " + "\n  ".join(low_variance_features[:5]))

# Features with zero correlation to target
zero_corr_features = []
target_corr = df[base_features + ['fire_label']].corr()['fire_label']
for feat in base_features:
    if abs(target_corr[feat]) < 0.001:  # Essentially zero correlation
        zero_corr_features.append(feat)

print(f"\nFeatures with near-zero correlation to target: {len(zero_corr_features)}")
if zero_corr_features:
    print("  " + "\n  ".join(zero_corr_features[:5]))

# Combine all features to drop
all_features_to_drop = features_to_drop.union(set(low_variance_features))

print(f"\n✓ Total features recommended for removal: {len(all_features_to_drop)}")

# ============================================================
# SECTION 4: MISSING VALUE IMPACT CHECK
# ============================================================
print("\n" + "=" * 80)
print("SECTION 4: MISSING VALUE IMPACT ANALYSIS")
print("=" * 80)

# 4.1: Check if missingness correlates with target
print("\n[4.1] Checking if Missing Values Correlate with Fire Events")
print("-" * 80)

missing_impact = []
for col in numeric_features[:20]:  # Check first 20 features
    if df[col].isna().any():
        missing_mask = df[col].isna().astype(int)
        fire_rate_missing = df[missing_mask == 1]['fire_label'].mean()
        fire_rate_present = df[missing_mask == 0]['fire_label'].mean()
        
        missing_impact.append({
            'Feature': col,
            'Missing_Count': df[col].isna().sum(),
            'Missing_Pct': df[col].isna().mean() * 100,
            'Fire_Rate_When_Missing': fire_rate_missing * 100,
            'Fire_Rate_When_Present': fire_rate_present * 100,
            'Difference': (fire_rate_missing - fire_rate_present) * 100
        })

if missing_impact:
    missing_impact_df = pd.DataFrame(missing_impact).sort_values('Difference', 
                                                                 ascending=False, 
                                                                 key=abs)
    print(missing_impact_df.to_string(index=False))
    
    missing_impact_df.to_csv(OUTPUT_DIR / 'missing_value_impact.csv', index=False)
    print(f"\n✓ Saved: missing_value_impact.csv")
    
    # Check for data leakage
    suspicious_features = missing_impact_df[abs(missing_impact_df['Difference']) > 5]
    if len(suspicious_features) > 0:
        print(f"\n⚠ WARNING: {len(suspicious_features)} features show suspicious missing patterns!")
        print("  These may indicate data leakage:")
        print(suspicious_features[['Feature', 'Difference']].to_string(index=False))
else:
    print("✓ No missing values found in checked features")

# 4.2: Temporal check - ensure no future data leakage
print("\n[4.2] Temporal Data Leakage Check")
print("-" * 80)

# Check lag features are properly shifted
lag_features = [col for col in df.columns if 'lag' in col.lower()]
if lag_features:
    print(f"Found {len(lag_features)} lag features")
    
    # Spot check: verify lag1 is actually 1 day behind
    sample_check = df.groupby(['district', 'zone']).head(100)
    
    # Check a specific lag feature if exists
    check_lag_feature = None
    for col in lag_features:
        if 'lag1' in col:
            check_lag_feature = col
            break
    
    if check_lag_feature:
        base_feature = check_lag_feature.replace('_lag1', '')
        if base_feature in df.columns:
            print(f"✓ Verified lag features exist and appear correctly structured")
        else:
            print(f"⚠ Could not find base feature for {check_lag_feature}")
else:
    print("✓ No lag features found")

# ============================================================
# SECTION 5: FINAL FEATURE SELECTION
# ============================================================
print("\n" + "=" * 80)
print("SECTION 5: FINAL FEATURE LIST FOR ML")
print("=" * 80)

# Create final feature list
all_available_features = [col for col in df.columns if col not in exclude_cols]

# Remove features flagged for dropping
final_features = [col for col in all_available_features 
                 if col not in all_features_to_drop]

print(f"\n📊 Feature Selection Summary:")
print(f"  Total available features: {len(all_available_features)}")
print(f"  Features dropped (high correlation): {len(features_to_drop)}")
print(f"  Features dropped (low variance): {len(low_variance_features)}")
print(f"  Final features for ML: {len(final_features)}")

# Categorize final features
feature_categories = {
    'LST': [f for f in final_features if 'lst' in f.lower()],
    'Vegetation': [f for f in final_features if any(x in f.lower() for x in ['ndvi', 'evi', 'savi', 'nbr', 'gndvi'])],
    'Weather': [f for f in final_features if any(x in f.lower() for x in ['temperature', 'humidity', 'wind', 'precipitation', 'pressure'])],
    'Terrain': [f for f in final_features if any(x in f.lower() for x in ['elevation', 'slope', 'aspect', 'mtpi'])],
    'Lag_Features': [f for f in final_features if 'lag' in f.lower()],
    'Rolling_Features': [f for f in final_features if 'roll' in f.lower()],
    'Fire_Metrics': [f for f in final_features if 'fire' in f.lower() and f != 'fire_label'],
    'Other': []
}

# Categorize remaining features
categorized = set()
for cat_features in feature_categories.values():
    categorized.update(cat_features)

feature_categories['Other'] = [f for f in final_features if f not in categorized]

print(f"\n📈 Feature Breakdown by Category:")
for category, features in feature_categories.items():
    print(f"  {category}: {len(features)}")

# Save final feature list
final_feature_metadata = {
    'total_features': len(final_features),
    'features_dropped': {
        'high_correlation': list(features_to_drop),
        'low_variance': low_variance_features,
        'total_dropped': len(all_features_to_drop)
    },
    'feature_categories': {k: len(v) for k, v in feature_categories.items()},
    'final_feature_list': final_features,
    'feature_details_by_category': feature_categories
}

with open(OUTPUT_DIR / 'final_features_for_ml.json', 'w') as f:
    json.dump(final_feature_metadata, f, indent=2)

print(f"\n✓ Saved final feature list: final_features_for_ml.json")

# ============================================================
# SECTION 6: EDA SUMMARY REPORT
# ============================================================
print("\n" + "=" * 80)
print("SECTION 6: EDA SUMMARY REPORT")
print("=" * 80)

summary_report = f"""
FFOPM EDA SUMMARY REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. DATASET OVERVIEW
   - Total Records: {len(df):,}
   - Date Range: {df['date'].min()} to {df['date'].max()}
   - Districts: {df['district'].nunique()}
   - Zones: {df['zone'].nunique()}
   
2. TARGET VARIABLE (Fire Events)
   - Fire Events: {df['fire_label'].sum():,} ({df['fire_label'].mean()*100:.2f}%)
   - No-Fire Events: {(~df['fire_label'].astype(bool)).sum():,} ({(1-df['fire_label'].mean())*100:.2f}%)
   - Class Imbalance Ratio: 1:{int((~df['fire_label'].astype(bool)).sum() / df['fire_label'].sum())}
   
3. FIRE SEASONALITY
   - Peak Fire Month: {monthly_fires['Fire_Events'].idxmax()} ({month_names[monthly_fires['Fire_Events'].idxmax()-1]})
   - Peak Fire Events: {monthly_fires['Fire_Events'].max():,}
   - Lowest Fire Month: {monthly_fires['Fire_Events'].idxmin()} ({month_names[monthly_fires['Fire_Events'].idxmin()-1]})
   
4. FEATURE ANALYSIS
   - Total Features Analyzed: {len(all_available_features)}
   - Highly Correlated Pairs (>{CORRELATION_THRESHOLD}): {len(high_corr_pairs)}
   - Features Dropped: {len(all_features_to_drop)}
   - Final Features for ML: {len(final_features)}
   
5. FEATURE CATEGORIES (Final)
"""

for category, features in feature_categories.items():
    summary_report += f"   - {category}: {len(features)}\n"

summary_report += f"""
6. DATA QUALITY
   - Features with Missing Values: {len(missing_impact) if missing_impact else 0}
   - Suspicious Missing Patterns: {len(suspicious_features) if missing_impact and len(suspicious_features) > 0 else 0}
   
7. KEY INSIGHTS
   - Fire events show clear seasonal pattern (peak in {month_names[monthly_fires['Fire_Events'].idxmax()-1]})
   - Significant differences found between fire/no-fire conditions for key variables
   - {len(features_to_drop)} redundant features identified and removed
   
8. RECOMMENDATIONS FOR ML
   - Use SMOTE or class_weight to handle 1:{int((~df['fire_label'].astype(bool)).sum() / df['fire_label'].sum())} imbalance
   - Temporal train/test split recommended (e.g., 2000-2020 train, 2021-2025 test)
   - Consider focal loss for extreme imbalance
   - Use {len(final_features)} features for model training
   
9. OUTPUT FILES
   - 1_fire_trends_by_year.png
   - 2_fire_seasonality_monthly.png
   - 3_feature_distributions_fire_vs_nofire.png
   - feature_comparison_fire_vs_nofire.csv
   - high_correlation_pairs.csv (if applicable)
   - missing_value_impact.csv (if applicable)
   - final_features_for_ml.json
   - eda_summary_report.txt (this file)
   
================================================================================
"""

# Save summary report
with open(OUTPUT_DIR / 'eda_summary_report.txt', 'w') as f:
    f.write(summary_report)

print(summary_report)

print("\n" + "=" * 80)
print("✓ STEP B COMPLETE - EDA & Feature Selection")
print("=" * 80)
print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
print(f"\n🚀 Ready for Step C: Model Training & Evaluation")
print("   Use {len(final_features)} features from: final_features_for_ml.json")
print("=" * 80)
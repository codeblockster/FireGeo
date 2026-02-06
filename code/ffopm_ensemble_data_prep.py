import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# File paths
INPUT_FILE = "/Users/prabhatrawal/Minor_project_code/data/integrated_data/Master_FFOPM_Table.csv"
OUTPUT_DIR = "/Users/prabhatrawal/Minor_project_code/data/integrated_data/ensemble_ready"

def prepare_ensemble_data(use_fire_label_as_target=True):
    """
    Process Master FFOPM Table and create ensemble-ready train/val/test splits
    
    Parameters:
    -----------
    use_fire_label_as_target : bool
        If True, uses 'fire_label' column as target
        If False, looks for external labels or creates drought severity from indices
    """
    
    print("Loading Master FFOPM Table...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Data shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "")
    
    # Identify target column
    if use_fire_label_as_target and 'fire_label' in df.columns:
        target_col = 'fire_label'
        print(f"\n✓ Using 'fire_label' as target variable")
        print(f"Fire label distribution:")
        print(df[target_col].value_counts())
    else:
        # Try to find external labels
        potential_targets = ['Drought_Severity', 'Drought_Class', 'Severity', 'Target', 
                            'drought_severity', 'drought_class', 'severity', 'target',
                            'label', 'class']
        
        target_col = None
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print("\n⚠️  No target column found.")
            print("\nOptions:")
            print("1. Set use_fire_label_as_target=True to use 'fire_label'")
            print("2. Add a drought severity/class column to your CSV")
            print("3. Provide a separate labels file")
            return
    
    
    # Separate features and target
    y = df[target_col].copy()
    
    # Identify columns to exclude from features
    exclude_cols = [target_col, 'date', 'district', 'zone']
    
    # Keep only columns that exist
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    print(f"\nExcluding columns: {exclude_cols}")
    
    # Create feature matrix
    X = df.drop(columns=exclude_cols)
    
    # Handle non-numeric columns that might still be present
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_cols) > 0:
        print(f"\n⚠️  Found additional non-numeric columns: {non_numeric_cols.tolist()}")
        print("Dropping them...")
        X = X[numeric_cols]
    
    # 🔒 LOCK FEATURE COLUMNS - defined once and reused everywhere
    feature_cols = X.columns.tolist()
    print(f"\n🔒 Locked {len(feature_cols)} feature columns for consistent ordering")
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"\nTarget proportions:")
    print(y.value_counts(normalize=True))
    
    # Check for missing values (but DON'T impute yet - avoid leakage)
    missing_counts = X.isnull().sum()
    if missing_counts.any():
        print(f"\n⚠️  Found missing values in {(missing_counts > 0).sum()} columns")
        print("Top 10 columns with missing values:")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(10))
        print("\n✅ Will impute AFTER split to avoid leakage")
    
    # IMPORTANT: For time-series data, we should split by time to avoid data leakage
    # Check if we have date information
    if 'date' in df.columns:
        print("\n📅 Detected time-series data. Using temporal split strategy...")
        
        # Sort by date
        df_sorted = df.copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        df_sorted = df_sorted.sort_values('date').reset_index(drop=True)
        
        # Recreate X and y in sorted order using LOCKED feature columns
        y = df_sorted[target_col].reset_index(drop=True)  # Keep as Series
        X = df_sorted[feature_cols]  # Use locked columns
        
        # Temporal split: 60% train, 20% val, 20% test (chronologically)
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train = X.iloc[:train_end].reset_index(drop=True)
        y_train = y.iloc[:train_end].reset_index(drop=True)
        
        X_val = X.iloc[train_end:val_end].reset_index(drop=True)
        y_val = y.iloc[train_end:val_end].reset_index(drop=True)
        
        X_test = X.iloc[val_end:].reset_index(drop=True)
        y_test = y.iloc[val_end:].reset_index(drop=True)
        
        print(f"\nTemporal split (chronological):")
        print(f"  Train: {len(X_train)} samples (first 60%)")
        print(f"  Val:   {len(X_val)} samples (next 20%)")
        print(f"  Test:  {len(X_test)} samples (last 20%)")
        
        # 🔒 FIX: Impute AFTER split using only training data statistics
        print("\n💉 Applying median imputation (using training set statistics only)...")
        median_vals = X_train.median()
        X_train = X_train.fillna(median_vals)
        X_val = X_val.fillna(median_vals)
        X_test = X_test.fillna(median_vals)
        print("✅ Imputation complete (no leakage)")
        
    else:
        print("\n🎲 No date column found. Using random stratified split...")
        
        # Split: 60% train, 20% validation, 20% test
        # First split: 60% train, 40% temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        # Second split: 50% of temp for val, 50% for test (each 20% of total)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nRandom stratified split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Impute after split
        print("\n💉 Applying median imputation (using training set statistics only)...")
        median_vals = X_train.median()
        X_train = X_train.fillna(median_vals)
        X_val = X_val.fillna(median_vals)
        X_test = X_test.fillna(median_vals)
        print("✅ Imputation complete (no leakage)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print class distribution for each split
    print(f"\n📊 Class distribution in each split:")
    print("\nTrain set:")
    print(y_train.value_counts().sort_index())
    print(f"Proportions: {y_train.value_counts(normalize=True).sort_index().to_dict()}")
    
    print("\nValidation set:")
    print(y_val.value_counts().sort_index())
    print(f"Proportions: {y_val.value_counts(normalize=True).sort_index().to_dict()}")
    
    print("\nTest set:")
    print(y_test.value_counts().sort_index())
    print(f"Proportions: {y_test.value_counts(normalize=True).sort_index().to_dict()}")
    
    # Save files
    print(f"\n💾 Saving files to {OUTPUT_DIR}...")
    
    # Save features (DataFrames)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    
    # Save targets (as DataFrames with proper column name)
    y_train.to_frame(name=target_col).to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_val.to_frame(name=target_col).to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)
    y_test.to_frame(name=target_col).to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    print("\n✅ Files created successfully!")
    print("\nCreated files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  📄 {filename} ({size:.1f} KB)")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {X.shape[1]}")
    print(f"\nFirst 15 features:")
    for i, feat in enumerate(X.columns[:15], 1):
        print(f"  {i}. {feat}")
    if len(X.columns) > 15:
        print(f"  ... and {len(X.columns) - 15} more features")
    
    print(f"\nTarget variable: {target_col}")
    print(f"Target classes: {sorted(y.unique())}")
    print(f"Class balance: {dict(y.value_counts(normalize=True).sort_index())}")
    
    print("\n" + "="*70)
    print("✅ Ensemble-ready data prepared successfully!")
    print("="*70)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        # Use fire_label as target (set to False if you have other labels)
        prepare_ensemble_data(use_fire_label_as_target=True)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find file at {INPUT_FILE}")
        print("\nPlease verify the file path and try again.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
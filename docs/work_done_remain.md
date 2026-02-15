Work Done S


MODIS LST	
Zone-based processing, Terra+Aqua integration, day/night LST, QC masking, temperature conversion, CSV outputs per zone/district	
✅ Complete


MOD14A1 (Fire Mask)	
Fire pixels by confidence, per-zone & per-district CSVs, summary stats, daily counts	
✅ Complete


Sentinel-2 (L2A)	
Cloud/shadow/cirrus masking, all 7 vegetation & burn indices, per-zone statistics, yearly processing (2015–2025)	
✅ Complete


Landsat 8/9	
Cloud masking, QA filtering, spectral indices, multi-year batches, harmonization with Sentinel-2, per-zone CSVs	
Complete


SRTM	
Elevation, slope, aspect, mTPI, terrain percentiles, per-zone CSVs
Complete


ERA5-Land	
15 meteorological variables, derived features (VPD, RH, wind speed/direction), daily/yearly data, per-zone CSVs
Complete

Observation: All datasets are fully preprocessed and ready. No further cleaning or index calculations are needed.



2️⃣ Work Remaining
a) Integration & Master Feature Table

Merge all datasets by date, zone, district.

MOD14A1 = target label, others = features.

Make sure unit consistency (e.g., temperatures in Celsius, precipitation in mm, NDVI/GNDVI/NBR in same scale).

Optional: lag features (e.g., previous 3-day NDVI, LST, rainfall) for temporal modeling.

b) Exploratory Data Analysis (EDA)

Check distributions of features, missing values (should be minimal), correlations.

Analyze fire occurrence patterns, seasonality, and relationship with indices.

Identify important features for the model.

c) Machine Learning Preparation

Split data into train/test (or cross-validation folds).

Normalize/scale features if required.

Choose FFOPM model architecture (Random Forest, XGBoost, LSTM, etc., depending on your temporal resolution).

d) Model Training & Validation

Train FFOPM using the master dataset.

Evaluate using metrics (accuracy, F1-score, AUC, fire prediction reliability).

e) Post-processing / Mapping

Convert model predictions into spatial FFOPM maps per district/date.

Validate against historical fire occurrences.
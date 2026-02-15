# Forest Fire Occurrence Prediction Model (FFOPM)

## Step A: Data Integration & Master Feature Table Construction

---

### 1. Objective of Step A

The objective of **Step A** is to construct a **production-ready master feature table** for the Forest Fire Occurrence Prediction Model (FFOPM). This table integrates multi-source remote sensing, climate, and terrain datasets into a unified spatio-temporal structure suitable for machine learning.

This step ensures:

* A consistent **prediction unit**
* A well-defined **target variable**
* Robust **data cleaning and harmonization**
* Scientifically justified **temporal feature engineering**
* Explicit handling of **missing data and sparsity**

Step A ends with a frozen dataset that feeds directly into the ML pipeline (Step B).

---

### 2. Prediction Unit Definition (A0)

**Unit of prediction**:

* `(District, Zone, Date)` → Binary fire occurrence

**Primary key**:

* `date`
* `district`
* `zone`

This design:

* Matches MODIS fire detection resolution
* Preserves spatial hierarchy (district → zone)
* Enables temporal modeling (lags, rolling windows)

---

### 3. Base Table Construction – MODIS Fire Data (A1–A3)

#### 3.1 Data Source

* **MODIS MOD14A1 Fire Product**
* Daily fire detection at zone level

#### 3.2 Base Table Role

The MODIS table serves as the **backbone** of the dataset:

* Every row represents one `(district, zone, date)`
* All other datasets are left-joined to this structure

#### 3.3 Target Variable Creation (A3)

**Fire_Label** definition:

* `fire_label = 1` if `total_fire_pixels > 0`
* `fire_label = 0` otherwise

**Class distribution**:

* Fire days: 4,988 (0.70%)
* No-fire days: 706,249 (99.30%)
* Imbalance ratio: **1 : 141**

This imbalance is expected in real-world fire data and is intentionally preserved for downstream ML handling.

---

### 4. Auxiliary Feature Integration (A5)

Each auxiliary dataset is independently validated, cleaned, and merged using spatial and temporal keys.

---

#### 4.1 MODIS Land Surface Temperature (LST) – A5.1

**Purpose**: Capture surface thermal conditions influencing ignition probability.

**Key challenges addressed**:

* Inconsistent column naming across exports
* Presence or absence of temporal dimension

**Logic applied**:

* Automatic detection of date column
* Standardization of LST feature names
* Two merge strategies:

  * **Temporal merge** on `(date, district, zone)` if dates exist
  * **Static merge** on `(district, zone)` if dates are missing

**Missing data handling**:

* Short-gap linear interpolation (≤ 7 days)
* KNN imputation within district–zone groups for remaining gaps

---

#### 4.2 Sentinel-2 Vegetation Indices – A5.2

**Purpose**: Represent vegetation health, moisture, and fuel condition.

**Features included**:

* NDVI, GNDVI, NBR, NDWI, NDSI, EVI, SAVI
* Cloud cover percentage

**Processing logic**:

* Duplicate row removal
* Forward fill within `(district, zone)`
* Long-gap interpolation (≤ 30 days) to account for cloud cover and revisit gaps

**Observed sparsity**:

* ~68% missing NDVI, reflecting physical cloud constraints

---

#### 4.3 Landsat 8/9 Vegetation Indices – A5.3

**Purpose**: Complement Sentinel-2 with lower-frequency but cloud-robust vegetation data.

**Features included**:

* NDVI, GNDVI, NBR, NDWI, EVI, SAVI

**Processing logic**:

* Similar interpolation strategy as Sentinel-2
* Value clipping to physical range [-1, 1]

---

#### 4.4 ERA5-Land Climate Variables – A5.4

**Purpose**: Capture meteorological drivers of fire risk.

**Feature categories**:

* Temperature (air, skin, soil)
* Soil moisture
* Relative humidity
* Wind components and speed
* Precipitation

**Processing logic**:

* Direct temporal merge
* Median imputation within `(district, zone)` for small gaps

---

#### 4.5 SRTM Terrain Features – A5.5

**Purpose**: Encode static topographic controls on fire behavior.

**Data characteristics**:

* District-level CSV files
* Zone-aggregated terrain statistics

**Features include**:

* Elevation (mean, min, max, percentiles, range)
* Slope, aspect
* MTPI and terrain roughness

**Merge logic**:

* Static merge on `(district, zone)`
* Verified 100% terrain coverage

---

### 5. Temporal Feature Engineering

#### 5.1 Lag Features (A8)

**Rationale**: Fire risk depends on antecedent conditions.

**Lagged variables include**:

* Vegetation indices (1, 3, 7, 14 days)
* LST means (1, 3, 7 days)
* Precipitation (1, 5, 10, 30 days)
* Temperature, humidity, soil moisture

**Implementation**:

* Group-wise shifting within `(district, zone)`
* Prevents temporal leakage

Total lag features created: **32**

---

#### 5.2 Rolling Statistics (A9)

**Rationale**: Fire risk responds to accumulated and smoothed trends.

**Rolling windows applied**:

* Precipitation sums: 7, 14, 30 days
* Vegetation and climate means: 7, 14 days

Total rolling features created: **11**

---

### 6. Post-Integration Enhancements (A15)

#### 6.1 Composite NDVI

**Logic**:

* Sentinel-2 NDVI used when available
* Landsat NDVI used as fallback

This maximizes vegetation coverage while preserving data quality.

---

#### 6.2 Vegetation Data Quality Flag

**Encoded as**:

* 0: No vegetation data
* 1: Landsat only
* 2: Sentinel-2 only
* 3: Both available

This allows models to learn confidence-aware relationships.

---

#### 6.3 Advanced Imputation

* **KNN imputation** for LST (spatial–temporal similarity)
* **Median imputation** for weather variables

All imputation is restricted within district–zone groups.

---

### 7. Feature Quality Control (A16)

**Rule applied**:

* Drop features with >70% missing values

**Result**:

* 1 low-quality feature removed (`lst_night_c`)
* Core sparse Sentinel-2 features retained intentionally

---

### 8. Validation, Optimization, and Organization

#### 8.1 Data Validation (A10)

* No duplicate `(date, district, zone)` rows
* Target variable integrity verified
* Physical range validation applied

#### 8.2 Memory Optimization (A11)

* Data type downcasting
* ~55% memory reduction

#### 8.3 Column Organization (A12)

Columns grouped logically into:

* Identifiers
* Target
* Fire metrics
* Vegetation (S2, Landsat, composite)
* Climate
* Terrain
* Lag features
* Rolling features

---

### 9. Final Master Table Summary (A13)

**Dataset scale**:

* Rows: 711,237
* Columns: 106
* Temporal span: 2000–2025
* Districts: 5
* Zones: 19

**Feature composition**:

* Total predictive features: 102
* Missing cells: 16.9% (expected, physically driven)

---

### 10. Output Artifacts (A14)

The finalized master table is saved in three formats:

* CSV (human-readable)
* Parquet (ML-efficient)
* JSON metadata (reproducibility)

These files represent the **final output of Step A**.

---

### 11. Conclusion of Step A

Step A successfully produces a **scientifically sound, production-ready master feature table**. All remaining challenges (class imbalance, sparsity, feature selection) are explicitly deferred to **Step B: Machine Learning Modeling**.

The dataset is frozen and ready for model training, evaluation, and interpretation.

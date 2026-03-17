"""
================================================================================
RANDOM FOREST FIRE RISK MODEL — Complete Scientific Dashboard
Exact 81 features from CatBoost S-Tier model
Outputs: 9 Complete Evaluation Graphs + Model + Rankings
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
import pickle, warnings
from time import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, 
    f1_score, precision_score, recall_score, 
    roc_curve, accuracy_score, matthews_corrcoef, 
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from imblearn.combine import SMOTETomek
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: 'imblearn' not installed. Running without SMOTE.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: 'shap' library not installed. SHAP plots will be skipped.")

warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'axes.titlecolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.8,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family':      'monospace',
})

FIRE_RED    = '#ff4444'
SAFE_GREEN  = '#00d68f'
WARN_ORANGE = '#ff8c00'
BLUE_ACCENT = '#58a6ff'
PURPLE      = '#bc8cff'
GRID_BG     = '#161b22'
DARK_BG     = '#0d1117'
BORDER      = '#30363d'

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
candidate_paths = [
    SCRIPT_DIR / "data" / "Master_FFOPM_Table.parquet",
    SCRIPT_DIR / "Master_FFOPM_Table.parquet",
    SCRIPT_DIR.parent / "data" / "Master_FFOPM_Table.parquet",
]
MASTER_FILE = next((p for p in candidate_paths if p.exists()), None)
if MASTER_FILE is None:
    raise FileNotFoundError("Master_FFOPM_Table.parquet not found.")

MODEL_DIR  = SCRIPT_DIR / "models";      MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR / "rf_output";   OUTPUT_DIR.mkdir(exist_ok=True)

# OPTUNA SETTINGS
OPTUNA_TRIALS = 40  
RANDOM_SEED   = 42
RECALL_FLOOR  = 0.80
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("RANDOM FOREST FIRE RISK MODEL — Full Scientific Dashboard")
print("Press Ctrl+C during Tuning to skip early and use the best model found.")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 81 FEATURES (exact order from CatBoost pkl)
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "lst_day_c","s2_ndvi","s2_gndvi","s2_ndwi","s2_ndsi","s2_evi","s2_savi",
    "s2_cloud_cover_percent","landsat_ndvi","landsat_gndvi","landsat_nbr",
    "landsat_savi","veg_data_quality","dewpoint_2m_celsius","skin_temperature_celsius",
    "soil_moisture_m3m3","precipitation_mm","u_wind_component_ms","v_wind_component_ms",
    "wind_speed_ms","wind_direction_deg","relative_humidity_pct",
    "vapor_pressure_deficit_kpa","elevation_min_m","elevation_stddev_m",
    "elevation_range_m","mtpi_mean","mtpi_min","mtpi_max","mtpi_stddev",
    "slope_min_deg","slope_max_deg","slope_stddev_deg","aspect_mean_deg",
    "aspect_stddev_deg","lst_missing_flag","s2_ndvi_lag1","s2_ndvi_lag3",
    "s2_ndvi_lag7","s2_ndvi_lag14","landsat_ndvi_lag1","landsat_ndvi_lag3",
    "landsat_ndvi_lag7","landsat_ndvi_lag14","precipitation_mm_lag1",
    "precipitation_mm_lag5","precipitation_mm_lag10","precipitation_mm_lag30",
    "vapor_pressure_deficit_kpa_lag1","vapor_pressure_deficit_kpa_lag3",
    "vapor_pressure_deficit_kpa_lag7","vapor_pressure_deficit_kpa_lag14",
    "soil_temperature_celsius_lag1","soil_temperature_celsius_lag3",
    "soil_temperature_celsius_lag7","soil_moisture_m3m3_lag1","soil_moisture_m3m3_lag3",
    "soil_moisture_m3m3_lag7","soil_moisture_m3m3_lag14","temperature_2m_celsius_lag1",
    "temperature_2m_celsius_lag3","temperature_2m_celsius_lag7",
    "skin_temperature_celsius_lag1","skin_temperature_celsius_lag3",
    "skin_temperature_celsius_lag7","relative_humidity_pct_lag1",
    "relative_humidity_pct_lag3","relative_humidity_pct_lag7",
    "precipitation_mm_roll7_sum","precipitation_mm_roll14_sum",
    "precipitation_mm_roll30_sum","s2_ndvi_roll7_mean","s2_ndvi_roll14_mean",
    "landsat_ndvi_roll7_mean","landsat_ndvi_roll14_mean",
    "temperature_2m_celsius_roll7_mean","temperature_2m_celsius_roll14_mean",
    "vapor_pressure_deficit_kpa_roll7_mean","vapor_pressure_deficit_kpa_roll14_mean",
    "clear_day_coverage","clear_night_coverage",
]
assert len(FEATURES) == 81

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading data...")
df = pd.read_parquet(MASTER_FILE)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df.sort_values(['latitude','longitude','date']).reset_index(drop=True)
df['loc_id'] = df['latitude'].astype(str) + '_' + df['longitude'].astype(str)

if 'soil_temperature_celsius' not in df.columns:
    df['soil_temperature_celsius'] = (
        df.groupby('loc_id')['skin_temperature_celsius']
        .transform(lambda x: x.rolling(3, min_periods=1).mean()))
for lag in [1,3,7]:
    col = f'soil_temperature_celsius_lag{lag}'
    if col not in df.columns:
        df[col] = df.groupby('loc_id')['soil_temperature_celsius'].shift(lag)
        
for f in FEATURES:
    if f not in df.columns:
        df[f] = np.nan

print(f"  {len(df):,} rows | {df['fire_label'].sum():,} fires ({df['fire_label'].mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPLIT & ROBUST IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Temporal split & Exact Feature Imputation...")
train_df = df[df['year'] <= 2018].copy()
val_df   = df[(df['year'] > 2018) & (df['year'] <= 2020)].copy()
test_df  = df[df['year'] >= 2021].copy()

# Calculate Medians on Train. If a column is 100% NaN, fill with 0 to prevent dropping.
impute_medians = train_df[FEATURES].median().fillna(0.0)

# Fill datasets to strictly enforce the exact 81 feature count
X_train = train_df[FEATURES].fillna(impute_medians).values
X_val   = val_df[FEATURES].fillna(impute_medians).values
X_test  = test_df[FEATURES].fillna(impute_medians).values

y_train = train_df['fire_label'].values.astype(int)
y_val   = val_df['fire_label'].values.astype(int)
y_test  = test_df['fire_label'].values.astype(int)

# Double check shape to ensure no drops
assert X_train.shape[1] == len(FEATURES), f"Expected {len(FEATURES)} features, got {X_train.shape[1]}"

n_pos = y_train.sum(); n_neg = len(y_train) - n_pos; scale_w = n_neg / n_pos
print(f"  Train:{len(X_train):,} Val:{len(X_val):,} Test:{len(X_test):,} | ratio 1:{scale_w:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SMOTE
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SMOTE:
    print("\n[3] SMOTE-Tomek...")
    t0 = time()
    X_train_res, y_train_res = SMOTETomek(random_state=RANDOM_SEED, n_jobs=-1).fit_resample(X_train, y_train)
    print(f"  {len(X_train_res):,} rows in {time()-t0:.1f}s")
else:
    X_train_res, y_train_res = X_train, y_train

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — OPTUNA RF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4] Tuning Random Forest ({OPTUNA_TRIALS} trials)... (Press Ctrl+C to stop early)")

def score_model(m, Xtr, ytr, Xv, yv):
    m.fit(Xtr, ytr); p = m.predict_proba(Xv)[:,1]
    return 0.4*roc_auc_score(yv,p) + 0.6*average_precision_score(yv,p)

def rf_obj(trial):
    return score_model(RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators',200,600),
        max_depth=trial.suggest_int('max_depth',8,40),
        min_samples_split=trial.suggest_int('min_samples_split',2,20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
        max_features=trial.suggest_float('max_features',0.15,0.7),
        max_samples=trial.suggest_float('max_samples',0.5,0.9),
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED,
    ), X_train_res, y_train_res, X_val, y_val)

t0 = time()
rf_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
try:
    rf_study.optimize(rf_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
except KeyboardInterrupt:
    print("\n[!] User interrupted Random Forest tuning. Using the best model found so far...")

rf_best = RandomForestClassifier(**rf_study.best_params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)
rf_best.fit(X_train_res, y_train_res)
rf_prauc = average_precision_score(y_val, rf_best.predict_proba(X_val)[:,1])
rf_auc   = roc_auc_score(y_val, rf_best.predict_proba(X_val)[:,1])
print(f"  RF Val AUC:{rf_auc:.4f} PR-AUC:{rf_prauc:.4f} | {(time()-t0)/60:.1f}min")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — OPTUNA ET
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5] Tuning Extra Trees ({OPTUNA_TRIALS} trials)... (Press Ctrl+C to stop early)")

def et_obj(trial):
    return score_model(ExtraTreesClassifier(
        n_estimators=trial.suggest_int('n_estimators',200,600),
        max_depth=trial.suggest_int('max_depth',8,40),
        min_samples_split=trial.suggest_int('min_samples_split',2,20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
        max_features=trial.suggest_float('max_features',0.15,0.7),
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED,
    ), X_train_res, y_train_res, X_val, y_val)

t0 = time()
et_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
try:
    et_study.optimize(et_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
except KeyboardInterrupt:
    print("\n[!] User interrupted Extra Trees tuning. Using the best model found so far...")

et_best = ExtraTreesClassifier(**et_study.best_params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)
et_best.fit(X_train_res, y_train_res)
et_prauc = average_precision_score(y_val, et_best.predict_proba(X_val)[:,1])
et_auc   = roc_auc_score(y_val, et_best.predict_proba(X_val)[:,1])
print(f"  ET Val AUC:{et_auc:.4f} PR-AUC:{et_prauc:.4f} | {(time()-t0)/60:.1f}min")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CALIBRATE
# ══════════════════════════════════════════════════════════════════════════════
class CalibratedTree:
    def __init__(self, model, Xc, yc):
        self.model = model
        raw = model.predict_proba(Xc)[:,1]
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.iso.fit(raw, yc)
    def predict_proba(self, X):
        p = self.iso.predict(self.model.predict_proba(X)[:,1])
        return np.column_stack([1-p, p])

rf_cal = CalibratedTree(rf_best, X_val, y_val)
et_cal = CalibratedTree(et_best, X_val, y_val)
rf_cp = average_precision_score(y_val, rf_cal.predict_proba(X_val)[:,1])
et_cp = average_precision_score(y_val, et_cal.predict_proba(X_val)[:,1])
print(f"\n[6] Calibrated PR-AUC → RF:{rf_cp:.4f} ET:{et_cp:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — ENSEMBLE + THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
total_w = rf_cp + et_cp
w_rf = rf_cp / total_w; w_et = et_cp / total_w

def ens(X): return w_rf*rf_cal.predict_proba(X)[:,1] + w_et*et_cal.predict_proba(X)[:,1]

val_probs  = ens(X_val)
test_probs = ens(X_test)
test_df['pred_prob'] = test_probs

p_v, r_v, t_v = precision_recall_curve(y_val, val_probs)
f1_v = 2*p_v[:-1]*r_v[:-1]/(p_v[:-1]+r_v[:-1]+1e-8)

idx_f1    = np.argmax(f1_v);              thresh_bal = float(t_v[idx_f1])
mask_rec  = r_v[:-1] >= RECALL_FLOOR
idx_hp    = int(np.where(mask_rec)[0][np.argmax(p_v[:-1][mask_rec])]) if mask_rec.any() else idx_f1
thresh_hp = float(t_v[idx_hp])
mask_prec = p_v[:-1] >= 0.60
idx_hr    = int(np.where(mask_prec)[0][np.argmax(r_v[:-1][mask_prec])]) if mask_prec.any() else idx_f1
thresh_hr = float(t_v[idx_hr])

test_df['pred_bal'] = (test_probs >= thresh_bal).astype(int)

print(f"\n[7] Ensemble RF({w_rf:.2f})+ET({w_et:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — TEST METRICS & FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
test_auc   = roc_auc_score(y_test, test_probs)
test_prauc = average_precision_score(y_test, test_probs)
p_t, r_t, t_t = precision_recall_curve(y_test, test_probs)
f1_t = 2*p_t[:-1]*r_t[:-1]/(p_t[:-1]+r_t[:-1]+1e-8)

results = {}
for label, thresh in [('balanced',thresh_bal),('high_precision',thresh_hp),('high_recall',thresh_hr)]:
    yp = (test_probs >= thresh).astype(int)
    cm = confusion_matrix(y_test, yp)
    tn,fp,fn,tp_ = cm.ravel()
    results[label] = dict(
        thresh=thresh, pred=yp, cm=cm,
        acc=accuracy_score(y_test,yp), bacc=balanced_accuracy_score(y_test,yp),
        prec=precision_score(y_test,yp,zero_division=0), rec=recall_score(y_test,yp,zero_division=0),
        f1=f1_score(y_test,yp,zero_division=0), mcc=matthews_corrcoef(y_test,yp),
        tp=int(tp_), fp=int(fp), fn=int(fn), tn=int(tn),
        specificity=tn/(tn+fp+1e-9), npv=tn/(tn+fn+1e-9),
    )

print(f"\n[8] Test AUC:{test_auc:.4f} PR-AUC:{test_prauc:.4f}")
for k,v in results.items():
    print(f"  {k}: Prec={v['prec']:.3f} Rec={v['rec']:.3f} F1={v['f1']:.3f} ACC={v['acc']:.3f}")

# EXACT MATCH GUARANTEED HERE
imp = pd.DataFrame({'feature': FEATURES, 'rf': rf_best.feature_importances_, 'et': et_best.feature_importances_})
imp['combined'] = w_rf*imp['rf'] + w_et*imp['et']
imp = imp.sort_values('combined', ascending=False).reset_index(drop=True)
imp.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

def feat_color(f):
    if any(x in f for x in ['vpd','humidity','moisture','precipitation','wind','dewpoint','skin_temp','temperature']):
        return BLUE_ACCENT
    if any(x in f for x in ['ndvi','nbr','evi','ndwi','gndvi','savi','ndsi','lst_day','veg']):
        return SAFE_GREEN
    if any(x in f for x in ['slope','aspect','elevation','mtpi']):
        return WARN_ORANGE
    return PURPLE

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE ALL 9 PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Generating 9 Comprehensive Scientific Plots...")

b  = results['balanced']
hp = results['high_precision']
hr = results['high_recall']
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=15)

# ─── FIGURE 1: Core Curves ───────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 4, figsize=(24, 6))
fig1.patch.set_facecolor(DARK_BG)
fig1.suptitle('FIRE RISK MODEL — Core Evaluation Curves', fontsize=14, fontweight='bold', color='#e6edf3', y=1.02)

fpr_e, tpr_e, _ = roc_curve(y_test, test_probs)
axes[0].fill_between(fpr_e, tpr_e, alpha=0.15, color=FIRE_RED)
axes[0].plot(fpr_e, tpr_e, color=FIRE_RED, lw=2.5, label=f'Ensemble AUC={test_auc:.4f}')
axes[0].plot([0,1],[0,1], color=BORDER, lw=1, linestyle=':')
axes[0].set_title('ROC Curve', fontweight='bold'); axes[0].legend(fontsize=8)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate'); axes[0].grid(True, alpha=0.3)

axes[1].fill_between(r_t, p_t, alpha=0.12, color=WARN_ORANGE)
axes[1].plot(r_t, p_t, color=WARN_ORANGE, lw=2.5, label=f'Ensemble PR-AUC={test_prauc:.4f}')
axes[1].scatter(b['rec'],  b['prec'],  s=150, color=SAFE_GREEN,  zorder=6, label=f"Balanced", marker='o')
axes[1].scatter(hp['rec'], hp['prec'], s=150, color=BLUE_ACCENT, zorder=6, label=f"Hi-Prec", marker='s')
axes[1].scatter(hr['rec'], hr['prec'], s=150, color=PURPLE,      zorder=6, label=f"Hi-Recall", marker='^')
axes[1].axhline(y=y_test.mean(), color=BORDER, lw=1, linestyle=':', label='Baseline')
axes[1].set_title('Precision-Recall Curve', fontweight='bold'); axes[1].legend(fontsize=8)
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0,1); axes[1].set_ylim(0,1.05)

axes[2].plot(t_t, p_t[:-1], color=BLUE_ACCENT, lw=2, label='Precision')
axes[2].plot(t_t, r_t[:-1], color=FIRE_RED, lw=2, label='Recall')
axes[2].plot(t_t, f1_t, color=SAFE_GREEN, lw=2.5, label='F1 Score')
for thresh, color, lbl in [(thresh_bal,SAFE_GREEN,'Bal'),(thresh_hp,BLUE_ACCENT,'HiPrec'),(thresh_hr,PURPLE,'HiRec')]:
    axes[2].axvline(thresh, color=color, lw=1.5, linestyle='--', alpha=0.8, label=f'{lbl}={thresh:.3f}')
axes[2].set_title('Metrics vs Threshold', fontweight='bold'); axes[2].legend(fontsize=7)
axes[2].set_xlabel('Decision Threshold'); axes[2].set_ylabel('Score'); axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0,1); axes[2].set_ylim(0,1.05)

axes[3].plot([0,1],[0,1], color=BORDER, lw=1.5, linestyle=':', label='Perfect calibration')
axes[3].plot(prob_pred, prob_true, color=FIRE_RED, lw=2.5, marker='o', ms=6, label='RF+ET Ensemble')
axes[3].fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=FIRE_RED)
axes[3].set_title('Calibration Curve', fontweight='bold'); axes[3].legend(fontsize=9)
axes[3].set_xlabel('Mean Predicted Probability'); axes[3].set_ylabel('Fraction of Positives')
axes[3].grid(True, alpha=0.3); axes[3].set_xlim(0,1); axes[3].set_ylim(0,1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_core_curves.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ fig1_core_curves.png")

# ─── FIGURE 2: Confusion Matrices ────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor(DARK_BG)
fig2.suptitle('Confusion Matrices — Three Threshold Modes', fontsize=14, fontweight='bold', color='#e6edf3', y=1.02)

for ax, (label, res), cmap, title_color in zip(axes, [('balanced',b),('high_precision',hp),('high_recall',hr)], ['Greens','Blues','Reds'], [SAFE_GREEN, BLUE_ACCENT, FIRE_RED]):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap=cmap, ax=ax, xticklabels=['No Fire','Fire'], yticklabels=['No Fire','Fire'], annot_kws={'size':16, 'weight':'bold'}, linewidths=2, linecolor=DARK_BG, cbar_kws={'shrink':0.8})
    ax.set_title(f"{label.replace('_',' ').title()}\nthresh={res['thresh']:.3f}", fontweight='bold', color=title_color, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('Actual', fontsize=10)
    textstr = (f"TP={res['tp']:,}  FP={res['fp']:,}\nFN={res['fn']:,}  TN={res['tn']:,}\nPrec={res['prec']:.3f}  Rec={res['rec']:.3f}\nF1={res['f1']:.3f}  MCC={res['mcc']:.3f}")
    ax.text(0.5, -0.22, textstr, transform=ax.transAxes, fontsize=9, ha='center', va='top', color='#8b949e', bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', alpha=0.8))

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_DIR / 'fig2_confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ fig2_confusion_matrices.png")

# ─── FIGURE 3: Feature Importance Analysis ───────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(20, 10))
fig3.patch.set_facecolor(DARK_BG)
fig3.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold', color='#e6edf3')

top30 = imp.head(30)
colors = [feat_color(f) for f in top30['feature']]
axes[0].barh(range(len(top30)), top30['combined'].values[::-1]*100, color=colors[::-1], edgecolor=BORDER, linewidth=0.5)
axes[0].set_yticks(range(len(top30))); axes[0].set_yticklabels(top30['feature'].values[::-1], fontsize=8)
axes[0].set_title('Top 30 Features (Weighted Ensemble)', fontweight='bold')
axes[0].set_xlabel('Importance Score (%)'); axes[0].grid(True, alpha=0.3, axis='x')
from matplotlib.patches import Patch
axes[0].legend(handles=[
    Patch(facecolor=BLUE_ACCENT, label='Weather'), Patch(facecolor=SAFE_GREEN,  label='Remote Sensing'),
    Patch(facecolor=WARN_ORANGE, label='Terrain'), Patch(facecolor=PURPLE,      label='Other')
], loc='lower right', fontsize=8)

colors_all = [feat_color(f) for f in imp['feature']]
axes[1].scatter(imp['rf']*100, imp['et']*100, c=colors_all, s=60, alpha=0.8, edgecolors=BORDER, lw=0.5)
max_val = max(imp['rf'].max(), imp['et'].max()) * 100
axes[1].plot([0, max_val], [0, max_val], color=BORDER, lw=1, linestyle='--', alpha=0.6)
axes[1].set_title('RF vs Extra Trees — Agreement', fontweight='bold')
axes[1].set_xlabel('Random Forest Importance (%)'); axes[1].set_ylabel('Extra Trees Importance (%)'); axes[1].grid(True, alpha=0.3)
for _, row in imp.head(10).iterrows():
    axes[1].annotate(row['feature'], (row['rf']*100, row['et']*100), textcoords='offset points', xytext=(4,2), fontsize=6, color='#8b949e')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ fig3_feature_importance.png")

# ─── FIGURE 4: Risk Score Distribution + Metrics Summary ─────────────────────
fig4, axes = plt.subplots(2, 2, figsize=(16, 12))
fig4.patch.set_facecolor(DARK_BG)
fig4.suptitle('Risk Score Distribution & Metrics Summary', fontsize=14, fontweight='bold', color='#e6edf3')

bins = np.linspace(0, 1, 60)
axes[0,0].hist(test_probs[y_test==0], bins=bins, alpha=0.65, color=BLUE_ACCENT, label='No Fire', density=True)
axes[0,0].hist(test_probs[y_test==1], bins=bins, alpha=0.75, color=FIRE_RED,    label='Fire',    density=True)
for thresh, color, ls, lbl in [(thresh_bal, SAFE_GREEN, '--', f'Bal={thresh_bal:.3f}'), (thresh_hp, BLUE_ACCENT, ':', f'HiP={thresh_hp:.3f}'), (thresh_hr, PURPLE, '-.', f'HiR={thresh_hr:.3f}')]:
    axes[0,0].axvline(thresh, color=color, lw=2, linestyle=ls, label=lbl)
axes[0,0].set_title('Predicted Risk Score Distribution', fontweight='bold'); axes[0,0].set_xlabel('Risk Score'); axes[0,0].set_ylabel('Density'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

from scipy.stats import gaussian_kde
for probs_subset, color, label in [(test_probs[y_test==0], BLUE_ACCENT, 'No Fire'), (test_probs[y_test==1], FIRE_RED, 'Fire')]:
    kde = gaussian_kde(probs_subset, bw_method=0.1)
    x_range = np.linspace(0, 1, 300)
    axes[0,1].fill_between(x_range, kde(x_range), alpha=0.4, color=color)
    axes[0,1].plot(x_range, kde(x_range), color=color, lw=2, label=label)
axes[0,1].axvline(thresh_bal, color=SAFE_GREEN, lw=2, linestyle='--', label=f'Balanced={thresh_bal:.3f}')
axes[0,1].set_title('Risk Score KDE', fontweight='bold'); axes[0,1].set_xlabel('Risk Score'); axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_xlim(0, 1)

modes, metrics, labels = ['Balanced', 'Hi Prec', 'Hi Recall'], ['prec','rec','f1','acc','bacc'], ['Precision','Recall','F1','Accuracy','Bal Acc']
x = np.arange(len(modes)); width = 0.15
for i, (metric, label, color) in enumerate(zip(metrics, labels, [BLUE_ACCENT, FIRE_RED, SAFE_GREEN, WARN_ORANGE, PURPLE])):
    axes[1,0].bar(x + i*width, [b[metric], hp[metric], hr[metric]], width, label=label, color=color, alpha=0.85, edgecolor=BORDER)
axes[1,0].set_title('Metrics Across Threshold Modes', fontweight='bold'); axes[1,0].set_xticks(x + width*2); axes[1,0].set_xticklabels(modes, fontsize=9); axes[1,0].set_ylabel('Score'); axes[1,0].set_ylim(0, 1.1); axes[1,0].legend(fontsize=8, ncol=2); axes[1,0].grid(True, alpha=0.3, axis='y')

categories = ['Precision','Recall','Specificity','NPV','F1','Bal Acc','MCC']
x2 = np.arange(len(categories)); w2 = 0.25
axes[1,1].bar(x2-w2, [b['prec'], b['rec'], b['specificity'], b['npv'], b['f1'], b['bacc'], max(0,b['mcc'])], w2, label='Balanced', color=SAFE_GREEN, alpha=0.85, edgecolor=BORDER)
axes[1,1].bar(x2,    [hp['prec'],hp['rec'],hp['specificity'],hp['npv'],hp['f1'],hp['bacc'],max(0,hp['mcc'])], w2, label='Hi-Precision', color=BLUE_ACCENT, alpha=0.85, edgecolor=BORDER)
axes[1,1].bar(x2+w2, [hr['prec'],hr['rec'],hr['specificity'],hr['npv'],hr['f1'],hr['bacc'],max(0,hr['mcc'])], w2, label='Hi-Recall', color=FIRE_RED, alpha=0.85, edgecolor=BORDER)
axes[1,1].set_title('Complete Metrics Comparison', fontweight='bold'); axes[1,1].set_xticks(x2); axes[1,1].set_xticklabels(categories, fontsize=8); axes[1,1].set_ylabel('Score'); axes[1,1].set_ylim(0, 1.1); axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3, axis='y'); axes[1,1].axhline(1.0, color=BORDER, lw=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_distribution_metrics.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ fig4_distribution_metrics.png")

# ─── FIGURE 5: Optuna Optimization History ────────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(16, 6))
fig5.patch.set_facecolor(DARK_BG)
fig5.suptitle('Optuna Hyperparameter Optimization History', fontsize=14, fontweight='bold', color='#e6edf3')

for ax, study, name, color in [(axes[0], rf_study, 'Random Forest', SAFE_GREEN), (axes[1], et_study, 'Extra Trees', BLUE_ACCENT)]:
    vals = [t.value for t in study.trials if t.value is not None]
    if not vals:
        ax.set_title(f"{name} (No completed trials)", color=BORDER)
        continue
    best = np.maximum.accumulate(vals)
    trials = range(1, len(vals)+1)
    ax.scatter(trials, vals, color=color, alpha=0.4, s=20, label='Trial score')
    ax.plot(trials, best, color=FIRE_RED, lw=2.5, label='Best so far')
    ax.fill_between(trials, best, min(vals), alpha=0.08, color=color)
    ax.axhline(max(vals), color=BORDER, lw=1, linestyle='--', alpha=0.6)
    ax.set_title(f'{name} — Optimization', fontweight='bold'); ax.set_xlabel('Trial'); ax.set_ylabel('Score')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.text(len(vals)*0.7, max(vals)*0.98, f'Best: {max(vals):.4f}', color=FIRE_RED, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_optuna_history.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ fig5_optuna_history.png")

# ─── FIGURE 6: SHAP SUMMARY PLOT ──────────────────────────────────────────────
if HAS_SHAP:
    plt.figure(figsize=(12, 10))
    plt.gcf().patch.set_facecolor(DARK_BG)
    ax = plt.gca()
    ax.set_facecolor(DARK_BG)
    
    # SHAP requires a manageable sample size
    sample_idx = np.random.choice(len(X_test), min(1500, len(X_test)), replace=False)
    X_test_df = pd.DataFrame(X_test[sample_idx], columns=FEATURES)
    
    explainer = shap.TreeExplainer(rf_best)
    shap_values = explainer.shap_values(X_test_df)
    
    # Extract the correct class shape based on sklearn version / structure
    if isinstance(shap_values, list): 
        shap_values = shap_values[1] # Class 1 (Fire)
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    shap.summary_plot(shap_values, X_test_df, max_display=20, show=False)
    
    # Recolor for dark theme
    for text in plt.gcf().texts: text.set_color('#e6edf3')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#e6edf3')
    plt.title('SHAP Feature Impacts (Directional Importance)', color='#e6edf3', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_shap_summary.png', dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print("  ✓ fig6_shap_summary.png")

# ─── FIGURE 7: TEMPORAL STABILITY ─────────────────────────────────────────────
fig7, ax1 = plt.subplots(figsize=(14, 6))
fig7.patch.set_facecolor(DARK_BG)

# Group by Month-Year to track model stability over seasons
temporal = test_df.copy()
temporal['month_year'] = temporal['date'].dt.to_period('M')
temporal_agg = temporal.groupby('month_year').agg({
    'fire_label': 'mean',
    'pred_prob': 'mean'
}).dropna()
temporal_agg.index = temporal_agg.index.to_timestamp()

ax1.plot(temporal_agg.index, temporal_agg['fire_label'], color=SAFE_GREEN, lw=2.5, marker='o', label='Actual Fire Rate')
ax1.plot(temporal_agg.index, temporal_agg['pred_prob'], color=FIRE_RED, lw=2.5, marker='s', linestyle='--', label='Mean Predicted Risk')
ax1.set_title('Temporal Reliability: Predicted Risk vs Actual Fire Rate Over Time', fontweight='bold', pad=15)
ax1.set_ylabel('Probability / Rate')
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_temporal_stability.png', dpi=150, facecolor=DARK_BG)
plt.close()
print("  ✓ fig7_temporal_stability.png")

# ─── FIGURE 8: SPATIAL ERROR DISTRIBUTION ────────────────────────────────────
fig8, ax = plt.subplots(figsize=(12, 10))
fig8.patch.set_facecolor(DARK_BG)

yp = test_df['pred_bal'].values
yt = test_df['fire_label'].values
lats = test_df['latitude'].values
lons = test_df['longitude'].values

# Masks
tn_mask = (yt == 0) & (yp == 0)
fp_mask = (yt == 0) & (yp == 1)
fn_mask = (yt == 1) & (yp == 0)
tp_mask = (yt == 1) & (yp == 1)

# Plot in order of visual importance
ax.scatter(lons[tn_mask], lats[tn_mask], c='#21262d', s=5, alpha=0.3, label='True Negative (No Fire)')
ax.scatter(lons[fp_mask], lats[fp_mask], c=BLUE_ACCENT, s=20, alpha=0.7, label='False Positive (Over-predicted)', marker='^')
ax.scatter(lons[fn_mask], lats[fn_mask], c=WARN_ORANGE, s=20, alpha=0.8, label='False Negative (Missed Fire)', marker='v')
ax.scatter(lons[tp_mask], lats[tp_mask], c=FIRE_RED, s=25, alpha=0.9, label='True Positive (Caught Fire)', marker='*')

ax.set_title('Spatial Distribution of Model Performance (Test Set)', fontweight='bold', pad=15)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.1)
ax.legend(markerscale=2, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_spatial_errors.png', dpi=150, facecolor=DARK_BG)
plt.close()
print("  ✓ fig8_spatial_errors.png")

# ─── FIGURE 9: TOP FEATURES CORRELATION HEATMAP ──────────────────────────────
fig9, ax = plt.subplots(figsize=(12, 10))
fig9.patch.set_facecolor(DARK_BG)

top_15_feats = imp['feature'].head(15).tolist()
df_test_X = pd.DataFrame(X_test, columns=FEATURES)[top_15_feats]
corr_matrix = df_test_X.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            square=True, linewidths=0.5, linecolor=DARK_BG, 
            cbar_kws={"shrink": .8}, ax=ax, annot_kws={'size': 9, 'color': '#e6edf3'})

ax.set_title('Pearson Correlation Between Top 15 Predictors', fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', color='#8b949e')
plt.yticks(color='#8b949e')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig9_feature_correlation.png', dpi=150, facecolor=DARK_BG)
plt.close()
print("  ✓ fig9_feature_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Saving model...")
checkpoint = {
    'rf_model': rf_cal, 'et_model': et_cal,
    'weights': {'rf': float(w_rf), 'et': float(w_et)},
    'impute_medians': impute_medians.to_dict(), # Safe imputation values
    'feature_names': FEATURES,
    'thresholds': {'balanced': thresh_bal, 'high_precision': thresh_hp, 'high_recall': thresh_hr},
    'metrics': {'test_roc_auc': float(test_auc), 'test_pr_auc': float(test_prauc),
                **{k: {m: float(v[m]) for m in ['prec','rec','f1','acc','mcc']} for k,v in results.items()}},
    'best_rf_params': rf_study.best_params if len(rf_study.trials) > 0 else {}, 
    'best_et_params': et_study.best_params if len(et_study.trials) > 0 else {},
}
with open(MODEL_DIR / 'rf_fire_risk_model.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
print(f"  ✓ models/rf_fire_risk_model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{"="*70}
COMPLETE — All Standard and Scientific outputs saved to: {OUTPUT_DIR}
{"="*70}
  fig1_core_curves.png             (ROC, PR, Threshold sweeps, Calibration)
  fig2_confusion_matrices.png      (Performance at different thresholds)
  fig3_feature_importance.png      (Standard Gini importance & Model Agreement)
  fig4_distribution_metrics.png    (Risk probability density & Metrics)
  fig5_optuna_history.png          (Hyperparameter tuning track)
  fig6_shap_summary.png            (Directional feature impact via SHAP)
  fig7_temporal_stability.png      (Time-series track of predictions)
  fig8_spatial_errors.png          (Geospatial mapping of TP/FP/FN/TN)
  fig9_feature_correlation.png     (Collinearity check of top features)
  
  feature_importance.csv           (All 81 features ranked)

  ROC-AUC   : {test_auc:.4f}
  PR-AUC    : {test_prauc:.4f}
  F1 (Bal)  : {b['f1']:.4f}  Prec={b['prec']:.4f}  Rec={b['rec']:.4f}
  F1 (HiP)  : {hp['f1']:.4f}  Prec={hp['prec']:.4f}  Rec={hp['rec']:.4f}
  F1 (HiR)  : {hr['f1']:.4f}  Prec={hr['prec']:.4f}  Rec={hr['rec']:.4f}
""")
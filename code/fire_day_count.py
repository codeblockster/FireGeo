import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────
# 1. Load the data
# ────────────────────────────────────────────────
file_path = r"/Users/prabhatrawal/Minor_project_code/data/mod14a1_data/District_4/District_4_mod14a1_data.csv"

df = pd.read_csv(file_path)

# Make sure date is datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Create a column that tells us if fire was detected that day
df['fire_detected'] = df['total_fire_pixels'] > 0

print("Data loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min().date()}  to  {df['date'].max().date()}\n")

# ────────────────────────────────────────────────
# 2. Basic statistics per zone + grand totals
# ────────────────────────────────────────────────
stats = df.groupby('zone').agg(
    total_days=('date', 'count'),
    fire_days=('fire_detected', 'sum'),
    total_fire_pixels=('total_fire_pixels', 'sum'),
).reset_index()

stats['no_fire_days'] = stats['total_days'] - stats['fire_days']
stats['fire_percentage'] = (stats['fire_days'] / stats['total_days'] * 100).round(3)

# Fire to No-fire ratio (safe handling)
stats['fire_to_no_fire_ratio'] = stats.apply(
    lambda row: round(row['fire_days'] / row['no_fire_days'], 3) 
    if row['no_fire_days'] > 0 
    else '-', 
    axis=1
)

# Calculate grand totals
grand_total = pd.DataFrame({
    'zone': ['TOTAL'],
    'total_days': [stats['total_days'].sum()],
    'fire_days': [stats['fire_days'].sum()],
    'no_fire_days': [stats['no_fire_days'].sum()],
    'fire_percentage': ['-'],           # doesn't make sense for total
    'fire_to_no_fire_ratio': [round(stats['fire_days'].sum() / stats['no_fire_days'].sum(), 3)],
    'total_fire_pixels': [stats['total_fire_pixels'].sum()]
})

# Combine per-zone stats + grand total
stats_display = pd.concat([stats, grand_total], ignore_index=True)

# ── Print nicely formatted table ───────────────────────────────
print("\nFire detection statistics per zone + overall total:")
print(stats_display[[
    'zone',
    'total_days',
    'fire_days',
    'no_fire_days',
    'fire_percentage',
    'fire_to_no_fire_ratio',
    'total_fire_pixels'
]].to_string(index=False, float_format="{:,.1f}".format))

# ────────────────────────────────────────────────
# 3. Visualizations
# ────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig = plt.figure(figsize=(16, 12))

# ── Bar plot: Number of fire detected days per zone ─────────────
ax1 = plt.subplot(2, 2, 1)
sns.barplot(data=stats, x='zone', y='fire_days', ax=ax1)
ax1.set_title('Number of Days with Fire Detected per Zone', fontsize=13, pad=12)
ax1.set_ylabel('Number of Fire Days')
ax1.set_xlabel('Zone')

# Add value labels on bars
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9)

# ── Bar plot: Fire percentage per zone ───────────────────────────
ax2 = plt.subplot(2, 2, 2)
sns.barplot(data=stats, x='zone', y='fire_percentage', ax=ax2)
ax2.set_title('Percentage of Days with Fire Detected (%)', fontsize=13, pad=12)
ax2.set_ylabel('Fire Days %')
ax2.set_xlabel('Zone')

# ── Total fire pixels per zone ───────────────────────────────────
ax3 = plt.subplot(2, 2, 3)
sns.barplot(data=stats, x='zone', y='total_fire_pixels', ax=ax3)
ax3.set_title('Total Fire Pixels Detected (25+ years)', fontsize=13, pad=12)
ax3.set_ylabel('Total Fire Pixels')
ax3.set_xlabel('Zone')
ax3.tick_params(axis='y', labelsize=9)

# ── Time series of fire events (whole district) ──────────────────
ax4 = plt.subplot(2, 2, 4)
yearly = df.resample('YE', on='date')['fire_detected'].sum()
yearly.plot(kind='bar', ax=ax4, color='darkorange')
ax4.set_title('Yearly Number of Days with Fire (All Zones)', fontsize=13, pad=12)
ax4.set_ylabel('Fire Days per Year')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.suptitle('Fire Activity Analysis - District_0 (MOD14A1)', fontsize=16, y=1.02)
plt.show()


# Optional: More detailed yearly trend per zone (if you want)
# Uncomment the following block if interested:

"""
pivot_year_zone = df[df['fire_detected']].copy()
pivot_year_zone['year'] = pivot_year_zone['date'].dt.year
year_zone_count = pivot_year_zone.pivot_table(
    index='year', 
    columns='zone', 
    values='fire_detected', 
    aggfunc='sum',
    fill_value=0
)

plt.figure(figsize=(14, 8))
year_zone_count.plot(kind='line', marker='o', linewidth=1.2)
plt.title('Yearly Fire Days Trend per Zone', fontsize=14)
plt.ylabel('Number of Fire Days')
plt.grid(True, alpha=0.3)
plt.legend(title='Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""
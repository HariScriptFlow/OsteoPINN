"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 2: Exploratory Data Analysis & Visualizations              ║
║  Run: python step2_eda.py                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
warnings.filterwarnings("ignore")

os.makedirs("outputs/plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="husl")
TCAT_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
TCAT_NAMES  = {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}

print("\n📊 Loading combined dataset...")
df = pd.read_csv("outputs/combined_dataset.csv")
print(f"   {df.shape[0]} rows loaded")

# ─────────────────────────────────────────────────────────────────
# FIGURE 1: Dataset Overview (2x3 grid)
# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Osteoporosis Dataset — Exploratory Analysis", fontsize=18, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1a. BMD Distribution by T-Category
ax1 = fig.add_subplot(gs[0, 0])
for cat, color in TCAT_COLORS.items():
    sub = df[df['t_category'] == cat]['bmd']
    ax1.hist(sub, bins=25, alpha=0.65, color=color, label=TCAT_NAMES[cat], edgecolor='white')
ax1.axvline(0.833, color='orange', ls='--', lw=1.5, label='Osteopenia cut')
ax1.axvline(0.648, color='red',    ls='--', lw=1.5, label='Osteoporosis cut')
ax1.set_xlabel("Bone Mineral Density (g/cm²)")
ax1.set_ylabel("Count")
ax1.set_title("BMD Distribution by Category")
ax1.legend(fontsize=8)

# 1b. Age vs BMD scatter
ax2 = fig.add_subplot(gs[0, 1])
colors = [TCAT_COLORS[c] for c in df['t_category']]
ax2.scatter(df['age'], df['bmd'], c=colors, alpha=0.4, s=15)
z = np.polyfit(df['age'], df['bmd'], 1)
p = np.poly1d(z)
xline = np.linspace(df['age'].min(), df['age'].max(), 100)
ax2.plot(xline, p(xline), 'k--', lw=2, label=f'Trend (slope={z[0]:.4f})')
ax2.set_xlabel("Age (years)")
ax2.set_ylabel("BMD (g/cm²)")
ax2.set_title("Age vs BMD")
ax2.legend(fontsize=8)
from matplotlib.patches import Patch
legend_elements = [Patch(fc=c, label=TCAT_NAMES[k]) for k,c in TCAT_COLORS.items()]
ax2.legend(handles=legend_elements, fontsize=8)

# 1c. T-Category Counts
ax3 = fig.add_subplot(gs[0, 2])
counts = df['t_category'].value_counts().sort_index()
bars = ax3.bar([TCAT_NAMES[i] for i in counts.index],
               counts.values,
               color=[TCAT_COLORS[i] for i in counts.index],
               edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val}\n({100*val/len(df):.1f}%)', ha='center', va='bottom', fontsize=9)
ax3.set_title("T-Score Category Distribution")
ax3.set_ylabel("Count")

# 1d. BMD by Sex and Age Group
ax4 = fig.add_subplot(gs[1, 0])
age_groups = ['<30', '30-45', '45-60', '60-75', '75+']
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,75,120], labels=age_groups)
for sex, color, ls in [('Female','#e74c3c','-'), ('Male','#3498db','--')]:
    grp = df[df['sex']==sex].groupby('age_group', observed=True)['bmd'].mean()
    grp = grp.reindex(age_groups)
    ax4.plot(age_groups, grp.values, marker='o', color=color, ls=ls, lw=2, label=sex)
ax4.set_xlabel("Age Group")
ax4.set_ylabel("Mean BMD (g/cm²)")
ax4.set_title("BMD by Sex and Age Group")
ax4.legend()

# 1e. Risk Score Distribution
ax5 = fig.add_subplot(gs[1, 1])
for cat, color in TCAT_COLORS.items():
    sub = df[df['t_category']==cat]['risk_score']
    ax5.hist(sub, bins=20, alpha=0.65, color=color, label=TCAT_NAMES[cat], edgecolor='white')
ax5.set_xlabel("Composite Risk Score")
ax5.set_ylabel("Count")
ax5.set_title("Risk Score by T-Category")
ax5.legend(fontsize=8)

# 1f. Fracture Rate by Category
ax6 = fig.add_subplot(gs[1, 2])
frac_rate = df.groupby('t_category')['fracture_binary'].mean() * 100
bars6 = ax6.bar([TCAT_NAMES[i] for i in frac_rate.index],
                frac_rate.values,
                color=[TCAT_COLORS[i] for i in frac_rate.index],
                edgecolor='white')
for bar, val in zip(bars6, frac_rate.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax6.set_title("Fracture Rate by Category")
ax6.set_ylabel("Fracture Rate (%)")

plt.savefig("outputs/plots/01_eda_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 01_eda_overview.png")

# ─────────────────────────────────────────────────────────────────
# FIGURE 2: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
num_cols = ['age','sex_binary','bmi','medication_risk','family_history',
            'postmenopausal','low_calcium','low_vitd','sedentary',
            'underweight','smoking','alcohol','risk_score','bmd','t_score']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size":8}, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("outputs/plots/02_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 02_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────
# FIGURE 3: BMD Trajectory (Physics-based projection)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("BMD Trajectory Projection (Physics-Based)", fontsize=14, fontweight='bold')

ages_proj = np.linspace(30, 85, 100)

def bmd_trajectory(age0, bmd0, sex, years=20, medication_factor=1.0):
    ages, bmds = [age0], [bmd0]
    current_bmd = bmd0
    for yr in range(1, years+1):
        a = age0 + yr
        rate = 0.005 if sex == 'Male' else 0.007
        if a > 50 and sex == 'Female': rate = 0.012
        current_bmd = max(0.3, current_bmd - rate * current_bmd * medication_factor)
        ages.append(a)
        bmds.append(current_bmd)
    return ages, bmds

# Left: Female trajectories
ax = axes[0]
profiles = [
    (50, 0.95, "Female", 1.0, '#e74c3c', "Healthy, 50"),
    (50, 0.75, "Female", 1.0, '#c0392b', "Osteopenic, 50"),
    (60, 0.95, "Female", 1.0, '#f39c12', "Healthy, 60"),
    (60, 0.95, "Female", 1.5, '#8e44ad', "On Glucocorticoids, 60"),
]
ax.axhline(0.833, color='orange', ls=':', lw=1.2, label='Osteopenia threshold')
ax.axhline(0.648, color='red',    ls=':', lw=1.2, label='Osteoporosis threshold')
for age0, bmd0, sex, med, color, label in profiles:
    ages_t, bmds_t = bmd_trajectory(age0, bmd0, sex, years=25, medication_factor=med)
    ax.plot(ages_t, bmds_t, color=color, lw=2, label=label)
ax.fill_between([30,85], 0.648, 0.833, alpha=0.08, color='orange')
ax.fill_between([30,85], 0.3,   0.648, alpha=0.08, color='red')
ax.set_xlabel("Age (years)"); ax.set_ylabel("BMD (g/cm²)")
ax.set_title("Female BMD Trajectories")
ax.legend(fontsize=8); ax.set_xlim(48, 85); ax.set_ylim(0.3, 1.1)

# Right: Male vs Female comparison
ax2 = axes[1]
for sex, color, ls in [('Female','#e74c3c','-'), ('Male','#3498db','--')]:
    ages_t, bmds_t = bmd_trajectory(50, 0.95, sex, years=30, medication_factor=1.0)
    ax2.plot(ages_t, bmds_t, color=color, lw=2.5, ls=ls, label=f'{sex} - No medication')
    ages_t2, bmds_t2 = bmd_trajectory(50, 0.95, sex, years=30, medication_factor=1.5)
    ax2.plot(ages_t2, bmds_t2, color=color, lw=1.5, ls=':', alpha=0.6, label=f'{sex} - Glucocorticoids')
ax2.axhline(0.833, color='orange', ls=':', lw=1.2)
ax2.axhline(0.648, color='red',    ls=':', lw=1.2)
ax2.fill_between([49,82], 0.648, 0.833, alpha=0.08, color='orange')
ax2.fill_between([49,82], 0.3,   0.648, alpha=0.08, color='red')
ax2.set_xlabel("Age (years)"); ax2.set_ylabel("BMD (g/cm²)")
ax2.set_title("Male vs Female BMD Trajectory")
ax2.legend(fontsize=8); ax2.set_xlim(49, 82); ax2.set_ylim(0.3, 1.1)

plt.tight_layout()
plt.savefig("outputs/plots/03_bmd_trajectories.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 03_bmd_trajectories.png")

print("\n✅ EDA complete! Check outputs/plots/")
print("▶  Next: python step3_pinn.py")

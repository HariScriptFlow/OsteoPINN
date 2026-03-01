"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 1: Dataset Merging & Preprocessing                         ║
║  Combines bmd.csv (169 rows) + osteoporosis.csv (~1958 rows)     ║
║  Run: python step1_preprocess.py                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle, os, warnings
warnings.filterwarnings("ignore")

# Always save outputs next to the script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)  # Set working dir to script location

print("\n📂 Loading datasets...")

import os, sys

def find_file(filename):
    # Look in same folder as script, then data/ subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    candidates = [
        os.path.join(script_dir, filename),           # same folder as script
        os.path.join(script_dir, "data", filename),   # data/ subfolder
        os.path.join(os.getcwd(), filename),           # current working dir
        os.path.join(os.getcwd(), "data", filename),  # data/ from cwd
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"   Found: {path}")
            return path
    raise FileNotFoundError(
        f"\n❌ Cannot find {filename}\n"
        f"   Please make sure {filename} is in the same folder as this script:\n"
        f"   {script_dir}\\"
    )

bmd   = pd.read_csv(find_file("bmd.csv"))
osteo = pd.read_csv(find_file("osteoporosis.csv"))
print(f"   bmd.csv       → {bmd.shape[0]} rows")
print(f"   osteoporosis  → {osteo.shape[0]} rows")

# ─────────────────────────────────────────────────────────────────
# CLEAN bmd.csv
# ─────────────────────────────────────────────────────────────────
bmd.columns = [c.lower().strip().replace(" ", "_") for c in bmd.columns]

# Handle merged column name 'waiting_timebmd'
for col in bmd.columns:
    if 'bmd' in col and 'waiting' in col:
        bmd.rename(columns={col: 'bmd'}, inplace=True)
        break

if 'bmd' not in bmd.columns:
    bmd.rename(columns={bmd.columns[-1]: 'bmd'}, inplace=True)

bmd['sex'] = bmd['sex'].map({'M':'Male','F':'Female','Male':'Male','Female':'Female'})
bmd['fracture_binary'] = bmd['fracture'].apply(
    lambda x: 1 if str(x).lower().strip() in ['fracture','1','yes','true'] else 0)

med_map = {'no medication':0,'glucocorticoids':2,'anticonvulsant':1,'other':1}
bmd['medication_risk'] = bmd['medication'].str.lower().str.strip().map(med_map).fillna(1)
bmd['bmi'] = bmd['weight_kg'] / (bmd['height_cm']/100)**2

YOUNG_MEAN, YOUNG_SD = 0.964, 0.122
bmd['t_score']    = (bmd['bmd'] - YOUNG_MEAN) / YOUNG_SD
bmd['t_category'] = bmd['t_score'].apply(lambda t: 0 if t>=-1 else (1 if t>=-2.5 else 2))

# ─────────────────────────────────────────────────────────────────
# CLEAN osteoporosis.csv
# ─────────────────────────────────────────────────────────────────
osteo.columns = [c.strip() for c in osteo.columns]
osteo.rename(columns={'Age':'age','Gender':'sex','Osteoporosis':'osteoporosis_label'}, inplace=True)

bmap = {'Yes':1,'No':0,True:1,False:0,1:1,0:0}
osteo['family_history'] = osteo.get('Family History', pd.Series(0, index=osteo.index)).map(bmap).fillna(0)
osteo['postmenopausal'] = (osteo.get('Hormonal Changes', '') == 'Postmenopausal').astype(int)
osteo['low_calcium']    = (osteo.get('Calcium Intake','') == 'Low').astype(int)
osteo['low_vitd']       = (osteo.get('Vitamin D Intake','') == 'Insufficient').astype(int)
osteo['sedentary']      = (osteo.get('Physical Activity','') == 'Sedentary').astype(int)
osteo['underweight']    = (osteo.get('Body Weight','') == 'Underweight').astype(int)
osteo['smoking']        = osteo.get('Smoking', pd.Series(0,index=osteo.index)).map(bmap).fillna(0)
osteo['alcohol']        = osteo.get('Alcohol Consumption', pd.Series(0,index=osteo.index)).map(bmap).fillna(0)

if 'osteoporosis_label' in osteo.columns:
    osteo['osteoporosis_label'] = osteo['osteoporosis_label'].map(bmap).fillna(0).astype(int)
else:
    osteo['osteoporosis_label'] = 0

# ─────────────────────────────────────────────────────────────────
# ESTIMATE BMD for osteoporosis.csv using physics formula
# ─────────────────────────────────────────────────────────────────
def estimate_bmd(row):
    base = 0.96
    base -= max(0, (row['age'] - 50)) * 0.005
    if row['sex'] == 'Female':         base -= 0.07
    if row.get('postmenopausal',0):    base -= 0.06
    if row.get('low_calcium',0):       base -= 0.03
    if row.get('low_vitd',0):          base -= 0.03
    if row.get('sedentary',0):         base -= 0.02
    if row.get('underweight',0):       base -= 0.04
    if row.get('smoking',0):           base -= 0.03
    if row.get('alcohol',0):           base -= 0.02
    if row.get('family_history',0):    base -= 0.02
    base += np.random.normal(0, 0.035)
    return float(np.clip(base, 0.30, 1.40))

np.random.seed(42)
osteo['bmd']          = osteo.apply(estimate_bmd, axis=1)
osteo['t_score']      = (osteo['bmd'] - YOUNG_MEAN) / YOUNG_SD
osteo['t_category']   = osteo['t_score'].apply(lambda t: 0 if t>=-1 else (1 if t>=-2.5 else 2))
osteo['fracture_binary'] = osteo['osteoporosis_label']
osteo['weight_kg']    = np.where(osteo['underweight']==1,
                                  np.random.normal(51,6,len(osteo)),
                                  np.random.normal(68,10,len(osteo)))
osteo['height_cm']    = np.where(osteo['sex']=='Female',
                                  np.random.normal(160,7,len(osteo)),
                                  np.random.normal(172,7,len(osteo)))
osteo['bmi']          = osteo['weight_kg'] / (osteo['height_cm']/100)**2
osteo['medication_risk'] = 0

# ─────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────
COLS = ['age','sex','weight_kg','height_cm','bmi','medication_risk',
        'family_history','postmenopausal','low_calcium','low_vitd',
        'sedentary','underweight','smoking','alcohol',
        'bmd','t_score','t_category','fracture_binary']

bmd_p = bmd.copy()
for c in COLS:
    if c not in bmd_p.columns: bmd_p[c] = 0
bmd_p['source'] = 'bmd_real'

osteo_p = osteo.copy()
for c in COLS:
    if c not in osteo_p.columns: osteo_p[c] = 0
osteo_p['source'] = 'osteo_estimated'

combined = pd.concat([bmd_p[COLS+['source']], osteo_p[COLS+['source']]], ignore_index=True)
combined['sex_binary']  = (combined['sex'] == 'Female').astype(int)
combined['risk_score']  = (
    combined['postmenopausal']*3 + combined['low_calcium']*2 +
    combined['low_vitd']*2      + combined['sedentary']*1    +
    combined['underweight']*2   + combined['smoking']*2      +
    combined['alcohol']*1       + combined['family_history']*2 +
    combined['medication_risk']
)

print(f"\n✅ Combined: {combined.shape[0]} rows")
print(f"   BMD real (bmd.csv):        {(combined.source=='bmd_real').sum()}")
print(f"   BMD estimated (osteo.csv): {(combined.source=='osteo_estimated').sum()}")

# T-category distribution
tcat_names = {0:'Normal', 1:'Osteopenia', 2:'Osteoporosis'}
for k,v in tcat_names.items():
    n = (combined['t_category']==k).sum()
    print(f"   {v}: {n} ({100*n/len(combined):.1f}%)")

# ─────────────────────────────────────────────────────────────────
# SCALE & SAVE
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = ['age','sex_binary','weight_kg','height_cm','bmi',
                'medication_risk','family_history','postmenopausal',
                'low_calcium','low_vitd','sedentary','underweight',
                'smoking','alcohol','risk_score']

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(combined[FEATURE_COLS])
X_df     = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

combined.to_csv("outputs/combined_dataset.csv", index=False)
X_df.to_csv("outputs/X_scaled.csv", index=False)
combined[['bmd','t_score','t_category','fracture_binary']].to_csv("outputs/targets.csv", index=False)

with open("outputs/scaler.pkl","wb") as f: pickle.dump(scaler, f)
with open("outputs/feature_cols.pkl","wb") as f: pickle.dump(FEATURE_COLS, f)

print("\n✅ Files saved to outputs/")
print("▶  Next: python step2_eda.py")

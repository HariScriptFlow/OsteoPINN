"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 5: Single Patient Prediction                               ║
║  Run: python step5_predict.py                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle, warnings, os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
warnings.filterwarnings("ignore")

# ── Load scaler and feature cols ──
with open("outputs/scaler.pkl","rb") as f:  scaler = pickle.load(f)
with open("outputs/feature_cols.pkl","rb") as f: feat_cols = pickle.load(f)

YOUNG_MEAN, YOUNG_SD = 0.964, 0.122

# ── Reload model ──
class PINNOsteoporosis(nn.Module):
    def __init__(self, input_dim=15, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.7),
        )
        self.bmd_head    = nn.Sequential(nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        self.tscore_head = nn.Sequential(nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        self.frac_head   = nn.Sequential(nn.Linear(64,16), nn.SiLU(), nn.Linear(16,1))
        self.tcat_head   = nn.Sequential(nn.Linear(64,16), nn.SiLU(), nn.Linear(16,3))
    def forward(self, x):
        h = self.encoder(x)
        return (self.bmd_head(h).squeeze(-1),
                self.tscore_head(h).squeeze(-1),
                self.frac_head(h).squeeze(-1),
                self.tcat_head(h))

model = PINNOsteoporosis(input_dim=len(feat_cols), dropout=0.35)
state = torch.load("outputs/models/pinn_model.pth", map_location='cpu')
model.load_state_dict(state)
model.eval()

TCAT_NAMES = {0:"Normal 🟢", 1:"Osteopenia 🟡", 2:"Osteoporosis 🔴"}

def predict_patient(age, sex, weight_kg, height_cm, medication_risk=0,
                    family_history=0, postmenopausal=None,
                    low_calcium=0, low_vitd=0, sedentary=0,
                    underweight=0, smoking=0, alcohol=0):
    """
    Predict BMD, T-score, Fracture Risk, T-Category for a single patient.
    """
    sex_binary = 1 if str(sex).lower() in ['female','f'] else 0
    if postmenopausal is None:
        postmenopausal = 1 if (sex_binary == 1 and age > 50) else 0

    bmi = weight_kg / (height_cm / 100) ** 2
    underweight = max(underweight, 1 if bmi < 18.5 else 0)

    risk_score = (
        postmenopausal*3 + low_calcium*2 + low_vitd*2 + sedentary +
        underweight*2 + smoking*2 + alcohol + family_history*2 + medication_risk
    )

    raw_features = {
        'age': age, 'sex_binary': sex_binary, 'weight_kg': weight_kg,
        'height_cm': height_cm, 'bmi': bmi, 'medication_risk': medication_risk,
        'family_history': family_history, 'postmenopausal': postmenopausal,
        'low_calcium': low_calcium, 'low_vitd': low_vitd, 'sedentary': sedentary,
        'underweight': underweight, 'smoking': smoking, 'alcohol': alcohol,
        'risk_score': risk_score
    }

    row = pd.DataFrame([raw_features])[feat_cols]
    row_scaled = scaler.transform(row).astype(np.float32)
    x_tensor   = torch.tensor(row_scaled)

    with torch.no_grad():
        bmd_p, ts_p, fr_p, tc_p = model(x_tensor)
        bmd_pred  = float(bmd_p[0])
        ts_pred   = float(ts_p[0])
        frac_prob = float(torch.sigmoid(fr_p[0]))
        tcat_pred = int(tc_p[0].argmax())

    # Future BMD projections
    def project_bmd(bmd_now, age_now, sex_bin, years):
        k = 0.007 if sex_bin else 0.005
        b = bmd_now
        for a in range(age_now, age_now + years):
            if a > 50 and sex_bin: k = 0.012
            b = max(0.3, b - k * b)
        return round(b, 4)

    future_5yr  = project_bmd(bmd_pred, age, sex_binary, 5)
    future_10yr = project_bmd(bmd_pred, age, sex_binary, 10)

    return {
        'BMD (g/cm²)'        : round(bmd_pred, 4),
        'T-Score'            : round(ts_pred, 2),
        'Category'           : TCAT_NAMES[tcat_pred],
        'Fracture Risk'      : f"{frac_prob*100:.1f}%",
        'BMD in 5 years'     : future_5yr,
        'BMD in 10 years'    : future_10yr,
        'Risk Score'         : risk_score,
    }


# ─────────────────────────────────────────────────────────────────
# DEMO PREDICTIONS
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("🔬 PINN — Patient Fracture Risk Predictions")
    print("="*55)

    patients = [
        dict(name="Healthy Young Woman",   age=35, sex='Female', weight_kg=65, height_cm=162,
             low_calcium=0, low_vitd=0, sedentary=0),
        dict(name="Postmenopausal Woman",   age=62, sex='Female', weight_kg=58, height_cm=158,
             postmenopausal=1, low_calcium=1, low_vitd=1, sedentary=1),
        dict(name="Elderly Man on Steroids",age=75, sex='Male',   weight_kg=70, height_cm=175,
             medication_risk=2, smoking=1, sedentary=1),
        dict(name="Young Underweight Woman",age=28, sex='Female', weight_kg=42, height_cm=160,
             underweight=1, low_calcium=1),
    ]

    for p in patients:
        name = p.pop('name')
        result = predict_patient(**p)
        print(f"\n👤 {name}")
        for k,v in result.items():
            print(f"   {k:<20}: {v}")

    print("\n✅ Prediction module ready.")
    print("   Import predict_patient() in app.py for Streamlit UI")

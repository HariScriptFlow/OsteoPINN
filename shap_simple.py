import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Load
with open('outputs/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('outputs/feature_cols.pkl', 'rb') as f:
    feat_cols = pickle.load(f)

# Model
class PINNOsteoporosis(nn.Module):
    def __init__(self, input_dim=15, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(dropout * 0.7),
        )
        self.bmd_head = nn.Sequential(nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        self.tscore_head = nn.Sequential(nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        self.frac_head = nn.Sequential(nn.Linear(64,16), nn.SiLU(), nn.Linear(16,1))
        self.tcat_head = nn.Sequential(nn.Linear(64,16), nn.SiLU(), nn.Linear(16,3))
    
    def forward(self, x):
        h = self.encoder(x)
        return (self.bmd_head(h).squeeze(-1), self.tscore_head(h).squeeze(-1),
                self.frac_head(h).squeeze(-1), self.tcat_head(h))

model = PINNOsteoporosis(input_dim=len(feat_cols), dropout=0.35)
state = torch.load('outputs/models/pinn_model.pth', map_location='cpu')
model.load_state_dict(state)
model.eval()

print('✓ Model loaded')

def analyze_features(raw_features_dict):
    """Analyze feature importance using SHAP"""
    try:
        import shap
        
        row = pd.DataFrame([raw_features_dict])[feat_cols]
        X_scaled = scaler.transform(row).astype(np.float32)
        
        background = np.random.normal(0, 1, (100, len(feat_cols))).astype(np.float32)
        
        def predict_bmd(X):
            X_t = torch.tensor(X.astype(np.float32))
            with torch.no_grad():
                bmd, _, _, _ = model(X_t)
            return bmd.numpy()
        
        explainer = shap.KernelExplainer(predict_bmd, background)
        shap_values = explainer.shap_values(X_scaled)
        
        result = pd.DataFrame({
            'Feature': feat_cols,
            'Impact': shap_values[0],
            'Importance': np.abs(shap_values[0])
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top = result.head(12)
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top['Impact']]
        ax.barh(range(len(top)), top['Importance'], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('SHAP Feature Importance')
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/shap_analysis.png', dpi=300, bbox_inches='tight')
        result.to_csv('outputs/shap_features.csv', index=False)
        
        print('✓ SHAP complete')
        
        return result, fig
    
    except Exception as e:
        print(f'✗ Error: {e}')
        return None, None


if __name__ == '__main__':
    test_patient = {
        'age': 60, 'sex_binary': 1, 'weight_kg': 60, 'height_cm': 160,
        'bmi': 60/(1.6**2), 'medication_risk': 0, 'family_history': 1,
        'postmenopausal': 1, 'low_calcium': 1, 'low_vitd': 1,
        'sedentary': 1, 'underweight': 0, 'smoking': 0, 'alcohol': 0,
        'risk_score': 10
    }
    
    print('Running SHAP analysis...')
    result, fig = analyze_features(test_patient)
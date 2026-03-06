"""
╔══════════════════════════════════════════════════════════════════╗
║  SHAP INTEGRATION — Model Interpretability Module                ║
║  For Streamlit App (app.py) & Batch Analysis                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class SHAPOsteoporosis:
    """SHAP Analysis for Osteoporosis PINN Model"""
    
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.explainer = None
        
    def create_explainer(self, n_samples=100):
        """Create SHAP KernelExplainer"""
        try:
            # Create synthetic background data
            background = np.random.normal(0, 1, (n_samples, len(self.feature_cols))).astype(np.float32)
            
            def model_predict(X):
                X_tensor = torch.tensor(X.astype(np.float32))
                with torch.no_grad():
                    bmd_p, _, _, _ = self.model(X_tensor)
                    return bmd_p.numpy()
            
            self.explainer = shap.KernelExplainer(model_predict, background)
            print("✓ SHAP Explainer created")
            return self.explainer
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def explain_patient(self, raw_features_dict):
        """Explain single patient prediction"""
        if self.explainer is None:
            self.create_explainer()
        
        try:
            row = pd.DataFrame([raw_features_dict])[self.feature_cols]
            X_scaled = self.scaler.transform(row).astype(np.float32)
            shap_values = self.explainer.shap_values(X_scaled)
            
            contributions = pd.DataFrame({
                'Feature': self.feature_cols,
                'SHAP_Value': shap_values[0],
                'Abs_SHAP': np.abs(shap_values[0])
            }).sort_values('Abs_SHAP', ascending=False)
            
            return {
                'shap_values': shap_values[0],
                'base_value': self.explainer.expected_value,
                'contributions': contributions
            }
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def plot_contributions(self, explanation_dict):
        """Plot top features"""
        contrib = explanation_dict['contributions'].head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in contrib['SHAP_Value']]
        ax.barh(range(len(contrib)), contrib['SHAP_Value'], color=colors)
        ax.set_yticks(range(len(contrib)))
        ax.set_yticklabels(contrib['Feature'], fontsize=9)
        ax.set_xlabel('SHAP Value (Impact on BMD)', fontsize=10)
        ax.set_title('Top Features Affecting Your BMD Prediction', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        return fig
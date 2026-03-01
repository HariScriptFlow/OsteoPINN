"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 4: Comparison Models + Full Visualization                  ║
║  Models: Random Forest | XGBoost | SVM vs PINN                  ║
║  Run: python step4_comparison.py                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import pickle, os, sys, warnings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                              roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

os.makedirs("outputs/plots", exist_ok=True)
sns.set_theme(style="whitegrid")
TCAT_NAMES  = {0:"Normal", 1:"Osteopenia", 2:"Osteoporosis"}

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & PINN RESULTS
# ─────────────────────────────────────────────────────────────────
print("\n📂 Loading data...")
X_df   = pd.read_csv("outputs/X_scaled.csv")
tgt_df = pd.read_csv("outputs/targets.csv")
X      = X_df.values.astype(np.float32)
y_bmd  = tgt_df['bmd'].values.astype(np.float32)
y_frac = tgt_df['fracture_binary'].values.astype(np.float32)
y_tcat = tgt_df['t_category'].values.astype(int)

te_idx = np.load("outputs/models/test_indices.npy")
tr_idx = np.array([i for i in range(len(X)) if i not in te_idx])

Xtr, Xte = X[tr_idx], X[te_idx]
ytr_bmd, yte_bmd   = y_bmd[tr_idx],  y_bmd[te_idx]
ytr_frac, yte_frac = y_frac[tr_idx], y_frac[te_idx]
ytr_tcat, yte_tcat = y_tcat[tr_idx], y_tcat[te_idx]

with open("outputs/models/pinn_results.pkl","rb") as f:
    pinn_res = pickle.load(f)

# ─────────────────────────────────────────────────────────────────
# TRAIN COMPARISON MODELS
# ─────────────────────────────────────────────────────────────────
print("\n🤖 Training comparison models...")

results = {}

# ── Random Forest ──
print("   Training Random Forest...")
rf_reg  = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_cls  = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_reg.fit(Xtr, ytr_bmd)
rf_cls.fit(Xtr, ytr_tcat)

rf_bmd_pred  = rf_reg.predict(Xte)
rf_tcat_pred = rf_cls.predict(Xte)
rf_frac_prob = rf_cls.predict_proba(Xte)
rf_frac_prob_bin = (rf_tcat_pred >= 2).astype(float)  # Osteoporosis = fracture risk

results['Random Forest'] = {
    'bmd_pred': rf_bmd_pred,
    'tcat_pred': rf_tcat_pred,
    'frac_prob': rf_frac_prob_bin,
}

# ── SVM ──
print("   Training SVM...")
svr = SVR(kernel='rbf', C=10, gamma='scale')
svc = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svr.fit(Xtr, ytr_bmd)
svc.fit(Xtr, ytr_tcat)

svm_bmd_pred  = svr.predict(Xte)
svm_tcat_pred = svc.predict(Xte)
svm_frac_prob = (svm_tcat_pred >= 2).astype(float)

results['SVM'] = {
    'bmd_pred': svm_bmd_pred,
    'tcat_pred': svm_tcat_pred,
    'frac_prob': svm_frac_prob,
}

# ── XGBoost ──
if HAS_XGB:
    print("   Training XGBoost...")
    xgb_reg = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                            random_state=42, verbosity=0)
    xgb_cls = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             random_state=42, verbosity=0, use_label_encoder=False,
                             eval_metric='mlogloss')
    xgb_reg.fit(Xtr, ytr_bmd)
    xgb_cls.fit(Xtr, ytr_tcat)

    xgb_bmd_pred  = xgb_reg.predict(Xte)
    xgb_tcat_pred = xgb_cls.predict(Xte)
    xgb_frac_prob = (xgb_tcat_pred >= 2).astype(float)

    results['XGBoost'] = {
        'bmd_pred': xgb_bmd_pred,
        'tcat_pred': xgb_tcat_pred,
        'frac_prob': xgb_frac_prob,
    }

# Add PINN results
results['PINN (Ours)'] = {
    'bmd_pred': pinn_res['bmd_pred'],
    'tcat_pred': pinn_res['tcat_pred'],
    'frac_prob': pinn_res['frac_prob'],
}

# ─────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────
metrics = {}
for name, res in results.items():
    bp = res['bmd_pred'][:len(yte_bmd)]
    cp = res['tcat_pred'][:len(yte_tcat)]
    fp = res['frac_prob'][:len(yte_frac)]

    metrics[name] = {
        'BMD MAE':  round(mean_absolute_error(yte_bmd, bp), 4),
        'BMD R²':   round(r2_score(yte_bmd, bp), 3),
        'T-Cat Acc':round(accuracy_score(yte_tcat, cp), 3),
        'Frac AUC': round(roc_auc_score(yte_frac, fp) if len(np.unique(yte_frac))>1 else 0.5, 3),
    }

metrics_df = pd.DataFrame(metrics).T
print("\n" + "="*60)
print("📊 MODEL COMPARISON RESULTS")
print("="*60)
print(metrics_df.to_string())
metrics_df.to_csv("outputs/model_comparison.csv")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION 1: Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Comparison: PINN vs Baseline Models", fontsize=15, fontweight='bold')

model_names = list(metrics.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
if len(model_names) < len(colors): colors = colors[:len(model_names)]

metric_keys = ['BMD MAE', 'BMD R²', 'T-Cat Acc', 'Frac AUC']
titles      = ['BMD MAE (↓ better)', 'BMD R² (↑ better)',
               'T-Category Accuracy (↑ better)', 'Fracture Risk AUC (↑ better)']

for ax, mk, title in zip(axes.flat, metric_keys, titles):
    vals = [metrics[m][mk] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(mk)

    # Highlight PINN bar
    pinn_idx = model_names.index('PINN (Ours)')
    bars[pinn_idx].set_edgecolor('#000')
    bars[pinn_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig("outputs/plots/05_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n   ✅ Saved: 05_model_comparison.png")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION 2: BMD Prediction Scatter (all models)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
fig.suptitle("Predicted vs Actual BMD", fontsize=14, fontweight='bold')

if len(results) == 1: axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    bp = res['bmd_pred'][:len(yte_bmd)]
    r2 = r2_score(yte_bmd, bp)
    ax.scatter(yte_bmd, bp, alpha=0.4, s=20, color='#3498db' if name=='PINN (Ours)' else '#95a5a6')
    lims = [min(yte_bmd.min(), bp.min()), max(yte_bmd.max(), bp.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect')
    ax.set_xlabel("Actual BMD"); ax.set_ylabel("Predicted BMD")
    ax.set_title(f"{name}\nR²={r2:.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/plots/06_bmd_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 06_bmd_scatter.png")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION 3: Confusion Matrices
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
fig.suptitle("T-Category Confusion Matrices", fontsize=14, fontweight='bold')
if len(results) == 1: axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    cp = res['tcat_pred'][:len(yte_tcat)]
    cm = confusion_matrix(yte_tcat, cp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=TCAT_NAMES.values(),
                yticklabels=TCAT_NAMES.values())
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/plots/07_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 07_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION 4: Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────
with open("outputs/feature_cols.pkl","rb") as f:
    feat_cols = pickle.load(f)

importances = rf_reg.feature_importances_
sorted_idx  = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 7))
colors_fi = ['#e74c3c' if importances[i] > np.median(importances) else '#3498db' for i in sorted_idx]
ax.barh([feat_cols[i] for i in sorted_idx], importances[sorted_idx], color=colors_fi, edgecolor='white')
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Feature Importance for BMD Prediction", fontweight='bold')
ax.axvline(np.median(importances), color='black', ls='--', lw=1, label='Median')
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plots/08_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 08_feature_importance.png")

# ─────────────────────────────────────────────────────────────────
# VISUALIZATION 5: ROC Curves
# ─────────────────────────────────────────────────────────────────
if len(np.unique(yte_frac)) > 1:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random')
    model_colors = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    for (name, res), color in zip(results.items(), model_colors):
        fp = res['frac_prob'][:len(yte_frac)]
        try:
            fpr, tpr, _ = roc_curve(yte_frac, fp)
            auc = roc_auc_score(yte_frac, fp)
            lw = 3 if name == 'PINN (Ours)' else 1.5
            ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.3f})")
        except: pass

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fracture Risk", fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/plots/09_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 09_roc_curves.png")

print("\n✅ All comparisons done!")
print("▶  Next: python step5_predict.py (single patient prediction)")
print("▶  Then: streamlit run app.py")

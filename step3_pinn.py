"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 3 (IMPROVED): Physics-Informed Neural Network              ║
║  Fixes: Early stopping, higher dropout, weight decay,            ║
║         BatchNorm, label smoothing, warm restarts                ║
║  Run: python step3_pinn.py                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pickle, os, sys, warnings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
warnings.filterwarnings("ignore")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("📂 Loading preprocessed data...")
X_df    = pd.read_csv("outputs/X_scaled.csv")
tgt_df  = pd.read_csv("outputs/targets.csv")
df_full = pd.read_csv("outputs/combined_dataset.csv")

X        = X_df.values.astype(np.float32)
y_bmd    = tgt_df['bmd'].values.astype(np.float32)
y_tscore = tgt_df['t_score'].values.astype(np.float32)
y_frac   = tgt_df['fracture_binary'].values.astype(np.float32)
y_tcat   = tgt_df['t_category'].values.astype(np.int64)
ages_raw = df_full['age'].values.astype(np.float32)
sex_raw  = df_full['sex_binary'].values.astype(np.float32)

print(f"   X shape: {X.shape} | Samples: {len(X)}")

# Split
idx = np.arange(len(X))
tr_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_tcat)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=42, stratify=y_tcat[tr_idx])

def make_tensors(i):
    return (torch.tensor(X[i]),
            torch.tensor(y_bmd[i]),
            torch.tensor(y_tscore[i]),
            torch.tensor(y_frac[i]),
            torch.tensor(y_tcat[i]).long(),
            torch.tensor(ages_raw[i]),
            torch.tensor(sex_raw[i]))

Xtr,Btr,Ttr,Ftr,Ctr,Atr,Str = make_tensors(tr_idx)
Xva,Bva,Tva,Fva,Cva,Ava,Sva = make_tensors(va_idx)
Xte,Bte,Tte,Fte,Cte,Ate,Ste = make_tensors(te_idx)

train_dl = DataLoader(TensorDataset(Xtr,Btr,Ttr,Ftr,Ctr,Atr,Str), batch_size=64, shuffle=True)
val_dl   = DataLoader(TensorDataset(Xva,Bva,Tva,Fva,Cva,Ava,Sva), batch_size=64, shuffle=False)

print(f"   Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")

# ─────────────────────────────────────────────────────────────────
# IMPROVED PINN — BatchNorm + higher dropout + smaller network
# ─────────────────────────────────────────────────────────────────
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

YOUNG_MEAN = 0.964

def physics_loss(bmd_pred, age, sex):
    k = torch.where(sex > 0.5,
                    torch.full_like(sex, 0.007),
                    torch.full_like(sex, 0.005))
    bmd_expected = 0.96 * torch.exp(-k * torch.clamp(age - 50, min=0))
    bmd_expected -= 0.07 * sex
    return ((bmd_pred - bmd_expected.float()) ** 2).mean()

# ─────────────────────────────────────────────────────────────────
# TRAINING — AdamW + warm restarts + early stopping
# ─────────────────────────────────────────────────────────────────
model     = PINNOsteoporosis(input_dim=X.shape[1]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.05)

LAMBDA_PHYSICS = 0.20
EPOCHS         = 200
PATIENCE       = 25

history = {'train_loss':[], 'val_loss':[], 'val_r2_bmd':[], 'val_auc_frac':[]}
best_val_loss  = float('inf')
best_state     = None
patience_count = 0
best_epoch     = 1

print(f"\n🚀 Training improved PINN (max {EPOCHS} epochs, patience={PATIENCE})...")
print(f"   dropout=0.35 | weight_decay=1e-3 | label_smoothing=0.05\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_dl:
        xb,bb,tb,fb,cb,ab,sb = [t.to(DEVICE) for t in batch]
        bmd_p, ts_p, fr_p, tc_p = model(xb)
        loss = (mse_loss(bmd_p, bb) +
                0.4 * mse_loss(ts_p, tb) +
                0.8 * bce_loss(fr_p, fb) +
                0.3 * ce_loss(tc_p, cb.long()) +
                LAMBDA_PHYSICS * physics_loss(bmd_p, ab, sb))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0
    bmd_preds, bmd_trues = [], []
    frac_preds, frac_trues = [], []

    with torch.no_grad():
        for batch in val_dl:
            xb,bb,tb,fb,cb,ab,sb = [t.to(DEVICE) for t in batch]
            bmd_p, ts_p, fr_p, tc_p = model(xb)
            vloss = (mse_loss(bmd_p, bb) +
                     0.4 * mse_loss(ts_p, tb) +
                     0.8 * bce_loss(fr_p, fb) +
                     0.3 * ce_loss(tc_p, cb.long()) +
                     LAMBDA_PHYSICS * physics_loss(bmd_p, ab, sb))
            val_loss += vloss.item()
            bmd_preds.extend(bmd_p.cpu().numpy())
            bmd_trues.extend(bb.cpu().numpy())
            frac_preds.extend(torch.sigmoid(fr_p).cpu().numpy())
            frac_trues.extend(fb.cpu().numpy())

    train_loss /= len(train_dl)
    val_loss   /= len(val_dl)
    r2  = r2_score(bmd_trues, bmd_preds)
    try: auc = roc_auc_score(frac_trues, frac_preds)
    except: auc = 0.5

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_r2_bmd'].append(r2)
    history['val_auc_frac'].append(auc)

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
        best_epoch     = epoch + 1
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch+1} (best: epoch {best_epoch})")
            break

    if (epoch+1) % 20 == 0:
        print(f"   Epoch {epoch+1:3d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"R²: {r2:.3f}  AUC: {auc:.3f}")

# ─────────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────────
print(f"\n✅ Best model from epoch {best_epoch}")
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    bmd_p, ts_p, fr_p, tc_p = model(Xte.to(DEVICE))
    bmd_pred  = bmd_p.cpu().numpy()
    ts_pred   = ts_p.cpu().numpy()
    frac_prob = torch.sigmoid(fr_p).cpu().numpy()
    tcat_pred = tc_p.argmax(dim=1).cpu().numpy()

bmd_true  = Bte.numpy()
ts_true   = Tte.numpy()
frac_true = Fte.numpy()
tcat_true = Cte.numpy()

mae_bmd  = mean_absolute_error(bmd_true, bmd_pred)
r2_bmd   = r2_score(bmd_true, bmd_pred)
mae_ts   = mean_absolute_error(ts_true, ts_pred)
r2_ts    = r2_score(ts_true, ts_pred)
auc_frac = roc_auc_score(frac_true, frac_prob)
acc_cat  = accuracy_score(tcat_true, tcat_pred)

print("\n" + "="*55)
print("📈 IMPROVED PINN — FINAL TEST RESULTS")
print("="*55)
print(f"   BMD     MAE: {mae_bmd:.4f}  R²: {r2_bmd:.3f}")
print(f"   T-score MAE: {mae_ts:.4f}  R²: {r2_ts:.3f}")
print(f"   Fracture AUC:    {auc_frac:.3f}")
print(f"   T-Category Acc:  {acc_cat:.3f}")
print(f"   Stopped at:      epoch {best_epoch}")
print("="*55)

# Save
torch.save(best_state, "outputs/models/pinn_model.pth")
np.save("outputs/models/test_indices.npy", te_idx)
results = {'bmd_pred':bmd_pred,'bmd_true':bmd_true,'ts_pred':ts_pred,'ts_true':ts_true,
           'frac_prob':frac_prob,'frac_true':frac_true,'tcat_pred':tcat_pred,'tcat_true':tcat_true}
with open("outputs/models/pinn_results.pkl","wb") as f: pickle.dump(results,f)
with open("outputs/models/history.pkl","wb") as f:      pickle.dump(history,f)

# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────
ep = range(1, len(history['train_loss'])+1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Improved PINN Training History", fontsize=14, fontweight='bold')

axes[0].plot(ep, history['train_loss'], color='#3498db', lw=2, label='Train')
axes[0].plot(ep, history['val_loss'],   color='#e74c3c', lw=2, label='Val')
axes[0].axvline(best_epoch, color='green', ls='--', lw=1.5, label=f'Best={best_epoch}')
axes[0].fill_between(ep, history['train_loss'], history['val_loss'], alpha=0.08, color='purple')
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8)

axes[1].plot(ep, history['val_r2_bmd'], color='#2ecc71', lw=2)
axes[1].axhline(r2_bmd, color='#2ecc71', ls='--', lw=1.5, label=f'Test R²={r2_bmd:.3f}')
axes[1].axhline(0.7, color='gray', ls=':', lw=1, label='Target 0.7')
axes[1].set_title("Val R² (BMD)"); axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0,1); axes[1].legend(fontsize=8)

axes[2].plot(ep, history['val_auc_frac'], color='#f39c12', lw=2)
axes[2].axhline(auc_frac, color='#f39c12', ls='--', lw=1.5, label=f'Test AUC={auc_frac:.3f}')
axes[2].axhline(0.8, color='gray', ls=':', lw=1, label='Clinical 0.8')
axes[2].set_title("Val AUC (Fracture)"); axes[2].set_xlabel("Epoch")
axes[2].set_ylim(0.4,1); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/plots/04_pinn_training.png", dpi=150, bbox_inches='tight')
plt.close()

# Before/after comparison
fig, ax = plt.subplots(figsize=(9,5))
metrics    = ['BMD R²', 'T-score R²', 'Fracture AUC', 'T-Cat Acc']
old_scores = [0.548, 0.688, 0.859, 0.812]
new_scores = [r2_bmd, r2_ts, auc_frac, acc_cat]
x = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x-w/2, old_scores, w, label='Previous PINN', color='#95a5a6', edgecolor='white')
b2 = ax.bar(x+w/2, new_scores, w, label='Improved PINN', color='#2ecc71', edgecolor='white')
for bar,v in zip(b1,old_scores): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.3f}', ha='center', fontsize=9)
for bar,v in zip(b2,new_scores): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0,1.1); ax.set_title("Previous vs Improved PINN", fontsize=13, fontweight='bold')
ax.legend(); ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("outputs/plots/10_pinn_improvement.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n   ✅ Saved: 04_pinn_training.png")
print("   ✅ Saved: 10_pinn_improvement.png")
print("   ✅ Saved: outputs/models/pinn_model.pth")
print("\n▶  Next: python step4_comparison.py")
print("▶  Then: python step5_predict.py")
print("▶  Then: & \"C:\\Program Files\\Python313\\python.exe\" -m streamlit run app.py")

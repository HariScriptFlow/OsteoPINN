"""
╔══════════════════════════════════════════════════════════════════╗
║  STREAMLIT APP — Osteoporosis Risk Predictor                     ║
║  Run: streamlit run app.py                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import pickle, os

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OsteoPINN — Bone Health AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center; color: white;
        margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; margin-top: 0.2rem; }
    .card-normal     { background: linear-gradient(135deg,#2ecc71,#27ae60); border-radius:12px; padding:1rem; color:white; text-align:center; }
    .card-osteopenia { background: linear-gradient(135deg,#f39c12,#e67e22); border-radius:12px; padding:1rem; color:white; text-align:center; }
    .card-osteoporosis{ background: linear-gradient(135deg,#e74c3c,#c0392b); border-radius:12px; padding:1rem; color:white; text-align:center; }
    .info-box { background:#f0f4ff; border-left:4px solid #667eea; padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; color:#1a1a2e !important; font-size:1rem; }
    .section-header { font-size:1.3rem; font-weight:700; color:#1a1a2e; border-bottom:2px solid #667eea; padding-bottom:0.3rem; margin:1.5rem 0 0.8rem 0; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
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

    with open("outputs/scaler.pkl","rb") as f:  scaler = pickle.load(f)
    with open("outputs/feature_cols.pkl","rb") as f: feat_cols = pickle.load(f)
    model = PINNOsteoporosis(input_dim=len(feat_cols), dropout=0.35)
    state = torch.load("outputs/models/pinn_model.pth", map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    for m in model.modules():
        if isinstance(m, __import__("torch").nn.BatchNorm1d): m.eval()
    return model, scaler, feat_cols

model_loaded = os.path.exists("outputs/models/pinn_model.pth")
if model_loaded:
    model, scaler, feat_cols = load_model()

# ─────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────
def predict_patient(age, sex_binary, weight_kg, height_cm, medication_risk,
                    family_history, postmenopausal, low_calcium, low_vitd,
                    sedentary, underweight, smoking, alcohol):
    bmi = weight_kg / (height_cm / 100) ** 2
    underweight = max(underweight, 1 if bmi < 18.5 else 0)

    risk_score = (postmenopausal*3 + low_calcium*2 + low_vitd*2 + sedentary +
                  underweight*2 + smoking*2 + alcohol + family_history*2 + medication_risk)

    raw = {'age':age,'sex_binary':sex_binary,'weight_kg':weight_kg,'height_cm':height_cm,
           'bmi':bmi,'medication_risk':medication_risk,'family_history':family_history,
           'postmenopausal':postmenopausal,'low_calcium':low_calcium,'low_vitd':low_vitd,
           'sedentary':sedentary,'underweight':underweight,'smoking':smoking,
           'alcohol':alcohol,'risk_score':risk_score}

    row = pd.DataFrame([raw])[feat_cols]
    xs  = scaler.transform(row).astype(np.float32)
    xt  = torch.tensor(xs)

    with torch.no_grad():
        bmd_p, ts_p, fr_p, tc_p = model(xt)
        bmd_pred  = float(bmd_p[0])
        ts_pred   = float(ts_p[0])
        frac_prob = float(torch.sigmoid(fr_p[0]))
        tcat_pred = int(tc_p[0].argmax())

    def proj_bmd(b0, a0, sx, yrs):
        b = b0
        for a in range(a0, a0+yrs):
            k = 0.007 if sx else 0.005
            if a > 50 and sx: k = 0.012
            b = max(0.3, b - k * b)
        return round(b, 4)

    return {
        'bmd': round(bmd_pred, 4),
        'tscore': round(ts_pred, 2),
        'frac_prob': round(frac_prob * 100, 1),
        'tcat': tcat_pred,
        'bmd_5yr': proj_bmd(bmd_pred, age, sex_binary, 5),
        'bmd_10yr': proj_bmd(bmd_pred, age, sex_binary, 10),
        'bmd_20yr': proj_bmd(bmd_pred, age, sex_binary, 20),
        'risk_score': risk_score,
    }

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🦴 OsteoPINN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Physics-Informed Neural Network for Bone Health & Osteoporosis Risk Prediction</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — Patient Input
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Patient Profile")
    st.markdown("---")

    age         = st.slider("Age (years)", 18, 95, 55)
    sex         = st.radio("Sex", ["Female", "Male"], horizontal=True)
    sex_binary  = 1 if sex == "Female" else 0
    weight_kg   = st.slider("Weight (kg)", 35, 120, 65)
    height_cm   = st.slider("Height (cm)", 140, 200, 162)

    st.markdown("---")
    st.markdown("### 💊 Medical History")
    medication  = st.selectbox("Current Medication",
                               ["No medication", "Glucocorticoids", "Anticonvulsant", "Other"])
    med_map     = {"No medication":0, "Glucocorticoids":2, "Anticonvulsant":1, "Other":1}
    med_risk    = med_map[medication]
    family_hist = st.toggle("Family history of osteoporosis")
    postmenop   = st.toggle("Postmenopausal", value=(sex=="Female" and age>50))

    st.markdown("---")
    st.markdown("### 🥗 Lifestyle Factors")
    low_calcium = st.toggle("Low calcium intake")
    low_vitd    = st.toggle("Insufficient Vitamin D")
    sedentary   = st.toggle("Sedentary lifestyle")
    underweight_t= st.toggle("Underweight (BMI < 18.5)")
    smoking     = st.toggle("Smoker")
    alcohol     = st.toggle("Alcohol consumption")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Now", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────
if not model_loaded:
    st.warning("⚠️ Model not found. Please run `python step3_pinn.py` first to train the model.")
    st.stop()

# Auto-predict or on button press
result = predict_patient(
    age, sex_binary, weight_kg, height_cm, med_risk,
    int(family_hist), int(postmenop), int(low_calcium),
    int(low_vitd), int(sedentary), int(underweight_t),
    int(smoking), int(alcohol)
)

tcat_names   = {0:"Normal", 1:"Osteopenia", 2:"Osteoporosis"}
tcat_colors  = {0:"#2ecc71", 1:"#f39c12", 2:"#e74c3c"}
tcat_cards   = {0:"card-normal", 1:"card-osteopenia", 2:"card-osteoporosis"}
tcat_icons   = {0:"🟢", 1:"🟡", 2:"🔴"}

# ── Row 1: Key Metrics ──
st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Bone Mineral Density", f"{result['bmd']} g/cm²",
              delta=f"{result['bmd']-0.964:.3f} vs avg")

with col2:
    st.metric("T-Score", f"{result['tscore']}",
              delta="Normal" if result['tscore'] >= -1 else ("Osteopenia" if result['tscore'] >= -2.5 else "Osteoporosis"))

with col3:
    st.metric("Fracture Risk", f"{result['frac_prob']}%")

with col4:
    st.metric("Risk Score", f"{result['risk_score']}/18")

# ── Category Banner ──
tcat = result['tcat']
st.markdown(f"""
<div class="{tcat_cards[tcat]}" style="margin:1rem 0; padding:1.2rem;">
    <div style="font-size:2rem;">{tcat_icons[tcat]} {tcat_names[tcat]}</div>
    <div style="font-size:0.9rem; opacity:0.9; margin-top:0.4rem;">
        T-Score: {result['tscore']} &nbsp;|&nbsp; Fracture Risk: {result['frac_prob']}%
    </div>
</div>
""", unsafe_allow_html=True)

# ── Row 2: BMD Trajectory + Risk Gauge ──
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">📈 BMD Trajectory (Next 20 Years)</div>', unsafe_allow_html=True)

    ages_proj = list(range(age, age+21))
    k = 0.007 if sex_binary else 0.005

    bmd_proj = [result['bmd']]
    b = result['bmd']
    for a in range(age+1, age+21):
        kr = 0.012 if (a>50 and sex_binary) else k
        b = max(0.3, b - kr * b)
        bmd_proj.append(b)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(ages_proj, 0.833, 1.4,  alpha=0.08, color='#2ecc71', label='Normal')
    ax.fill_between(ages_proj, 0.648, 0.833,alpha=0.10, color='#f39c12', label='Osteopenia')
    ax.fill_between(ages_proj, 0.3,   0.648, alpha=0.10, color='#e74c3c', label='Osteoporosis')
    ax.axhline(0.833, color='#f39c12', ls='--', lw=1.2)
    ax.axhline(0.648, color='#e74c3c', ls='--', lw=1.2)
    ax.plot(ages_proj, bmd_proj, color='#2c3e50', lw=2.5, marker='o', markersize=3, label='Your BMD')
    ax.scatter([age], [result['bmd']], color='#e74c3c', s=100, zorder=5, label='Current')
    ax.scatter([age+5, age+10], [result['bmd_5yr'], result['bmd_10yr']],
               color='#f39c12', s=80, zorder=5, marker='D')
    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_ylabel("BMD (g/cm²)", fontsize=10)
    ax.set_title("Projected BMD Over Time", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0.25, 1.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.markdown('<div class="section-header">🔮 Future Projections</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
            <div style="background:rgba(255,255,255,0.07);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="color:#a0aec0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Now</div>
                <div style="color:#ffffff;font-size:1.5rem;font-weight:800;">{result['bmd']}</div>
                <div style="color:#a0aec0;font-size:0.75rem;">g/cm²</div>
            </div>
            <div style="background:rgba(46,204,113,0.15);border:1px solid rgba(46,204,113,0.3);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="color:#2ecc71;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">In 5 Years</div>
                <div style="color:#2ecc71;font-size:1.5rem;font-weight:800;">{result['bmd_5yr']}</div>
                <div style="color:#2ecc71;font-size:0.75rem;">g/cm²</div>
            </div>
            <div style="background:rgba(243,156,18,0.15);border:1px solid rgba(243,156,18,0.3);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="color:#f39c12;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">In 10 Years</div>
                <div style="color:#f39c12;font-size:1.5rem;font-weight:800;">{result['bmd_10yr']}</div>
                <div style="color:#f39c12;font-size:0.75rem;">g/cm²</div>
            </div>
            <div style="background:rgba(231,76,60,0.15);border:1px solid rgba(231,76,60,0.3);border-radius:10px;padding:0.8rem;text-align:center;">
                <div style="color:#e74c3c;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">In 20 Years</div>
                <div style="color:#e74c3c;font-size:1.5rem;font-weight:800;">{result['bmd_20yr']}</div>
                <div style="color:#e74c3c;font-size:0.75rem;">g/cm²</div>
            </div>
        </div>
        <div style="margin-top:0.8rem;text-align:center;color:#718096;font-size:0.75rem;">
            ↓ BMD declining · green=safe · orange=watch · red=critical
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
    risk_factors = []
    if postmenop:       risk_factors.append(("🔴 Postmenopausal", 3))
    if low_calcium:     risk_factors.append(("🟠 Low Calcium", 2))
    if low_vitd:        risk_factors.append(("🟠 Low Vitamin D", 2))
    if underweight_t:   risk_factors.append(("🟠 Underweight", 2))
    if smoking:         risk_factors.append(("🟠 Smoking", 2))
    if family_hist:     risk_factors.append(("🟡 Family History", 2))
    if sedentary:       risk_factors.append(("🟡 Sedentary", 1))
    if alcohol:         risk_factors.append(("🟡 Alcohol", 1))
    if med_risk >= 2:   risk_factors.append(("🔴 Glucocorticoids", 2))

    if risk_factors:
        for factor, score in sorted(risk_factors, key=lambda x: -x[1]):
            st.markdown(f"• {factor}")
    else:
        st.success("✅ No major risk factors identified!")

# ── Row 3: WHO T-Score Scale ──
st.markdown('<div class="section-header">📏 WHO T-Score Classification</div>', unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(10, 1.5))
ax2.barh([0], [1], color='#2ecc71', height=0.4, label='Normal (T ≥ -1.0)')
ax2.barh([0], [1.5], left=1,  color='#f39c12', height=0.4, label='Osteopenia (-2.5 to -1.0)')
ax2.barh([0], [2.5], left=2.5,color='#e74c3c', height=0.4, label='Osteoporosis (T < -2.5)')

# Mark patient's T-score
ts_norm = (result['tscore'] + 4) / 8
ts_x    = ts_norm * 6
ax2.scatter([ts_x], [0], color='black', s=250, zorder=10, marker='v', label=f"Your T-score: {result['tscore']}")
ax2.set_xlim(0, 6); ax2.axis('off')
ax2.legend(loc='upper right', fontsize=9, ncol=4)
ax2.set_title("T-Score Scale", fontsize=10)
st.pyplot(fig2)
plt.close()

# ── Row 4: Model Comparison (if available) ──
if os.path.exists("outputs/model_comparison.csv"):
    st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)
    comp_df = pd.read_csv("outputs/model_comparison.csv", index_col=0)
    st.dataframe(comp_df.style.highlight_max(axis=0, color='#d4f7d4')
                              .highlight_min(axis=0, subset=['BMD MAE'], color='#d4f7d4')
                              .format("{:.3f}"), use_container_width=True)

# ── Row 5: Saved plots ──
plot_files = {
    "EDA Overview": "outputs/plots/01_eda_overview.png",
    "Correlation Heatmap": "outputs/plots/02_correlation_heatmap.png",
    "BMD Trajectories": "outputs/plots/03_bmd_trajectories.png",
    "Training History": "outputs/plots/04_pinn_training.png",
    "Model Comparison": "outputs/plots/05_model_comparison.png",
    "ROC Curves": "outputs/plots/09_roc_curves.png",
}

available = {k: v for k, v in plot_files.items() if os.path.exists(v)}
if available:
    st.markdown('<div class="section-header">📉 Analysis Plots</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(available.keys()))
    for tab, (name, path) in zip(tabs, available.items()):
        with tab:
            st.image(path, use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem;'>
    🦴 OsteoPINN — Physics-Informed Neural Network | Built with PyTorch + Streamlit<br>
    ⚠️ For research purposes only. Not a substitute for clinical diagnosis.
</div>
""", unsafe_allow_html=True)

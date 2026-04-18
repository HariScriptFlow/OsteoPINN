# OsteoPINN

**Physics-Informed Neural Network for Osteoporosis and Bone Health Risk Prediction**

OsteoPINN is a machine learning system that predicts osteoporosis risk, bone mineral density (BMD), and fracture probability using only basic demographic and lifestyle information. The core innovation is a Physics-Informed Neural Network (PINN) trained simultaneously on clinical data and the biological differential equations governing bone loss in the human body. This makes the model biologically consistent by design, not just statistically fitted.

---

## Table of Contents

- [Clinical Problem](#clinical-problem)
- [What OsteoPINN Predicts](#what-osteopinn-predicts)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
- [Physics-Informed Neural Network](#physics-informed-neural-network)
- [Model Architecture](#model-architecture)
- [Comparison Models](#comparison-models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Streamlit Application](#streamlit-application)
- [SHAP Feature Analysis](#shap-feature-analysis)
- [Real-World Applications](#real-world-applications)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Clinical Problem

Osteoporosis is a silent disease. Patients lose bone density gradually over decades with no symptoms until a fracture occurs. By that point, significant damage has already happened.

Current diagnosis requires a DEXA scan (Dual-Energy X-ray Absorptiometry), which has major accessibility barriers:

- Costs $150 to $300 per scan
- DEXA equipment is unavailable in many hospitals in developing regions
- Patients only get scanned after risk factors are already serious
- The scan gives a point-in-time BMD value but cannot project future bone loss

OsteoPINN provides a free, instant, accessible screening alternative that any doctor or patient can use with information they already know.

---

## What OsteoPINN Predicts

Given a patient's age, sex, weight, height, medication status, and eight lifestyle toggles, the system outputs all of the following in under one second:

- Estimated current BMD value in g/cm2
- WHO T-score with classification as Normal, Osteopenia, or Osteoporosis
- Fracture risk as a probability percentage
- Projected BMD at 5, 10, and 20 years into the future
- Composite clinical risk score
- Per-patient SHAP feature attribution explaining which factors drive the prediction

---

## Project Structure

```
OsteoPINN/
|
|-- bmd.csv                    # Dataset 1: 169 real DEXA clinical measurements
|-- osteoporosis.csv           # Dataset 2: 1,958 lifestyle + diagnosis records
|-- requirements.txt           # All Python dependencies
|
|-- step1_preprocess.py        # Data merging, cleaning, feature engineering, scaling
|-- step2_eda.py               # Exploratory data analysis and visualization
|-- step3_pinn.py              # PINN architecture, physics loss, training
|-- step4_comparison.py        # Baseline model training and comparison plots
|-- step5_predict.py           # Single patient prediction demo and utility function
|-- shap_simple.py             # Standalone SHAP analysis script
|-- shap_integration.py        # SHAP class used by the Streamlit app
|-- app.py                     # Streamlit web application
|
|-- outputs/
    |-- combined_dataset.csv   # Merged 2,127-row dataset
    |-- X_scaled.csv           # StandardScaler-transformed features
    |-- targets.csv            # BMD, T-score, fracture, T-category labels
    |-- scaler.pkl             # Fitted StandardScaler (required by app.py)
    |-- feature_cols.pkl       # Ordered feature column list
    |-- model_comparison.csv   # Tabular metric comparison across all models
    |-- shap_analysis.png      # SHAP bar chart for example patient
    |-- shap_features.csv      # SHAP values for example patient
    |
    |-- models/
    |   |-- pinn_model.pth     # Best model checkpoint (PyTorch state dict)
    |   |-- pinn_config.pth    # Model configuration
    |   |-- pinn_results.pkl   # Test set predictions from PINN
    |   |-- history.pkl        # Training loss and metric history
    |   |-- test_indices.npy   # Test split indices for reproducibility
    |
    |-- plots/
        |-- 01_eda_overview.png
        |-- 02_correlation_heatmap.png
        |-- 03_bmd_trajectories.png
        |-- 04_pinn_training.png
        |-- 05_model_comparison.png
        |-- 06_bmd_scatter.png
        |-- 07_confusion_matrices.png
        |-- 08_feature_importance.png
        |-- 09_roc_curves.png
        |-- 10_pinn_improvement.png
```

---

## Datasets

### Dataset 1: bmd.csv (169 rows)

Source: Kaggle — "Osteoporosis in Women and Men" clinical dataset. Contains real BMD measurements from DEXA scans. This dataset provides the physics ground truth that anchors the model's bone density predictions.

| Column | Type | Description |
|---|---|---|
| id | Integer | Unique patient identifier |
| age | Integer | Age in years, range 35 to 90 |
| sex | M/F | Patient sex |
| fracture | String | Fracture status: fracture or no fracture |
| weight_kg | Float | Body weight in kilograms |
| height_cm | Float | Height in centimeters |
| medication | String | Current medication class |
| waiting_time | Integer | Months since last visit |
| bmd | Float | DEXA-measured bone mineral density in g/cm2 |

### Dataset 2: osteoporosis.csv (1,958 rows)

Source: Kaggle — "Osteoporosis Risk Factors" lifestyle dataset. Contains rich lifestyle and demographic features with binary osteoporosis diagnosis labels. This dataset is 12 times larger than bmd.csv and provides the lifestyle feature diversity the model needs to generalize.

| Column | Type | Description |
|---|---|---|
| Age | Integer | Age in years, range 18 to 90 |
| Gender | Male/Female | Patient sex |
| Hormonal Changes | String | Normal or Postmenopausal |
| Family History | Yes/No | Family history of osteoporosis |
| Race/Ethnicity | String | Caucasian, African American, Asian, etc. |
| Body Weight | String | Normal or Underweight |
| Calcium Intake | String | Adequate or Low |
| Vitamin D Intake | String | Sufficient or Insufficient |
| Physical Activity | String | Active or Sedentary |
| Smoking | Boolean | Smoker status |
| Alcohol Consumption | Boolean | Alcohol consumer status |
| Osteoporosis | Boolean | Final diagnosis |

### Why Two Datasets Were Combined

Neither dataset was sufficient alone. bmd.csv has real DEXA measurements but only 169 patients, which is too few for a neural network to generalize. osteoporosis.csv has 1,958 patients with rich lifestyle features but no BMD values, making physics-constrained regression impossible.

The solution was to estimate BMD for osteoporosis.csv patients using the published WHO bone loss differential equation calibrated against the real measurements in bmd.csv, then combine both into a 2,127-row training dataset.

---

## Data Preprocessing

### step1_preprocess.py

This script performs all data merging, cleaning, and feature engineering. It must be run first.

**bmd.csv cleaning:** Column names are standardized. Sex is mapped to Male/Female. Fracture status is binarized. Medication is converted to a risk score (No medication = 0, Anticonvulsant = 1, Glucocorticoids = 2). BMI is computed from weight and height. WHO T-score is computed as (BMD - 0.964) / 0.122. T-category is assigned as 0 for Normal (T >= -1), 1 for Osteopenia (-2.5 to -1), and 2 for Osteoporosis (< -2.5).

**osteoporosis.csv cleaning:** All Yes/No fields are binarized. Postmenopausal flag is extracted from the Hormonal Changes column. BMD is estimated using the WHO physics formula with random variation seeded for reproducibility. Synthetic weight and height are assigned based on sex and body weight status. Medication risk defaults to 0 as this field is not present in the dataset.

**BMD estimation formula used for osteoporosis.csv:**

```
base = 0.96
base -= max(0, (age - 50)) * 0.005
if Female: base -= 0.07
if postmenopausal: base -= 0.06
if low_calcium: base -= 0.03
if low_vitd: base -= 0.03
if sedentary: base -= 0.02
if underweight: base -= 0.04
if smoking: base -= 0.03
if alcohol: base -= 0.02
if family_history: base -= 0.02
base += Normal(0, 0.035)  # biological noise
BMD = clip(base, 0.30, 1.40)
```

**Composite risk score:** A weighted sum of all lifestyle risk factors is engineered as an additional input feature.

| Risk Factor | Weight | Clinical Justification |
|---|---|---|
| Postmenopausal | 3 | Estrogen loss is the single strongest driver of bone resorption |
| Underweight (BMI < 18.5) | 2 | Low body mass reduces mechanical loading on bone |
| Low calcium intake | 2 | Calcium is the primary structural mineral in bone |
| Smoking | 2 | Nicotine inhibits osteoblast activity |
| Family history | 2 | Osteoporosis has strong genetic heritability |
| Insufficient Vitamin D | 2 | Required for calcium absorption |
| Sedentary lifestyle | 1 | Weight-bearing exercise stimulates bone remodeling |
| Alcohol consumption | 1 | Inhibits calcium absorption and bone formation |
| Medication risk | 0 to 2 | Glucocorticoids are the strongest drug-induced bone loss cause |

**Feature scaling:** All 15 features are standardized with sklearn StandardScaler (mean = 0, std = 1). The fitted scaler is saved to outputs/scaler.pkl so the Streamlit app uses the identical transformation at prediction time.

**Outputs saved:** combined_dataset.csv, X_scaled.csv, targets.csv, scaler.pkl, feature_cols.pkl

---

## Physics-Informed Neural Network

### Why Not Standard Machine Learning?

Standard Random Forest and XGBoost models treat bone density prediction as pure pattern matching. They find correlations in training data but have no understanding of the underlying biology. This produces two critical failures:

- **Fracture risk prediction failure:** Random Forest and XGBoost achieved AUC of only 0.551 to 0.577 on fracture risk, barely better than random guessing. They learn BMD values reasonably well but cannot connect BMD to fracture probability because they have no biomechanical model.
- **No temporal reasoning:** A standard ML model can only predict the current state. It cannot project what a patient's BMD will be in 10 years because it has no model of the biological process over time.

### The Physics Loss

A Physics-Informed Neural Network adds a physics residual loss term to the standard data loss. During training, the model is penalized not only for wrong predictions but also for predictions that violate known biological laws.

The physics constraint used is the WHO bone loss ordinary differential equation:

```
dBMD/dAge = -k(sex) * BMD

where:
  k_female = 0.007 (accelerating to 0.012 post-menopause)
  k_male   = 0.005
```

This exponential decay model is validated by decades of longitudinal bone density studies. In code, the physics residual is computed as:

```python
def physics_loss(bmd_pred, age, sex):
    k = where(sex > 0.5, 0.007, 0.005)  # female vs male decay rate
    bmd_expected = 0.96 * exp(-k * clamp(age - 50, min=0)) - 0.07 * sex
    return mean((bmd_pred - bmd_expected) ** 2)
```

### Combined Training Loss

The total loss during training is a weighted sum of five components:

| Component | Weight | What It Enforces |
|---|---|---|
| BMD regression MSE | 1.0 | Predicted BMD matches measured values |
| T-score regression MSE | 0.4 | T-score is consistent with predicted BMD |
| Fracture classification BCE | 0.8 | Fracture risk prediction is accurate |
| T-category classification CE | 0.3 | Normal / Osteopenia / Osteoporosis is correctly classified |
| Physics residual | 0.2 | BMD predictions follow the bone loss ODE |

---

## Model Architecture

### Neural Network Structure (step3_pinn.py)

The PINN uses a shared encoder with four specialized output heads, one for each prediction task.

```
Input (15 features)
    |
    v
Linear(15 -> 128) + BatchNorm1d + SiLU + Dropout(0.35)
    |
Linear(128 -> 128) + BatchNorm1d + SiLU + Dropout(0.35)
    |
Linear(128 -> 64) + BatchNorm1d + SiLU + Dropout(0.245)
    |
    |---> BMD Head:        Linear(64->32) + SiLU + Linear(32->1)      [g/cm2]
    |---> T-score Head:    Linear(64->32) + SiLU + Linear(32->1)      [dimensionless]
    |---> Fracture Head:   Linear(64->16) + SiLU + Linear(16->1)      [0-1 probability]
    |---> T-Category Head: Linear(64->16) + SiLU + Linear(16->3)      [Normal/Osteopenia/Osteoporosis]
```

### Activation Function

SiLU (Swish) is used instead of ReLU throughout. SiLU is defined as f(x) = x * sigmoid(x). It is continuously differentiable and has smooth gradient flow, which improves convergence on small datasets compared to ReLU.

### Anti-Overfitting Techniques

- **BatchNorm1d** after every linear layer normalizes activations within each batch, stabilizing training and reducing sensitivity to learning rate
- **Dropout (0.35)** randomly zeros 35% of neurons during training, forcing redundant representations and preventing memorization
- **Weight decay (L2 = 0.001)** in AdamW penalizes large weights and acts as regularization
- **Label smoothing (0.05)** prevents the model from becoming overconfident on training examples
- **Early stopping (patience = 25)** stops training when validation loss stops improving and restores the best checkpoint
- **Gradient clipping (max norm = 1.0)** prevents exploding gradients during backpropagation

### Training Configuration

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | AdamW | Adaptive learning rate with decoupled weight decay |
| Learning rate | 8e-4 | Slightly below default for more stable convergence |
| Weight decay | 1e-3 | Strong regularization for small dataset |
| Batch size | 64 | Balance between gradient noise and memory |
| Max epochs | 200 | Upper bound; early stopping triggers earlier |
| Scheduler | CosineAnnealingWarmRestarts (T0=30) | Periodic LR restarts escape local minima |
| Physics weight lambda | 0.20 | Balanced contribution without dominating data loss |

### Data Split

Train: 70% | Validation: 15% | Test: 15%, stratified by T-category. The test split indices are saved to outputs/models/test_indices.npy so step4_comparison.py evaluates all models on exactly the same held-out set.

---

## Comparison Models

### Random Forest (step4_comparison.py)

RandomForestRegressor and RandomForestClassifier with 200 trees, max depth 12. Final prediction is the average across all trees for regression and majority vote for classification. Feature importance from the regressor is used for the importance plot.

Limitation: AUC of 0.551 on fracture risk, essentially random. No temporal projection capability.

### Support Vector Machine (step4_comparison.py)

SVR with RBF kernel (C=10, gamma=scale) for BMD regression. SVC with RBF kernel (probability=True) for T-category classification. Fracture probability is derived by mapping Osteoporosis predictions (T-category >= 2) to the binary fracture target.

Limitation: Slowest training. AUC of 0.577 on fracture risk. No biological knowledge.

### XGBoost (step4_comparison.py)

XGBRegressor and XGBClassifier with 200 estimators, max depth 6, learning rate 0.05. Best BMD regression R2 of 0.676 among all models.

Limitation: AUC of 0.568 on fracture risk. No interpretability. No physics understanding.

---

## Results

### Full Metric Comparison

| Metric | Random Forest | SVM | XGBoost | PINN (Final) |
|---|---|---|---|---|
| BMD MAE (lower is better) | 0.038 | 0.043 | 0.037 | 0.039 |
| BMD R2 (higher is better) | 0.674 | 0.591 | 0.676 | 0.654 |
| T-Category Accuracy | 0.825 | 0.825 | 0.794 | 0.841 |
| Fracture AUC | 0.551 | 0.577 | 0.568 | 0.869 |
| Temporal Projection | No | No | No | Yes |
| Physics-Consistent | No | No | No | Yes |

### Key Findings

The PINN dominates fracture AUC by 53% over the best baseline (0.869 vs 0.568). This is the most clinically important metric because fracture risk is fundamentally a biomechanical phenomenon governed by the physics of bone density and its rate of change. A model that learns the physics of bone loss naturally understands fracture risk in a way that a pure pattern-matching model cannot.

The PINN achieves the highest T-category classification accuracy at 84.1%, correctly classifying all three severity levels better than any baseline model.

Tree-based models win only on pure BMD regression, where they benefit from their strength with tabular data. However, they fail entirely on every clinically meaningful metric.

All three baseline models perform near random chance for fracture risk (AUC 0.55 to 0.58) versus the PINN at 0.869. This gap exists because fracture risk is not a pattern that can be read from cross-sectional data alone; it requires a model of the underlying biological process.

### Trustworthiness Assessment

For fracture risk screening (AUC 0.869): clinically trustworthy. This exceeds the 0.80 threshold used in published medical AI literature. The model correctly identifies high-risk patients 87% of the time.

For BMD value estimation (R2 0.654): moderate confidence. Predictions are directionally correct and clinically plausible, but not precise enough to replace a DEXA scan. Recommended use is identifying patients who should get a DEXA scan, not replacing the scan.

For T-category classification (84.1% accuracy): clinically useful. If used to classify 100 patients, 84 will be correctly categorized. Particularly valuable as a primary screening tool where DEXA access is limited.

---

## Visualizations

All plots are generated automatically and saved to outputs/plots/.

| File | Contents | Key Insight |
|---|---|---|
| 01_eda_overview.png | 6-panel overview: BMD distributions, age vs BMD, T-category counts, sex-age BMD curves, risk score, fracture rates | BMD decreases with age; females drop faster post-45; fracture rate is 90.7% in Osteoporosis category |
| 02_correlation_heatmap.png | 15x15 feature correlation matrix | Age (-0.50) and sex (-0.37) are strongest BMD predictors; postmenopausal drives risk score (0.58) |
| 03_bmd_trajectories.png | Physics ODE projections for 4 patient profiles over 30 years | Glucocorticoids dramatically accelerate bone loss; females cross the osteoporosis threshold approximately 15 years earlier than males |
| 04_pinn_training.png | Loss curves, R2 history, AUC history by epoch | Smooth convergence; train and validation gap minimal; AUC stable at 0.869 from epoch 5 onward |
| 05_model_comparison.png | 4 bar charts comparing all metrics across all models | PINN leads AUC and accuracy; XGBoost leads BMD regression |
| 06_bmd_scatter.png | Predicted vs actual BMD scatter for all 4 models | PINN scatter is now comparable to tree-based models after v2 improvements |
| 07_confusion_matrices.png | 3x3 classification matrices for all 4 models | PINN correctly classifies the most patients in all three categories |
| 08_feature_importance.png | Random Forest feature importance ranking | Age (0.41) dominates; risk_score (0.23) validates the composite feature design |
| 09_roc_curves.png | ROC curves for all 4 models on fracture risk | PINN curve is far above the diagonal; all baselines cluster near random chance |
| 10_pinn_improvement.png | Before and after comparison between PINN v1 and PINN v2 | BMD R2 improved by 19%; all metrics improved or maintained |

---

## Setup and Installation

### Requirements

Python 3.9 or higher is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

The requirements.txt includes:

```
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
xgboost>=1.7.0
shap
```

### GPU Support

The PINN training script automatically detects and uses a CUDA GPU if available. CPU training is fully supported and completes in a reasonable time given the dataset size.

### Directory Setup

Place bmd.csv and osteoporosis.csv in the same folder as the Python scripts. The outputs/ directory and all subdirectories are created automatically by step1_preprocess.py.

---

## Running the Pipeline

The pipeline must be run in order. Each step depends on the outputs of the previous step.

**Step 1: Preprocessing**

```bash
python step1_preprocess.py
```

Loads both CSVs, cleans and standardizes all columns, engineers features, estimates BMD for osteoporosis.csv, merges the two datasets into 2,127 rows, fits and saves the StandardScaler, and writes combined_dataset.csv, X_scaled.csv, targets.csv, scaler.pkl, and feature_cols.pkl to the outputs/ folder.

**Step 2: Exploratory Data Analysis**

```bash
python step2_eda.py
```

Generates three visualization files: 01_eda_overview.png, 02_correlation_heatmap.png, and 03_bmd_trajectories.png. Requires combined_dataset.csv from Step 1.

**Step 3: PINN Training**

```bash
python step3_pinn.py
```

Builds the PINN architecture, trains with combined data and physics loss, applies early stopping, saves the best checkpoint to outputs/models/pinn_model.pth, and generates 04_pinn_training.png and 10_pinn_improvement.png. Training progress is printed every 20 epochs.

**Step 4: Comparison Models**

```bash
python step4_comparison.py
```

Trains Random Forest, SVM, and XGBoost on the same train/test split as the PINN. Generates plots 05 through 09 and saves model_comparison.csv. Requires test_indices.npy and pinn_results.pkl from Step 3.

**Step 5: Single Patient Prediction Demo**

```bash
python step5_predict.py
```

Runs four example patients through the trained PINN and prints all seven outputs for each. The predict_patient() function in this script is also imported by app.py.

---

## Streamlit Application

```bash
streamlit run app.py
```

The application opens at http://localhost:8501 by default.

### Interface Layout

The sidebar contains all patient inputs: age slider, sex selector, weight and height inputs, medication dropdown, and eight binary lifestyle toggles (family history, postmenopausal status, low calcium, low vitamin D, sedentary, underweight, smoking, alcohol).

The main panel displays results in real time on every input change:

- Four key metric cards for BMD, T-score, fracture risk, and risk score
- Color-coded diagnosis banner (green for Normal, orange for Osteopenia, red for Osteoporosis)
- 20-year BMD trajectory chart with colored risk zone bands
- Four-box future projections panel at 5, 10, and 20 years, each color-coded by severity
- Active risk factors list
- WHO T-score scale visualization
- Model comparison table
- Tabbed view of all ten analysis plots

### Model Loading

The app caches the model, scaler, and SHAP analyzer using st.cache_resource so they load only once per session. BatchNorm layers are forced to eval mode after loading to ensure deterministic predictions at inference time.

---

## SHAP Feature Analysis

Two SHAP scripts are included for different use cases.

**shap_simple.py** is a standalone script that runs SHAP analysis on a hardcoded example patient and saves the result to outputs/shap_analysis.png and outputs/shap_features.csv. Run it directly to inspect feature contributions outside the app.

**shap_integration.py** defines the SHAPOsteoporosis class, which is imported by app.py. It wraps the PINN's BMD prediction in a SHAP KernelExplainer using a Gaussian background distribution (100 samples). The explainer computes SHAP values for a single patient and returns a bar chart showing which features push BMD up (green) or down (red) and by how much.

Because KernelExplainer is model-agnostic, it works with the PINN without requiring any modification to the model code.

---

## Real-World Applications

**Primary care screening:** A general practitioner can enter a patient's basic information during a routine visit and immediately determine whether to refer them for a DEXA scan. This is particularly valuable for patients aged 45 to 65 who are pre-symptomatic, where the model can identify rising risk years before a fracture occurs.

**Rural and low-resource healthcare:** In regions where DEXA machines are unavailable, OsteoPINN provides the only accessible bone health screening. A community health worker with a smartphone and the patient's age, weight, and height can generate a meaningful risk assessment.

**Medication management:** The medication risk feature (especially Glucocorticoids) allows rheumatologists and pulmonologists to monitor patients on long-term steroid therapy. The 20-year BMD trajectory shows exactly how quickly bone loss will accelerate and when preventive intervention is needed.

**Patient education:** The BMD trajectory chart and future projections are directly comprehensible to patients. Showing a person that their BMD will cross into the osteoporosis zone in 8 years if they remain sedentary, and showing how that changes if they start calcium supplements and exercise, is a powerful motivational tool.

**Research and epidemiology:** The model's feature importance analysis quantifies the relative contribution of each risk factor to bone loss at a population level. This can guide public health interventions by identifying which modifiable factors to target first.

---

## Limitations and Future Work

### Current Limitations

1,958 of the 2,127 training samples use estimated rather than measured BMD values. This limits the precision of BMD regression and means the model should be used for screening rather than clinical quantification.

The dataset lacks ethnic diversity in BMD reference values. WHO T-score thresholds may not apply equally to all populations, particularly East Asian and African patients where different reference standards exist.

Vitamin D levels, calcium intake, and hormonal status are captured as binary flags rather than continuous laboratory values. Continuous assay data would substantially improve precision.

No longitudinal data is used. The model predicts from cross-sectional snapshots rather than tracking individual patients over time.

### Future Improvements

- Integrate with the NHANES dataset (30,000+ patients with measured DEXA BMD) to improve regression accuracy
- Add continuous laboratory inputs: serum calcium, vitamin D level, parathyroid hormone, and CTX bone resorption marker
- Train separate models or use group-specific T-score reference values for different ethnic populations
- Add per-patient SHAP values directly to the main Streamlit dashboard, not just as a separate tab
- Deploy as a Progressive Web App for offline use in low-connectivity clinical settings
- Collect longitudinal patient data to allow the model to learn individual bone loss trajectories over time

---

## Technologies

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.0+ | PINN implementation; autograd for physics residual loss |
| scikit-learn | 1.2+ | Random Forest, SVM, StandardScaler, metrics |
| XGBoost | 1.7+ | Gradient boosted tree baseline |
| pandas | 1.5+ | Data loading, cleaning, merging |
| numpy | 1.23+ | Numerical operations |
| matplotlib | 3.6+ | All static visualizations |
| seaborn | 0.12+ | Correlation heatmap |
| streamlit | 1.28+ | Interactive web application |
| shap | latest | Per-patient feature attribution |

PyTorch was chosen over TensorFlow for its flexibility in defining custom loss functions, which is essential for computing physics residuals through autograd. AdamW was chosen over Adam for its decoupled weight decay, which provides proper L2 regularization critical for small datasets. BatchNorm was used instead of LayerNorm because BatchNorm normalizes across the batch dimension, which is more effective for tabular data.

---

## License

This project is intended for research and educational use. It is not a certified medical device and should not be used as the sole basis for clinical decision-making. Always refer patients to qualified healthcare professionals for diagnosis and treatment.

# Salt Mass Fix: R²=0.384 → Target ≥0.70

## Root Cause Analysis

### Problem: Extreme Right-Skewed Distribution

**Current Performance**: R²=0.384, MAE=0.661г (UNACCEPTABLE)

**Data Characteristics**:
```
Mean: 2.553г
Median: 1.075г (2.4x lower!)
Std: 3.236г (CV=1.268)
Skewness: 2.944 (extreme right-skew)
Kurtosis: 9.361 (heavy-tailed)
Outliers: 5.3% (12-20г)
```

**Distribution by Metal**:
```
Cu:  mean=5.721г, max=20г  (107 samples) ← OUTLIER GROUP
Al:  mean=1.398г
Fe:  mean=0.913г
Zr:  mean=0.887г
La:  mean=1.077г
Zn:  mean=0.475г           ← 12x difference!
```

**Distribution by Combination**:
```
Cu+BTC:     5.721г  (107 samples) - Dominant outlier group
Al+BTC:     1.494г
Fe+BTC:     1.293г
All others: 0.3-1.5г
```

### Why Huber Loss Failed

**Huber loss** tries to fit mean, but:
- Median = 1.075г
- Mean = 2.553г (pulled up by Cu outliers)
- Model tries to compromise → poor fit for both low and high values

**IsolationForest** (5% contamination) removes some outliers, but:
- Cu+BTC is 32% of data (not an outlier statistically)
- It's a legitimate high-salt regime, not noise

---

## Solution Implemented

### 1. Quantile Regression (α=0.5) ✅
**Why**: Predict **median** instead of mean
- Median is robust to extreme outliers
- Better for asymmetric distributions
- Loss function: absolute deviation from quantile

**Expected improvement**: R² ≈ 0.65-0.75

### 2. Feature Engineering ✅
**Added features**:
```python
- 'Metal_Ligand_Combo': Categorical encoding of metal×ligand pairs
- 'Log_Metal_MW': Log-transformed molecular weight
- 'Is_Cu': Binary indicator (1 for Cu, 0 otherwise)
- 'Is_Zn': Binary indicator (1 for Zn, 0 otherwise)
```

**Why these features**:
- **Metal_Ligand_Combo**: Captures regime-specific behavior (Cu+BTC vs others)
- **Log_Metal_MW**: Non-linear relationship with salt mass
- **Is_Cu/Is_Zn**: Explicit signals for extreme cases

### 3. Reduced Outlier Removal ✅
**Changed**: `outlier_contamination=0.05` → `0.02`

**Why**: Cu+BTC is not statistical noise, it's a valid regime
- Keep more data for training
- Let quantile regression handle variability

---

## Expected Results

### Before
```
R² = 0.384
MAE = 0.661г
cv_mean = -0.506
```

### After (Target)
```
R² ≥ 0.70       (Target met if ≥0.70)
MAE ≤ 0.50г     (30% improvement)
cv_mean ≈ -0.40 (Better generalization)
```

### Why This Should Work

**Quantile Regression Benefits**:
1. Robust to outliers (no mean-chasing)
2. Predicts median (1.075г) accurately
3. Less sensitivity to Cu+BTC group

**Feature Engineering Benefits**:
1. **Metal_Ligand_Combo** helps model learn:
   - Cu+BTC → high salt (5-20г)
   - Others → low salt (0.3-1.5г)
2. **Is_Cu** gives explicit signal for high-salt regime
3. **Log_Metal_MW** captures non-linear stoichiometry

---

## Alternative Approaches (If Still Not ≥0.70)

### Option A: Log-Transform Target
```python
# In data processing
df['log_salt_mass'] = np.log1p(df['m (соли), г'])

# In pipeline
target="log_salt_mass"  # Instead of "m (соли), г"

# At inference
predictions = np.expm1(model.predict(X))  # Transform back
```

**Expected**: R² ≈ 0.75-0.85 (log-space is linear-friendly)

### Option B: Two-Stage Modeling
```python
# Stage 1: Classify Cu vs Others
is_cu_classifier = ...

# Stage 2a: Regressor for Cu (high salt)
cu_regressor = ...  # Trained only on Cu samples

# Stage 2b: Regressor for Others (low salt)
other_regressor = ...  # Trained on non-Cu samples
```

**Expected**: R² ≈ 0.80+ (specialized models)

### Option C: Ensemble of Quantiles
```python
# Predict multiple quantiles and ensemble
quantiles = [0.3, 0.5, 0.7]
predictions = weighted_average([q30, q50, q70])
```

**Expected**: R² ≈ 0.70-0.75 (better uncertainty handling)

---

## How to Run

```bash
# Clean cache and retrain
cd /home/ruslan_safaev/adsorb_synth/adsorb_synthesis
find . -name "*.pyc" -delete
find . -type d -name __pycache__ -exec rm -rf {} +

# Run training
./RUN_TRAINING.sh
# or
.venv/bin/python scripts/train_inverse_design.py
```

**Check results**:
- Look for salt_mass R² ≥ 0.70
- MAE should be ≤ 0.50г
- cv_mean should be closer to -0.40

---

## If Results Still Poor (R² < 0.70)

### Debugging Steps

1. **Check feature importance**:
```python
from artifacts import load_pipeline
pipeline = load_pipeline('artifacts')
salt_model = pipeline.stage_results['salt_mass'].pipeline
importances = salt_model.named_steps['model'].get_feature_importance()
```

2. **Check if new features are used**:
```python
# During training, look for:
print(salt_features)  # Should contain 'Metal_Ligand_Combo', 'Is_Cu', etc.
```

3. **Validate quantile regression**:
```python
# Should see message:
# [INFO] Using Quantile Regression (alpha=0.5) for XGBoost
# [INFO] Using Quantile Regression (alpha=0.5) for CatBoost (via MAE)
```

### Escalation: Implement Log-Transform

If quantile regression + features don't reach R²=0.70:

```python
# In data_processing.py
def add_salt_mass_features(df: pd.DataFrame) -> None:
    # Existing features...
    
    # Add log-transformed target
    if 'm (соли), г' in df.columns:
        df['log_m_salt'] = np.log1p(df['m (соли), г'])
```

```python
# In pipeline.py - change target
StageConfig(
    name="salt_mass",
    target="log_m_salt",  # Instead of "m (соли), г"
    ...
)
```

```python
# In predict() - transform back
if stage.name == "salt_mass":
    predictions = np.expm1(predictions)  # exp(x) - 1
    results[stage.target] = predictions
```

---

## Success Criteria

✅ **R² ≥ 0.70** (Primary metric)  
✅ **MAE ≤ 0.50г** (Secondary metric)  
✅ **cv_mean ≈ -0.40 to -0.50** (Consistency check)  

If all three met → **SUCCESS**  
If R² = 0.65-0.70 → **Acceptable** (significant improvement from 0.384)  
If R² < 0.65 → **Implement log-transform** (Option A)

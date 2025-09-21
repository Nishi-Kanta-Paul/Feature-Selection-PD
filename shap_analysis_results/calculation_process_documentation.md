# FEATURE IMPORTANCE CALCULATION PROCESS DOCUMENTATION

**Generated on:** 2025-09-21 20:10:12

## 📊 Step-by-Step Calculation Process

### Step 1: Raw Feature Importance Extraction
```python
# From each trained model
for model_name, model in models.items():
    importance_scores = model.feature_importances_  # Built-in importance
    # Create DataFrame with feature names and scores
```

### Step 2: Raw Data Summary
- **Total Records:** 60
- **Features Analyzed:** 30
- **Models Used:** ['RandomForest', 'XGBoost']

### Step 3: Sample Raw Importance Scores
| Feature | Model | Raw Importance |
|---------|-------|----------------|
| shimmer_apq | RandomForest | 0.073529 |
| shimmer_apq3 | RandomForest | 0.073529 |
| dfa | RandomForest | 0.073529 |
| jitter_percent | XGBoost | 0.000000 |
| jitter_abs | XGBoost | 0.000000 |
| jitter_rap | XGBoost | 0.000000 |

### Step 4: Aggregation Formula
```python
# Group by feature and calculate statistics
agg_importance = combined_df.groupby('feature')['importance'].agg([
    'mean',    # Average importance across models
    'std',     # Standard deviation
    'count'    # Number of models
]).reset_index()

# Sort by mean importance (descending)
agg_importance = agg_importance.sort_values('mean', ascending=False)
```

### Step 5: Mathematical Calculation
For each feature `i`:

**Mean Importance:**
```
Mean_i = (RF_importance_i + XGBoost_importance_i) / 2
```

**Standard Deviation:**
```
Std_i = sqrt(((RF_importance_i - Mean_i)² + (XGBoost_importance_i - Mean_i)²) / 1)
```

### Step 6: Calculation Examples (Top 5 Features)
| Rank | Feature | RF Score | XGBoost Score | Mean | Std | Formula |
|------|---------|----------|---------------|------|-----|----------|
| 1 | d2 | 0.073529 | 0.000000 | 0.036765 | 0.051993 | (0.0735 + 0.0000) / 2 |
| 2 | dfa | 0.073529 | 0.000000 | 0.036765 | 0.051993 | (0.0735 + 0.0000) / 2 |
| 3 | f0_cv | 0.073529 | 0.000000 | 0.036765 | 0.051993 | (0.0735 + 0.0000) / 2 |
| 4 | shimmer_apq | 0.073529 | 0.000000 | 0.036765 | 0.051993 | (0.0735 + 0.0000) / 2 |
| 5 | shimmer_apq3 | 0.073529 | 0.000000 | 0.036765 | 0.051993 | (0.0735 + 0.0000) / 2 |

### Step 7: Ranking Assignment
```python
# Assign ranks based on sorted mean importance
agg_importance['rank'] = range(1, len(agg_importance) + 1)
```

### Step 8: Top Features Selection
**Selection Criteria:**
- Features sorted by **Mean Importance** (highest first)
- Top 15 selected using: `agg_importance.head(15)`
- Selection is **automatic** based on ranking

**Why this approach?**
1. **Consensus-based:** Uses multiple models for robust selection
2. **Statistically sound:** Considers both mean and variance
3. **Reproducible:** Same process gives same results
4. **Interpretable:** Clear mathematical foundation

### Step 9: Final Feature Rankings
| Rank | Feature | Mean Importance | Std | Selection |
|------|---------|-----------------|-----|----------|
| 1 | d2 | 0.036765 | 0.051993 | ✅ Top 15 |
| 2 | dfa | 0.036765 | 0.051993 | ✅ Top 15 |
| 3 | f0_cv | 0.036765 | 0.051993 | ✅ Top 15 |
| 4 | shimmer_apq | 0.036765 | 0.051993 | ✅ Top 15 |
| 5 | shimmer_apq3 | 0.036765 | 0.051993 | ✅ Top 15 |
| 6 | f0_min | 0.029412 | 0.041595 | ✅ Top 15 |
| 7 | jitter_rap | 0.029412 | 0.041595 | ✅ Top 15 |
| 8 | spread2 | 0.029412 | 0.041595 | ✅ Top 15 |
| 9 | f0_std | 0.029412 | 0.041595 | ✅ Top 15 |
| 10 | jitter_abs | 0.026261 | 0.037138 | ✅ Top 15 |
| 11 | shimmer_apq5 | 0.022059 | 0.031196 | ✅ Top 15 |
| 12 | f0_range | 0.022059 | 0.031196 | ✅ Top 15 |
| 13 | jitter_ppq | 0.014706 | 0.020797 | ✅ Top 15 |
| 14 | jitter_percent | 0.014706 | 0.020797 | ✅ Top 15 |
| 15 | zcr_std | 0.014706 | 0.020797 | ✅ Top 15 |
| 16 | zcr_mean | 0.014706 | 0.020797 | ❌ Not selected |
| 17 | shimmer_db | 0.010504 | 0.014855 | ❌ Not selected |
| 18 | rpde | 0.010504 | 0.014855 | ❌ Not selected |
| 19 | voiced_ratio | 0.007353 | 0.010399 | ❌ Not selected |
| 20 | spread1 | 0.007353 | 0.010399 | ❌ Not selected |
| 21 | f0_max | 0.007353 | 0.010399 | ❌ Not selected |
| 22 | hnr | 0.007353 | 0.010399 | ❌ Not selected |
| 23 | shimmer_dda | 0.007353 | 0.010399 | ❌ Not selected |
| 24 | nhr | 0.004202 | 0.005942 | ❌ Not selected |
| 25 | jitter_ddp | 0.004085 | 0.005777 | ❌ Not selected |
| 26 | f0_mean | 0.003268 | 0.004622 | ❌ Not selected |
| 27 | ppe | 0.000000 | 0.000000 | ❌ Not selected |
| 28 | shimmer_percent | 0.000000 | 0.000000 | ❌ Not selected |
| 29 | ste_mean | 0.000000 | 0.000000 | ❌ Not selected |
| 30 | ste_std | 0.000000 | 0.000000 | ❌ Not selected |

---
**Total Features Analyzed:** 30
**Top 15 Selected Features:** 15
**Selection Threshold:** Mean importance ≥ 0.014706

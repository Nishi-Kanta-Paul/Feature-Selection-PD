# PARKINSON'S DISEASE FEATURE SELECTION REPORT
## Feature Importance Analysis Results

**Generated on:** 2025-09-21 19:40:08

## Dataset Summary

- **Total Samples:** 12
- **Total Features:** 30
- **Class Distribution:** {'HC': np.int64(10), 'PD': np.int64(2)}

## Model Performance Summary

| Model | CV Accuracy | Std | Test AUC |
|-------|-------------|-----|----------|
| RandomForest | 0.889 | 0.157 | 1.000 |
| XGBoost | nan | nan | 0.500 |

## Top 15 Most Important Features

| Rank | Feature | Mean Importance | Std |
|------|---------|-----------------|-----|
| 1 | d2 | 0.0368 | 0.0520 |
| 2 | dfa | 0.0368 | 0.0520 |
| 3 | f0_cv | 0.0368 | 0.0520 |
| 4 | shimmer_apq | 0.0368 | 0.0520 |
| 5 | shimmer_apq3 | 0.0368 | 0.0520 |
| 6 | f0_min | 0.0294 | 0.0416 |
| 7 | jitter_rap | 0.0294 | 0.0416 |
| 8 | spread2 | 0.0294 | 0.0416 |
| 9 | f0_std | 0.0294 | 0.0416 |
| 10 | jitter_abs | 0.0263 | 0.0371 |
| 11 | shimmer_apq5 | 0.0221 | 0.0312 |
| 12 | f0_range | 0.0221 | 0.0312 |
| 13 | jitter_ppq | 0.0147 | 0.0208 |
| 14 | jitter_percent | 0.0147 | 0.0208 |
| 15 | zcr_std | 0.0147 | 0.0208 |

## Feature Categories Analysis

| Category | Avg Importance | Max Importance | Count |
|----------|----------------|----------------|-------|
| Jitter | 0.0178 | 0.0294 | 5 |
| Shimmer | 0.0189 | 0.0368 | 6 |
| Noise | 0.0058 | 0.0074 | 2 |
| Prosodic | 0.0214 | 0.0368 | 6 |
| Nonlinear | 0.0201 | 0.0368 | 6 |
| Additional | 0.0147 | 0.0147 | 2 |

## Recommendations

### Feature Selection Strategy:

1. **Top 5 Features** for quick screening:
   - d2 (importance: 0.0368)
   - dfa (importance: 0.0368)
   - f0_cv (importance: 0.0368)
   - shimmer_apq (importance: 0.0368)
   - shimmer_apq3 (importance: 0.0368)

2. **Top 10 Features** for balanced performance:
   - d2 (importance: 0.0368)
   - dfa (importance: 0.0368)
   - f0_cv (importance: 0.0368)
   - shimmer_apq (importance: 0.0368)
   - shimmer_apq3 (importance: 0.0368)
   - f0_min (importance: 0.0294)
   - jitter_rap (importance: 0.0294)
   - spread2 (importance: 0.0294)
   - f0_std (importance: 0.0294)
   - jitter_abs (importance: 0.0263)

3. **Top 15 Features** for comprehensive analysis:
   - d2 (importance: 0.0368)
   - dfa (importance: 0.0368)
   - f0_cv (importance: 0.0368)
   - shimmer_apq (importance: 0.0368)
   - shimmer_apq3 (importance: 0.0368)
   - f0_min (importance: 0.0294)
   - jitter_rap (importance: 0.0294)
   - spread2 (importance: 0.0294)
   - f0_std (importance: 0.0294)
   - jitter_abs (importance: 0.0263)
   - shimmer_apq5 (importance: 0.0221)
   - f0_range (importance: 0.0221)
   - jitter_ppq (importance: 0.0147)
   - jitter_percent (importance: 0.0147)
   - zcr_std (importance: 0.0147)

**Analysis completed successfully with feature importance analysis!**

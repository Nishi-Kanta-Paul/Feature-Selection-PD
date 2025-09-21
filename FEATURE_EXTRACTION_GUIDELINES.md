# 🔬 FEATURE EXTRACTION GUIDELINES

## Complete Guide for Parkinson's Disease Voice Feature Analysis

---

## 📋 **OVERVIEW**

This guide provides comprehensive instructions for extracting voice features from preprocessed audio data for Parkinson's Disease (PD) research. The feature extraction system implements 35 research-validated features across 6 categories, following MDVP standards and international literature.

---

## 🎯 **FEATURE EXTRACTION OBJECTIVES**

1. **Comprehensive Analysis**: Extract all clinically relevant voice features
2. **Research Compliance**: Follow MDVP and literature standards
3. **PD Discrimination**: Focus on features that differentiate PD from healthy controls
4. **Statistical Robustness**: Ensure reliable, reproducible measurements
5. **Clinical Translation**: Generate features suitable for diagnostic applications

---

## 🔬 **FEATURE CATEGORIES & SPECIFICATIONS**

### **1. Jitter-Based Features (5 features)**

_Measures fundamental frequency (F0) period irregularities_

| Feature            | Definition                                 | PD Relevance               | Normal Range  |
| ------------------ | ------------------------------------------ | -------------------------- | ------------- |
| **jitter_percent** | Cycle-to-cycle F0 period variation (%)     | ↑ Increased in PD          | <1% (healthy) |
| **jitter_abs**     | Absolute jitter in microseconds            | ↑ Higher variability       | <40 μs        |
| **jitter_rap**     | Relative Average Perturbation              | ↑ Local period instability | <0.5%         |
| **jitter_ppq**     | 5-point Period Perturbation Quotient       | ↑ Multi-period variation   | <0.5%         |
| **jitter_ddp**     | Average absolute difference of differences | ↑ Rapid fluctuations       | <1%           |

**Clinical Significance**: Jitter reflects vocal fold vibration irregularity, typically elevated in PD due to motor control impairments.

### **2. Shimmer-Based Features (6 features)**

_Measures amplitude variation between vocal cycles_

| Feature             | Definition                                 | PD Relevance                    | Normal Range |
| ------------------- | ------------------------------------------ | ------------------------------- | ------------ |
| **shimmer_percent** | Cycle-to-cycle amplitude variation (%)     | ↑ Increased instability         | <4%          |
| **shimmer_db**      | Shimmer in decibel scale                   | ↑ Amplitude irregularity        | <0.35 dB     |
| **shimmer_apq3**    | 3-point Amplitude Perturbation Quotient    | ↑ Short-term variation          | <4%          |
| **shimmer_apq5**    | 5-point Amplitude Perturbation Quotient    | ↑ Medium-term variation         | <4%          |
| **shimmer_apq**     | General Amplitude Perturbation Quotient    | ↑ Overall amplitude instability | <4%          |
| **shimmer_dda**     | Average absolute difference of differences | ↑ Rapid amplitude changes       | <4%          |

**Clinical Significance**: Shimmer indicates reduced vocal amplitude control, commonly seen in PD hypophonia.

### **3. Voice Quality/Noise Features (2 features)**

_Assesses harmonic-to-noise characteristics_

| Feature | Definition                    | PD Relevance            | Normal Range |
| ------- | ----------------------------- | ----------------------- | ------------ |
| **hnr** | Harmonics-to-Noise Ratio (dB) | ↓ Decreased in PD       | >20 dB       |
| **nhr** | Noise-to-Harmonics Ratio      | ↑ Increased breathiness | <0.15        |

**Clinical Significance**: Reflects vocal fold closure efficiency and voice quality; PD often shows increased noise due to incomplete closure.

### **4. Prosodic/Frequency Features (6 features)**

_Characterizes fundamental frequency patterns_

| Feature      | Definition                      | PD Relevance        | Normal Range     |
| ------------ | ------------------------------- | ------------------- | ---------------- |
| **f0_mean**  | Mean fundamental frequency (Hz) | Variable (often ↓)  | Gender-dependent |
| **f0_std**   | F0 standard deviation           | ↓ Reduced range     | Variable         |
| **f0_min**   | Minimum F0 (Hz)                 | Clinical assessment | Gender-dependent |
| **f0_max**   | Maximum F0 (Hz)                 | Clinical assessment | Gender-dependent |
| **f0_range** | F0 range (max - min)            | ↓ Reduced in PD     | Variable         |
| **f0_cv**    | F0 coefficient of variation (%) | ↓ Monotonic speech  | <25%             |

**Clinical Significance**: Prosodic features reflect speech melody and intonation, often reduced in PD (hypoprosody).

### **5. Nonlinear Dynamical Features (6 features)**

_Captures complexity and predictability of vocal dynamics_

| Feature     | Definition                        | PD Relevance                 | Normal Range     |
| ----------- | --------------------------------- | ---------------------------- | ---------------- |
| **rpde**    | Recurrence Period Density Entropy | ↓ Reduced complexity         | Population-based |
| **d2**      | Correlation dimension             | ↓ Lower dimensionality       | Relative measure |
| **dfa**     | Detrended Fluctuation Analysis    | ↓ Altered scaling            | 0.5-1.5          |
| **spread1** | F0 spread measure 1               | ↑ Increased variability      | Relative measure |
| **spread2** | F0 spread measure 2               | ↑ Increased variability      | Relative measure |
| **ppe**     | Pitch Period Entropy              | ↑ Increased unpredictability | <2.0             |

**Clinical Significance**: Nonlinear measures capture subtle voice dynamics changes not detected by traditional linear measures.

### **6. Additional Signal Features (5 features)**

_Supplementary acoustic measurements_

| Feature          | Definition              | PD Relevance               | Normal Range     |
| ---------------- | ----------------------- | -------------------------- | ---------------- |
| **ste_mean**     | Short-Term Energy mean  | ↓ Reduced voice strength   | Relative measure |
| **ste_std**      | STE standard deviation  | ↓ Reduced energy variation | Relative measure |
| **zcr_mean**     | Zero-Crossing Rate mean | Variable                   | 0.1-0.3          |
| **zcr_std**      | ZCR standard deviation  | Spectral information       | Variable         |
| **voiced_ratio** | Voice activity ratio    | ↓ Reduced voicing          | >0.5             |

---

## 🚀 **FEATURE EXTRACTION PIPELINE**

### **Step 1: Audio Loading & Validation**

```python
# Load 16kHz preprocessed audio
audio_data = loader.load_wav_file(filepath)
signal = audio_data['signal']
sample_rate = audio_data['sample_rate']  # Should be 16000 Hz
```

**Process:**

1. Load WAV file using wave module
2. Validate sample rate (16 kHz expected)
3. Convert to normalized float array
4. Check signal duration and quality

### **Step 2: Voice Activity Detection**

```python
# Detect voiced segments using energy and ZCR
voiced_frames, periods, f0_values = get_voiced_segments(signal, sr)
```

**Algorithm:**

1. Frame signal (25ms windows, 10ms shift)
2. Calculate Short-Term Energy (STE) per frame
3. Calculate Zero-Crossing Rate (ZCR) per frame
4. Apply thresholds: STE > 0.01 AND ZCR < 0.3
5. Extract voiced frames for analysis

### **Step 3: Pitch Detection**

```python
# F0 estimation using autocorrelation
f0, period = estimate_pitch(frame, sr, min_f0=70, max_f0=400)
```

**Method:**

1. Compute autocorrelation of each voiced frame
2. Search for peak in valid F0 range (70-400 Hz)
3. Convert lag to fundamental frequency
4. Validate pitch continuity and reliability

### **Step 4: Feature Calculation**

#### **Jitter Analysis:**

```python
# Calculate all jitter measures
jitter_features = calculate_jitter_features(periods)
```

**Calculations:**

- **Jitter (%)**: `mean(|diff(periods)|) / mean(periods) * 100`
- **RAP**: `mean(|period[i] - mean(period[i-1:i+2])|) / mean(periods)`
- **PPQ**: Similar to RAP but over 5 periods
- **DDP**: `mean(|diff(diff(periods))|) / mean(periods)`

#### **Shimmer Analysis:**

```python
# Calculate amplitude variations
shimmer_features = calculate_shimmer_features(voiced_frames, sr)
```

**Calculations:**

- **Shimmer (%)**: `mean(|diff(amplitudes)|) / mean(amplitudes) * 100`
- **APQ3/APQ5**: Similar to jitter RAP/PPQ but for amplitudes
- **DDA**: `mean(|diff(diff(amplitudes))|) / mean(amplitudes)`

#### **Noise Analysis:**

```python
# Harmonic-to-noise analysis
noise_features = calculate_noise_features(voiced_frames, f0_values, sr)
```

**Method:**

1. Compute FFT of each voiced frame
2. Identify harmonic peaks (F0, 2*F0, 3*F0, ...)
3. Calculate harmonic energy vs noise energy
4. Compute HNR = 10\*log10(harmonic_power/noise_power)

### **Step 5: Statistical Aggregation**

```python
# Aggregate frame-based measurements
final_features = aggregate_features(frame_features)
```

**Aggregation Methods:**

- **Mean**: Primary measure for most features
- **Standard Deviation**: Variability assessment
- **Min/Max**: Range assessment for F0 features
- **Coefficient of Variation**: Normalized variability

---

## 🛠 **IMPLEMENTATION GUIDE**

### **Using comprehensive_pd_features.py:**

```bash
# Extract all features from preprocessed data
python comprehensive_pd_features.py

# Output: comprehensive_features/pd_features_comprehensive.csv
```

### **Custom Processing:**

```python
from comprehensive_pd_features import VoiceAnalyzer

# Initialize analyzer
analyzer = VoiceAnalyzer()

# Extract features from single file
features = analyzer.extract_all_features('path/to/audio.wav')

# Process entire dataset
for audio_file in audio_files:
    features = analyzer.extract_all_features(audio_file)
    # Save or analyze features
```

### **Configuration Options:**

```python
EXTRACTION_PARAMS = {
    'frame_length': 400,        # 25ms at 16kHz
    'hop_length': 160,          # 10ms at 16kHz
    'min_f0': 70,              # Minimum pitch (Hz)
    'max_f0': 400,             # Maximum pitch (Hz)
    'energy_threshold': 0.01,   # Voice activity threshold
    'zcr_threshold': 0.3,       # ZCR threshold for voicing
    'harmonic_window': 3        # Harmonic peak detection window
}
```

---

## 📊 **EXPECTED RESULTS**

### **Typical PD vs HC Differences:**

| Feature Category | HC Mean      | PD Mean      | Direction | Effect Size |
| ---------------- | ------------ | ------------ | --------- | ----------- |
| **Jitter (%)**   | 2.0 ± 2.3    | 5.1 ± 1.7    | ↑ Higher  | Large       |
| **Shimmer (%)**  | 4.1 ± 1.6    | 6.1 ± 1.1    | ↑ Higher  | Medium      |
| **HNR (dB)**     | -4.6 ± 9.8   | -2.9 ± 2.3   | Variable  | Small       |
| **F0 Mean (Hz)** | 153.9 ± 52.4 | 150.8 ± 20.7 | ↓ Lower   | Small       |
| **PPE**          | 0.86 ± 0.68  | 1.16 ± 0.22  | ↑ Higher  | Medium      |

### **Feature Quality Indicators:**

- **High Discriminative Power**: Jitter, Shimmer, PPE
- **Moderate Discriminative Power**: HNR, F0 variability, DFA
- **Supportive Features**: Energy measures, voice activity ratio

---

## 🔍 **QUALITY CONTROL**

### **Feature Validation Checks:**

1. **Range Validation:**

```python
# Check if features are within expected ranges
assert 0 <= jitter_percent <= 20, "Jitter out of range"
assert 0 <= shimmer_percent <= 20, "Shimmer out of range"
assert f0_mean >= 50 and f0_mean <= 500, "F0 out of range"
```

2. **Statistical Consistency:**

```python
# Verify statistical relationships
assert f0_std >= 0, "F0 std must be non-negative"
assert f0_max >= f0_min, "F0 max must be >= F0 min"
assert voiced_ratio >= 0 and voiced_ratio <= 1, "Voice ratio must be [0,1]"
```

3. **Clinical Plausibility:**

```python
# Check for extreme values that may indicate processing errors
if jitter_percent > 10:
    print("Warning: Extremely high jitter - check audio quality")
if hnr < -30:
    print("Warning: Very low HNR - possible noise contamination")
```

---

## 🎯 **FEATURE INTERPRETATION**

### **Clinical Significance by Category:**

#### **Jitter Features:**

- **High Jitter**: Indicates irregular vocal fold vibration
- **Clinical Threshold**: >1% abnormal (gender-adjusted)
- **PD Pattern**: Consistently elevated across all jitter measures

#### **Shimmer Features:**

- **High Shimmer**: Suggests amplitude control problems
- **Clinical Threshold**: >3.8% abnormal (gender-adjusted)
- **PD Pattern**: Elevated shimmer with reduced voice strength

#### **Noise Features:**

- **Low HNR**: Increased breathiness, poor vocal fold closure
- **High NHR**: Voice quality deterioration
- **PD Pattern**: Variable but often increased noise

#### **Prosodic Features:**

- **Reduced F0 Range**: Monotonic speech (hypoprosody)
- **Low F0 Variability**: Reduced speech melody
- **PD Pattern**: Flattened intonation patterns

#### **Nonlinear Features:**

- **Altered Complexity**: Changes in vocal system dynamics
- **Increased Entropy**: Reduced predictability
- **PD Pattern**: Decreased complexity in some measures, increased entropy in others

---

## 📈 **STATISTICAL ANALYSIS**

### **Recommended Analysis Approaches:**

1. **Descriptive Statistics:**

```python
# Basic descriptive analysis
hc_features = df[df['group'] == 'HC']
pd_features = df[df['group'] == 'PD']

for feature in feature_list:
    hc_mean = hc_features[feature].mean()
    pd_mean = pd_features[feature].mean()
    print(f"{feature}: HC={hc_mean:.3f}, PD={pd_mean:.3f}")
```

2. **Statistical Testing:**

```python
from scipy import stats

# Compare groups using appropriate tests
for feature in feature_list:
    hc_vals = hc_features[feature]
    pd_vals = pd_features[feature]

    # Check normality
    _, p_norm = stats.shapiro(hc_vals)

    if p_norm > 0.05:  # Normal distribution
        t_stat, p_val = stats.ttest_ind(hc_vals, pd_vals)
    else:  # Non-normal distribution
        u_stat, p_val = stats.mannwhitneyu(hc_vals, pd_vals)

    print(f"{feature}: p-value = {p_val:.4f}")
```

3. **Effect Size Calculation:**

```python
# Cohen's d effect size
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.var() + (n2-1)*group2.var()) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

for feature in feature_list:
    d = cohens_d(hc_features[feature], pd_features[feature])
    print(f"{feature}: Cohen's d = {d:.3f}")
```

---

## 🤖 **MACHINE LEARNING APPLICATIONS**

### **Feature Selection:**

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Univariate feature selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Random Forest feature importance
rf = RandomForestClassifier()
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### **Classification Pipeline:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Prepare data
X = df[feature_columns]
y = df['group']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = SVC(kernel='rbf')
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues & Solutions:**

1. **No Voiced Frames Detected:**

   - **Cause**: Audio too quiet or noisy
   - **Solution**: Adjust energy threshold, check preprocessing

2. **Extreme Feature Values:**

   - **Cause**: Poor audio quality or processing errors
   - **Solution**: Validate input audio, check parameter settings

3. **NaN/Infinite Values:**

   - **Cause**: Division by zero or empty arrays
   - **Solution**: Add validation checks, handle edge cases

4. **Inconsistent Results:**
   - **Cause**: Different audio preprocessing
   - **Solution**: Standardize preprocessing pipeline

### **Debug Commands:**

```python
# Check feature ranges
print(df.describe())

# Identify outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Validate feature correlations
correlation_matrix = df.corr()
high_corr = correlation_matrix[correlation_matrix.abs() > 0.95]
```

---

## 📁 **OUTPUT STRUCTURE**

### **Feature CSV Format:**

```csv
group,jitter_percent,jitter_abs,jitter_rap,jitter_ppq,jitter_ddp,shimmer_percent,shimmer_db,shimmer_apq3,shimmer_apq5,shimmer_apq,shimmer_dda,hnr,nhr,f0_mean,f0_std,f0_min,f0_max,f0_range,f0_cv,rpde,d2,dfa,spread1,spread2,ppe,ste_mean,ste_std,zcr_mean,zcr_std,voiced_ratio,filename,duration,sample_rate,num_voiced_frames,num_f0_values
HC,1.832,186.79,1.538,2.015,2.382,5.726,0.484,2.266,3.730,3.730,5.090,-13.624,0.937,100.57,24.98,94.12,333.33,239.22,24.84,9.67e-06,0.248,0.474,24.98,623.96,0.292,0.030,0.035,0.171,0.049,0.925,filtered_0001.wav,10.01,16000,924,887
```

### **Metadata Fields:**

- **group**: HC (Healthy Control) or PD (Parkinson's Disease)
- **filename**: Original audio filename
- **duration**: Audio duration in seconds
- **sample_rate**: Sampling rate (should be 16000)
- **num_voiced_frames**: Number of voiced segments detected
- **num_f0_values**: Number of valid F0 measurements

---

## ✅ **VALIDATION CHECKLIST**

Before using extracted features:

- [ ] All 35 features successfully extracted
- [ ] No NaN or infinite values present
- [ ] Feature values within expected ranges
- [ ] Sufficient voiced content detected (>50% voice activity ratio)
- [ ] Consistent results across similar audio files
- [ ] Metadata fields correctly populated
- [ ] HC vs PD differences align with literature
- [ ] Output CSV format properly structured

---

## 🎯 **NEXT STEPS**

After successful feature extraction:

1. **Statistical Analysis**: Compare HC vs PD feature distributions
2. **Feature Selection**: Identify most discriminative features
3. **Machine Learning**: Build classification models
4. **Clinical Validation**: Apply diagnostic thresholds
5. **Longitudinal Analysis**: Track feature changes over time
6. **Model Deployment**: Implement for clinical use

---

**⚠️ IMPORTANT**: Feature extraction results should be validated against known clinical patterns. Unexpected values may indicate processing errors or unique voice characteristics requiring investigation.

---

_This feature extraction system provides comprehensive voice analysis capabilities for Parkinson's Disease research and clinical applications._

# ⚠️ CORRECTED WORKFLOW - Percentile Filtering

## 🎯 **Critical Correction: When to Apply Percentile Filtering**

---

## ❌ **PREVIOUS (INCORRECT) APPROACH**

### Workflow:

```
Step 1: Raw Audio Files (data/HC/, data/PD/)
         ↓
Step 2: Percentile-based Band-pass Filtering
         • Calculate spectral centroids/rolloffs
         • Apply 1-99 percentile or 2.5-97.5 percentile filtering
         • Filter audio signal in frequency domain
         ↓
Step 3: Preprocessed Audio (preprocessed_data_percentile_1_99/)
         ↓
Step 4: Feature Extraction from FILTERED audio
         ↓
Step 5: Model Training
```

### Problems:

1. ❌ **Loss of Information**: Filtering removes frequency components that may contain important PD markers
2. ❌ **Irreversible**: Once audio is filtered, original information cannot be recovered
3. ❌ **Not Research Standard**: Most PD voice research extracts features from unfiltered audio
4. ❌ **Reduces Feature Discriminability**: Some features (like HNR, spectral measures) need full frequency content

---

## ✅ **CORRECTED (PROPER) APPROACH**

### Workflow:

```
Step 1: Raw Audio Files (data/HC/, data/PD/)
         ↓
Step 2: BASIC Preprocessing (Format Standardization ONLY)
         • Convert to 16kHz sampling rate
         • Convert to mono channel
         • Save as WAV format
         • NO FILTERING APPLIED
         ↓
Step 3: Basic Preprocessed Audio (preprocessed_data_basic/)
         ↓
Step 4: Feature Extraction from UNFILTERED audio
         • Extract all 35+ features
         • Jitter, Shimmer, HNR, F0, Nonlinear, etc.
         • Save to CSV
         ↓
Step 5: Feature Analysis & Selection (OPTIONAL Percentile Filtering)
         • Analyze feature distributions
         • Apply percentile-based filtering on FEATURES (not audio)
         • Remove outlier features if needed
         • Feature selection (SHAP, mutual information)
         ↓
Step 6: Model Training with Selected Features
```

### Advantages:

1. ✅ **Preserves All Information**: Full frequency content available for feature extraction
2. ✅ **Reversible**: Can experiment with different feature sets without re-extracting
3. ✅ **Research Standard**: Follows established PD voice analysis protocols
4. ✅ **Better Features**: Features extracted from full-bandwidth audio are more discriminative
5. ✅ **Flexible**: Can apply different feature selection methods post-extraction

---

## 📊 **Detailed Comparison**

### **Audio-Level Filtering (WRONG)**

```python
# WRONG: Filtering audio before feature extraction
audio = load_audio("voice.wav")
filtered_audio = apply_percentile_filter(audio)  # ❌ Removes information
features = extract_features(filtered_audio)       # ❌ Features from limited data
```

**Problems:**

- Spectral centroid/rolloff percentiles determine filter cutoffs
- Removes frequency components outside percentile range
- Jitter/shimmer calculated from incomplete signal
- HNR calculation has reduced harmonic content
- Nonlinear features lose complexity information

### **Feature-Level Filtering (CORRECT)**

```python
# CORRECT: Extract features first, then filter/select
audio = load_audio("voice.wav")
features = extract_features(audio)                # ✅ All features from full audio

# Optional: Filter outlier features
features_filtered = remove_outlier_features(features, percentile_range=[1, 99])

# Or: Feature selection
selected_features = feature_selection(features, method='SHAP')
```

**Benefits:**

- All features extracted from complete signal
- Can compare different feature subsets
- Outlier removal on feature space (not signal)
- Maintains feature interpretability
- Aligns with research methodology

---

## 🔧 **Implementation Changes**

### **File: audio_preprocessing.py**

#### Before (WRONG):

```python
def analyze_audio_frequencies(self, data_dir="data"):
    """Calculate percentile-based filter parameters"""
    # Analyzes spectral content
    # Returns cutoff frequencies

def apply_band_pass_filter(self, audio, sr, low_cutoff, high_cutoff):
    """Apply band-pass filtering"""
    # Filters audio signal

def preprocess_audio_data(self, data_dir, output_base_dir):
    """Preprocess with filtering strategies"""
    # Applies filtering at preprocessing stage ❌
```

#### After (CORRECT):

```python
def preprocess_audio_basic(self, data_dir="data", output_dir="preprocessed_data_basic"):
    """
    Basic preprocessing: Format standardization ONLY
    NO FILTERING - Just convert to 16kHz mono WAV
    """
    for audio_file in audio_files:
        audio = load_audio(audio_file)
        audio_16k = resample(audio, target_sr=16000)  # ✅ Format only
        save_wav(audio_16k, output_dir)               # ✅ No filtering
```

### **File: comprehensive_pd_features_final.py**

#### Before (WRONG):

```python
# Input from filtered audio
input_dir = 'preprocessed_data_percentile_1_99/HC'  # ❌ Filtered audio
```

#### After (CORRECT):

```python
# Input from basic preprocessed audio
input_dir = 'preprocessed_data_basic/HC'  # ✅ Unfiltered audio
```

---

## 📝 **Updated Pseudo Code**

### **Preprocessing (Corrected)**

```
PROGRAM: Basic Audio Preprocessing (NO FILTERING)

FUNCTION preprocess_audio_basic(data_dir, output_dir):
    INPUT: Raw audio files

    FOR EACH audio_file IN data_dir:
        // Load audio
        audio, sr = load_audio(audio_file)

        // Resample to 16kHz if needed
        IF sr != 16000:
            audio = resample(audio, target_sr=16000)

        // Convert to mono if stereo
        IF audio.channels == 2:
            audio = convert_to_mono(audio)

        // Save as 16kHz mono WAV
        // ✅ NO FILTERING APPLIED
        save_wav(audio, output_dir, format='WAV')

    OUTPUT: Basic preprocessed audio (16kHz mono WAV)
```

### **Feature Extraction (Unchanged - Always Correct)**

```
PROGRAM: Feature Extraction

FUNCTION extract_features(preprocessed_audio):
    INPUT: Basic preprocessed audio (16kHz, unfiltered)

    PROCESS:
        // Extract all features from FULL audio signal
        voiced_segments = detect_voice_activity(audio)
        f0_values = estimate_pitch(voiced_segments)

        jitter_features = calculate_jitter(f0_values)
        shimmer_features = calculate_shimmer(voiced_segments)
        hnr_nhr = calculate_noise_features(voiced_segments, f0_values)
        prosodic = calculate_prosodic_features(f0_values)
        nonlinear = calculate_nonlinear_features(f0_values)
        additional = calculate_additional_features(audio)

    OUTPUT: Feature vector (35+ features)
```

### **Feature Selection (NEW - Optional)**

```
PROGRAM: Feature-Level Filtering (Optional)

FUNCTION filter_features_percentile(feature_matrix):
    INPUT: Extracted features (N samples × 35 features)

    PROCESS:
        FOR EACH feature_column IN feature_matrix:
            // Calculate percentiles for THIS FEATURE
            p_low = percentile(feature_column, 1)
            p_high = percentile(feature_column, 99)

            // Remove outliers in FEATURE space
            feature_column_filtered = remove_values_outside(
                feature_column,
                range=[p_low, p_high]
            )

    OUTPUT: Feature matrix with outliers removed
```

---

## 🎯 **When to Use Percentile Filtering**

### ✅ **Appropriate Use Cases:**

1. **Feature-Level Outlier Removal**

   ```python
   # Remove extreme feature values
   features_cleaned = remove_outliers(features, method='percentile', range=[1, 99])
   ```

2. **Feature Selection Based on Distribution**

   ```python
   # Select features with good separation
   selected = select_features_by_distribution(features, labels)
   ```

3. **Data Quality Control**
   ```python
   # Remove samples with anomalous feature values
   valid_samples = filter_samples_by_feature_range(features, percentile=[2.5, 97.5])
   ```

### ❌ **Inappropriate Use Cases:**

1. **Audio Signal Filtering at Preprocessing**

   ```python
   # ❌ DON'T DO THIS
   audio_filtered = bandpass_filter(audio, percentile_cutoffs)
   ```

2. **Frequency Domain Filtering Before Feature Extraction**
   ```python
   # ❌ DON'T DO THIS
   audio_spectrum = fft(audio)
   audio_spectrum_filtered = apply_percentile_mask(audio_spectrum)
   ```

---

## 📊 **Research Evidence**

### **Standard PD Voice Analysis Protocols:**

1. **MDVP (Kay Elemetrics)**

   - Features extracted from **unfiltered** voice recordings
   - Only basic high-pass filter (>50 Hz) to remove DC component

2. **Parkinson's Voice Initiative**

   - Raw audio → Feature extraction → Feature selection
   - No band-pass filtering at preprocessing stage

3. **UCI Parkinson's Dataset**

   - Features extracted from **raw** voice recordings
   - 22 features from unfiltered sustained phonations

4. **Research Literature**
   - Most papers: Feature extraction → Feature selection → Classification
   - Filtering (if any) applied to remove environmental noise, not signal content

---

## 🔄 **Migration Guide**

### **If You Already Have Filtered Audio:**

1. **Re-extract Features (Recommended)**

   ```bash
   # Step 1: Basic preprocessing (if not done)
   python audio_preprocessing.py

   # Step 2: Extract features from unfiltered audio
   python comprehensive_pd_features_final.py
   ```

2. **Keep Both Versions (For Comparison)**

   ```bash
   # Extract from unfiltered
   python comprehensive_pd_features_final.py --input preprocessed_data_basic/

   # Extract from filtered (for comparison)
   python comprehensive_pd_features_final.py --input preprocessed_data_percentile_1_99/
   ```

3. **Compare Results**
   - Check if feature distributions differ
   - Evaluate classification performance
   - Determine if filtering helped or hurt

---

## ✅ **Corrected File Structure**

```
project/
├── data/
│   ├── HC/                          # Raw audio files
│   └── PD/
│
├── preprocessed_data_basic/         # ✅ NEW: Basic preprocessed (16kHz, no filtering)
│   ├── HC/
│   └── PD/
│
├── comprehensive_features/
│   └── pd_features_comprehensive.csv  # Features from UNFILTERED audio
│
├── audio_preprocessing.py           # ✅ UPDATED: Basic preprocessing only
├── comprehensive_pd_features_final.py  # ✅ UPDATED: Uses basic preprocessed
└── feature_selection.py             # ✅ NEW: Feature-level filtering/selection
```

---

## 🎯 **Summary**

### **Key Changes:**

| Aspect                       | Before (❌)                  | After (✅)                |
| ---------------------------- | ---------------------------- | ------------------------- |
| **Preprocessing**            | 16kHz + Percentile filtering | 16kHz only                |
| **Feature Extraction Input** | Filtered audio               | Unfiltered audio          |
| **Percentile Application**   | Audio signal                 | Feature values (optional) |
| **Information Preservation** | Reduced                      | Complete                  |
| **Research Alignment**       | Non-standard                 | Standard practice         |

### **Action Items:**

1. ✅ Update `audio_preprocessing.py` to remove filtering
2. ✅ Run basic preprocessing: `python audio_preprocessing.py`
3. ✅ Re-extract features: `python comprehensive_pd_features_final.py`
4. ✅ Apply feature selection if needed
5. ✅ Train model on properly extracted features

---

## 📚 **References**

1. Tsanas et al. (2010) - "Accurate telemonitoring of Parkinson's disease"
2. Sakar et al. (2013) - "Collection and analysis of a Parkinson speech dataset"
3. MDVP Manual - Kay Elemetrics voice analysis standards
4. Little et al. (2009) - "Exploiting nonlinear recurrence and fractal scaling"

---

**এখন থেকে এই corrected workflow follow করবে!** ✅

**Key Point: Percentile filtering features নিয়ে করবে, audio signal নিয়ে নয়!** 🎯

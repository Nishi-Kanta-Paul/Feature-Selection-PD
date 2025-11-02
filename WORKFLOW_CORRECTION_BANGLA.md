# 🔧 WORKFLOW CORRECTION - Bangla Explanation

## ⚠️ গুরুত্বপূর্ণ সংশোধন: Percentile Filtering কখন করতে হবে

---

## 🎯 **মূল পরিবর্তন**

### **আগের ভুল Approach:**

```
Raw Audio → Percentile Filter Apply → Feature Extraction → Model Training
         ❌ WRONG!
```

### **সঠিক Approach:**

```
Raw Audio → Basic Preprocessing (শুধু 16kHz conversion) →
Feature Extraction → Optional Feature Filtering → Model Training
         ✅ CORRECT!
```

---

## 📊 **কেন এই পরিবর্তন?**

### **❌ আগের Approach এর সমস্যা:**

1. **Information Loss**

   - Audio signal এ percentile filter apply করলে অনেক frequency component remove হয়ে যায়
   - যেসব frequency তে PD এর markers থাকতে পারে, সেগুলো হারিয়ে যায়

2. **Irreversible**

   - একবার audio filter হলে original information আর ফিরে পাওয়া যায় না
   - Different filtering strategy try করার সুযোগ থাকে না

3. **Research Standard নয়**

   - বেশিরভাগ PD voice research এ unfiltered audio থেকে features extract করা হয়
   - MDVP, UCI Parkinson's dataset সব unfiltered audio use করে

4. **Feature Quality কমে**
   - HNR, NHR calculation এর জন্য full harmonic content দরকার
   - Nonlinear features এর জন্য complete signal complexity দরকার
   - Filtering করলে এসব features ঠিকমতো calculate হয় না

### **✅ নতুন Approach এর সুবিধা:**

1. **Complete Information**

   - পুরো audio signal এর সব frequency content preserve করা
   - সব ধরনের features সঠিকভাবে extract করা যায়

2. **Flexible**

   - Features extract করার পর বিভিন্ন feature selection method try করা যায়
   - Percentile filtering চাইলে features এ apply করা যায়

3. **Research Compliant**

   - Standard PD voice analysis protocol follow করা
   - Published research এর সাথে comparison করা যায়

4. **Better Performance**
   - Model training এর জন্য better quality features পাওয়া যায়

---

## 🔄 **Updated Workflow - Step by Step**

### **Step 1: Basic Audio Preprocessing**

**File:** `audio_preprocessing.py`

```python
def preprocess_audio_basic(data_dir="data", output_dir="preprocessed_data_basic"):
    """
    শুধুমাত্র format standardization
    কোনো filtering নেই
    """
    for audio_file in audio_files:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)

        # Save as 16kHz mono WAV (NO FILTERING!)
        sf.write(output_path, audio, sr=16000)
```

**যা করবে:**

- ✅ 16kHz এ resample করবে
- ✅ Mono channel এ convert করবে
- ✅ WAV format এ save করবে
- ❌ কোনো frequency filtering করবে না

**Output:**

```
preprocessed_data_basic/
├── HC/
│   ├── preprocessed_0001.wav
│   ├── preprocessed_0002.wav
│   └── ...
└── PD/
    ├── preprocessed_0001.wav
    ├── preprocessed_0002.wav
    └── ...
```

---

### **Step 2: Feature Extraction**

**File:** `comprehensive_pd_features_final.py`

```python
# UNFILTERED audio থেকে features extract করবে
input_dir = 'preprocessed_data_basic/HC'  # ✅ Correct

# NOT from filtered audio
# input_dir = 'preprocessed_data_percentile_1_99/HC'  # ❌ Wrong
```

**যা করবে:**

- ✅ 35+ features extract করবে
- ✅ Jitter, Shimmer, HNR, F0, Nonlinear সব
- ✅ Full frequency content থেকে calculate করবে

**Output:**

```
comprehensive_features/
└── pd_features_comprehensive.csv
    (35+ columns, সব features)
```

---

### **Step 3: Feature Analysis (Optional Percentile Filtering)**

এখন তুমি চাইলে **features এ** percentile filtering apply করতে পারো:

```python
import pandas as pd

# Load features
df = pd.read_csv('comprehensive_features/pd_features_comprehensive.csv')

# Apply percentile filtering on FEATURES
for column in df.columns:
    if column not in ['group', 'filename']:
        # Calculate 1st and 99th percentile
        p1 = df[column].quantile(0.01)
        p99 = df[column].quantile(0.99)

        # Remove outliers
        df = df[(df[column] >= p1) & (df[column] <= p99)]

# Save cleaned features
df.to_csv('comprehensive_features/pd_features_cleaned.csv', index=False)
```

---

### **Step 4: Feature Selection**

```python
# SHAP-based feature selection
from shap import TreeExplainer
from sklearn.ensemble import RandomForestClassifier

X = df.drop(['group', 'filename'], axis=1)
y = df['group']

model = RandomForestClassifier()
model.fit(X, y)

explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Select top features
feature_importance = np.abs(shap_values).mean(axis=0)
top_features = X.columns[np.argsort(feature_importance)[-15:]]
```

---

### **Step 5: Model Training**

```python
# Selected features দিয়ে model train করো
X_selected = df[top_features]
y = df['group']

# Train classifier
model = RandomForestClassifier()
model.fit(X_selected, y)
```

---

## 📝 **Pseudo Code - Corrected Workflow**

### **Preprocessing (সংশোধিত)**

```
PROGRAM: Basic Audio Preprocessing

FUNCTION preprocess_audio_basic():
    INPUT: Raw audio files (data/HC/, data/PD/)

    FOR EACH audio_file:
        // Step 1: Load audio
        audio, sr = load_audio(audio_file)

        // Step 2: Resample to 16kHz
        IF sr != 16000:
            audio = resample(audio, target_sr=16000)

        // Step 3: Convert to mono
        IF audio.channels > 1:
            audio = to_mono(audio)

        // Step 4: Save (NO FILTERING!)
        save_wav(audio, output_path, sr=16000)

        // ✅ Filtering SKIP করা হচ্ছে

    OUTPUT: preprocessed_data_basic/
```

---

### **Feature Extraction (আগের মতোই)**

```
PROGRAM: Feature Extraction

FUNCTION extract_features():
    INPUT: Basic preprocessed audio (preprocessed_data_basic/)

    FOR EACH audio_file:
        // Full bandwidth audio থেকে features
        audio = load_wav(audio_file)

        // Voice activity detection
        voiced_frames, f0_values, periods = analyze_voice(audio)

        // All features calculate করো
        jitter = calculate_jitter(periods)
        shimmer = calculate_shimmer(voiced_frames)
        hnr_nhr = calculate_noise_features(voiced_frames, f0_values)
        prosodic = calculate_prosodic(f0_values)
        nonlinear = calculate_nonlinear(f0_values)

        features = COMBINE_ALL(jitter, shimmer, hnr_nhr, prosodic, nonlinear)

    OUTPUT: comprehensive_features/pd_features_comprehensive.csv
```

---

### **Feature Filtering (নতুন - Optional)**

```
PROGRAM: Feature-Level Percentile Filtering

FUNCTION filter_features_percentile():
    INPUT: Extracted features CSV

    FOR EACH feature_column:
        // Feature values এ percentile apply করো
        p_low = percentile(feature_column, 1)
        p_high = percentile(feature_column, 99)

        // Outlier samples remove করো
        REMOVE samples WHERE feature_value < p_low OR feature_value > p_high

    OUTPUT: Cleaned feature set
```

---

## 📂 **File Structure (Updated)**

```
project/
├── data/                              # Original raw audio
│   ├── HC/
│   └── PD/
│
├── preprocessed_data_basic/           # ✅ NEW: Basic preprocessed (NO filtering)
│   ├── HC/
│   │   ├── preprocessed_0001.wav
│   │   └── ...
│   └── PD/
│       ├── preprocessed_0001.wav
│       └── ...
│
├── comprehensive_features/
│   ├── pd_features_comprehensive.csv   # All features (unfiltered audio থেকে)
│   └── pd_features_cleaned.csv         # Feature-level filtering (optional)
│
├── audio_preprocessing.py             # ✅ UPDATED: Basic preprocessing only
├── comprehensive_pd_features_final.py # Feature extraction
└── feature_selection.py               # Feature selection & filtering
```

---

## 🎯 **তোমার করণীয়**

### **1. Code Update করো**

```bash
# audio_preprocessing.py ইতিমধ্যে update করা হয়েছে
# শুধু basic preprocessing run করো

python audio_preprocessing.py
```

**Output:**

- `preprocessed_data_basic/HC/` - Healthy Control files
- `preprocessed_data_basic/PD/` - Parkinson's Disease files

### **2. Feature Extraction করো**

```bash
# Unfiltered audio থেকে features extract করো
python comprehensive_pd_features_final.py
```

**Output:**

- `comprehensive_features/pd_features_comprehensive.csv`

### **3. Feature Selection করো (Optional)**

এখন তুমি features নিয়ে যা খুশি করতে পারো:

- Percentile-based outlier removal
- SHAP-based feature selection
- Mutual information-based selection
- Correlation analysis

---

## 📊 **তুলনা: আগে vs পরে**

| Aspect                | আগে (❌ Wrong)              | এখন (✅ Correct)           |
| --------------------- | --------------------------- | -------------------------- |
| **Preprocessing**     | 16kHz + Percentile filter   | শুধু 16kHz                 |
| **Audio Signal**      | Filtered (information loss) | Unfiltered (complete)      |
| **Features**          | Incomplete signal থেকে      | Complete signal থেকে       |
| **Flexibility**       | Fixed filtering             | Flexible feature selection |
| **Research Standard** | Non-standard                | Standard practice          |
| **Performance**       | Suboptimal                  | Optimal                    |

---

## ✅ **Summary - মূল বিষয়**

### **Key Changes:**

1. **Preprocessing:**

   - ❌ আগে: Audio filtering করতাম
   - ✅ এখন: শুধু format standardization (16kHz, mono, WAV)

2. **Feature Extraction:**

   - ✅ Unfiltered audio থেকে সব features extract করো
   - ✅ পুরো frequency content preserve করো

3. **Percentile Filtering:**

   - ❌ আগে: Audio signal এ apply করতাম
   - ✅ এখন: Features এ apply করো (optional)

4. **Workflow:**
   ```
   Raw Audio → Basic Preprocessing → Feature Extraction →
   Feature Filtering (optional) → Feature Selection → Model Training
   ```

---

## 🚀 **Next Steps**

1. ✅ `audio_preprocessing.py` update হয়ে গেছে
2. → Run basic preprocessing
3. → Extract features from unfiltered audio
4. → Apply feature selection
5. → Train model

**এখন থেকে এই workflow follow করবে!** 🎯

**মনে রাখো: Percentile filtering → Features এ করতে হবে, Audio তে নয়!** ✅

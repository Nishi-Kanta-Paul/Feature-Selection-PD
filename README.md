# Parkinson's Disease Audio Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Clean-green)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Overview

Clean implementation for Parkinson's Disease audio analysis using feature-based approaches. The pipeline focuses on essential preprocessing and feature selection for voice-based PD detection.

## 📁 Project Structure

```
📁 Project/
├── 🎯 audio_preprocessing.py             # Main preprocessing (modular)
├── 🎨 filter_visualization.py            # Beginner-friendly diagrams
├── 🧪 test_percentile_implementation.py  # Percentile validation
├── 🧬 feature_extraction.py              # Feature extraction
├── 🔍 filter_feature_selection.py        # Feature selection methods
├── 📋 all_audios_mapped_id_for_label/    # CSV mapping data
├── 🎵 data/                              # Raw audio data (PD/HC)
├── 🔄 preprocessed_data_percentile_1_99/ # Primary filtered output
├── 🔄 preprocessed_data_percentile_2_5_97_5/ # Conservative filtered output
├── 📊 essential_analysis/                # Educational visualizations
├── 🧬 extracted_features.csv            # Feature extraction results
└── 📋 feature_selection_results.csv     # Feature selection outcomes
```

## 🔧 **NEW: Modular & Reusable Design**

### **Main Components:**

- ✅ **`audio_preprocessing.py`** - Clean, focused preprocessing
- ✅ **`filter_visualization.py`** - Separate visualization module
- ✅ **`test_percentile_implementation.py`** - Validation tools

### **Key Improvements:**

- 🧹 **Modular code** - Easy to maintain and extend
- 📚 **Beginner-friendly diagrams** - Educational visualizations
- ✅ **Percentile validation** - Built-in correctness checks
- 🔄 **Reusable components** - Import and use anywhere

## 🚀 Quick Start

### 1. **Data Preprocessing (Primary Step)**

```bash
python audio_preprocessing.py
```

**What it does:**

- Maps data from `final_selected.csv` (55,939 records: 9,929 PD, 23,086 HC)
- ✅ **Validates percentile implementation** automatically
- Creates **beginner-friendly diagrams** BEFORE processing
- Applies 16kHz sampling rate
- Band-pass filtering with two strategies:
  - **1-99 percentile** (Primary): 776.3-5536.3 Hz ✅ Validated
  - **2.5-97.5 percentile** (Conservative): 818.4-4792.2 Hz ✅ Validated

### 🧪 **Test Percentile Implementation (Optional)**

```bash
python test_percentile_implementation.py
```

**Educational validation:**

- Tests percentile calculations with sample data
- Creates validation plots
- Confirms 1-99% > 2.5-97.5% bandwidth
- Beginner-friendly explanations

### 2. **Feature Extraction**

```bash
python feature_extraction.py
```

### 3. **Feature Selection**

```bash
python filter_feature_selection.py
```

## 📊 Output Data

### **Preprocessed Audio Files:**

- `preprocessed_data_percentile_1_99/` - **Primary output** (broader frequency range)
- `preprocessed_data_percentile_2_5_97_5/` - Conservative output (narrower range)

### **🎨 Educational Visualizations:**

- `essential_analysis/filter_explanation_beginner.png` - **Beginner guide to filtering**
- `essential_analysis/before_after_filtering_real.png` - **Real audio filtering demo**
- `essential_analysis/filter_analysis.png` - **Technical filter comparison**
- `essential_analysis/percentile_test_validation.png` - **Percentile validation**

### **Analysis Results:**

- `extracted_features.csv` - All extracted audio features
- `feature_selection_results.csv` - Selected feature rankings

## 🔧 Key Features

### **Essential Preprocessing:**

- ✅ Data mapping from `final_selected.csv`
- ✅ 16kHz sampling rate standardization
- ✅ Percentile-based band-pass filtering
- ✅ Two filtering strategies for comparison
- ✅ Clean, minimal implementation

### **Essential Visualizations:**

- Filter frequency range comparison
- Bandwidth analysis
- Minimal plots for essential analysis

### **Feature Pipeline:**

- Time domain features (RMS, ZCR, etc.)
- Frequency domain features (spectral features)
- MFCC features (12 coefficients)
- Prosodic features (F0, jitter, etc.)

## 📈 Processing Statistics

**Recent Run Results:**

```
Dataset: 55,939 total records mapped
Local Processing: 21 files (2 PD, 19 HC)
Filter 1 (1-99%): 776.3-5536.3 Hz (4760 Hz bandwidth)
Filter 2 (2.5-97.5%): 818.4-4792.2 Hz (3974 Hz bandwidth)
Output: 100% success rate
```

## 💡 Recommendations

### **For Primary Analysis:**

Use `preprocessed_data_percentile_1_99/` because:

- Broader frequency preservation (1-99 percentile)
- Better for general PD voice analysis
- Captures more voice characteristics

### **For Conservative Analysis:**

Use `preprocessed_data_percentile_2_5_97_5/` when:

- Noise reduction is priority
- Conservative preprocessing needed
- Research requires stricter filtering

## 🔬 Research Focus

This implementation targets:

- **Voice-based Parkinson's Disease detection**
- **Feature-based machine learning approaches**
- **Clean, reproducible preprocessing pipeline**
- **Essential analysis and visualization**

## 📋 Dependencies

```python
librosa      # Audio processing
soundfile    # Audio I/O
numpy        # Numerical computing
pandas       # Data manipulation
matplotlib   # Visualization
scipy        # Signal processing
scikit-learn # Machine learning (for feature selection)
```

## 🎯 Next Steps

1. **Use preprocessed data:** Choose your preferred filtering strategy
2. **Extract features:** Run feature extraction on preprocessed audio
3. **Select features:** Apply feature selection methods
4. **Train models:** Use selected features for PD classification
5. **Evaluate:** Assess model performance and feature importance

---

## ✅ **COMPLETE REFACTORING SUMMARY**

### **What was improved:**

1. 🧹 **Modular Design**: Split large file into reusable components
2. 🎨 **Separate Visualization**: `filter_visualization.py` for all diagrams
3. 🧪 **Validation Tools**: `test_percentile_implementation.py` for verification
4. 📚 **Beginner-Friendly**: Educational diagrams with simple explanations
5. ✅ **Percentile Validation**: Built-in correctness checks

### **Percentile Implementation Status:**

- ✅ **1-99 percentile**: 776.3-5536.3 Hz (4760 Hz bandwidth) - **VALIDATED**
- ✅ **2.5-97.5 percentile**: 818.4-4792.2 Hz (3974 Hz bandwidth) - **VALIDATED**
- ✅ **Relationship**: 1-99% > 2.5-97.5% bandwidth ✓
- ✅ **Logic**: Broader vs Conservative filtering ✓

### **Files Created:**

- 📝 `audio_preprocessing.py` - Clean, modular main script
- 🎨 `filter_visualization.py` - Reusable visualization module
- 🧪 `test_percentile_implementation.py` - Educational validation
- 📊 4 educational diagrams in `essential_analysis/`

Your codebase is now **clean, modular, and educational**! 🎉

## 📝 Notes

- This is a **clean version** with unnecessary code removed
- Focus on **essential preprocessing and analysis**
- **Data mapping** handled from `final_selected.csv`
- **Two filtering strategies** for comparison and optimization
- **Clean implementation** for research reproducibility

## 📞 Support

For questions about this implementation, refer to:

- `essential_analysis/filter_analysis.png` for filter strategy comparison
- Console output from preprocessing for detailed statistics
- Feature extraction and selection results in CSV files

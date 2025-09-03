# Advanced Audio Preprocessing - Usage Guide

## 🚀 How to Run the Updated Pipeline

### 1. **Main Preprocessing Pipeline**
```bash
# Run the complete percentile-based preprocessing pipeline
python advanced_audio_preprocessing.py
```

**What it does:**
- Analyzes your dataset (PD and HC audio files)
- Calculates frequency percentiles from your data
- Applies 3 different filtering strategies:
  - **1st-99th percentile** (Primary - broadest range)
  - **2.5th-97.5th percentile** (Conservative)
  - **95% energy range** (Energy-based)

### 2. **Generated Outputs**

#### **Filtered Audio Data:**
```
percentile_1_99_filtered_data/          # 🎯 PRIMARY OUTPUT
├── PD/
│   ├── percentile_1_99_0001.wav
│   └── percentile_1_99_0002.wav
└── HC/
    ├── percentile_1_99_0001.wav
    ├── percentile_1_99_0002.wav
    └── ... (19 HC files)

percentile_2_5_97_5_filtered_data/      # 🛡️ CONSERVATIVE
├── PD/ & HC/ (same structure)

frequency_range_95_filtered_data/       # ⚡ ENERGY-BASED
├── PD/ & HC/ (same structure)
```

#### **Analysis Visualizations:**
```
preprocessing_analysis/
├── audio_length_distribution.png      # Audio duration analysis
├── silence_analysis.png              # Silence ratio comparison
├── frequency_profiles.png             # Frequency characteristics
├── signal_statistics.png             # Signal variability
├── below_80hz_analysis.png           # Low frequency content
├── upgraded_processing_pipeline_diagram.png  # Pipeline flowchart
└── filtering_strategy_comparison.png  # Strategy comparison
```

### 3. **Filter Parameters Applied**

Based on your dataset analysis:
- **1st-99th Percentile:** `380.9-1257.8 Hz` (877 Hz bandwidth)
- **2.5th-97.5th Percentile:** `422.1-1234.4 Hz` (812 Hz bandwidth) 
- **95% Energy Range:** `125.0-1789.1 Hz` (1664 Hz bandwidth)

### 4. **Which Output to Use?**

#### **🎯 For Primary Analysis:**
Use `percentile_1_99_filtered_data/` because:
- Preserves 99% of frequency content
- Optimal for PD voice analysis
- Minimal signal loss
- Best balance of noise removal and content preservation

#### **🛡️ For Conservative Analysis:**
Use `percentile_2_5_97_5_filtered_data/` if:
- Your data has significant noise
- You want more aggressive filtering
- Research requires conservative preprocessing

#### **⚡ For Energy-Focused Analysis:**
Use `frequency_range_95_filtered_data/` for:
- Spectral energy studies
- Wide frequency range analysis
- When low-frequency content is important

### 5. **Next Steps: Feature Extraction**

```bash
# Run feature extraction on preprocessed data
python feature_extraction.py
```

**Modify feature_extraction.py to use your preferred filtered data:**
```python
# In feature_extraction.py, change the input directory:
features_df = extractor.process_dataset("percentile_1_99_filtered_data")
```

### 6. **Processing Statistics**

From your recent run:
```
✅ Dataset: 21 files (2 PD, 19 HC)
✅ Length retention: 100% (no truncation)
✅ Silence handling: Selective removal only
✅ Amplitude: Preserved (no normalization)
✅ Duration: Natural variability maintained
```

### 7. **Quality Assurance**

Each filtering strategy maintains:
- **Original audio length:** ~10.04 seconds
- **No amplitude normalization:** Preserves natural dynamics
- **No length standardization:** Maintains natural duration variability
- **Selective silence removal:** Only removes excessive silence

### 8. **Troubleshooting**

If you encounter issues:
```bash
# Check if data directory exists
dir data\PD
dir data\HC

# Verify Python dependencies
pip install librosa soundfile numpy matplotlib pandas scipy

# Run with error details
python advanced_audio_preprocessing.py 2>&1 | tee preprocessing_log.txt
```

### 9. **Advanced Usage**

#### **Use Specific Percentile Strategy Only:**
```python
from advanced_audio_preprocessing import AdvancedAudioPreprocessor

preprocessor = AdvancedAudioPreprocessor(target_sr=16000)
preprocessor.analyze_dataset_for_preprocessing("data")

# Use only 1st-99th percentile
preprocessor.preprocess_with_percentile_filtering(
    "data", 
    "my_custom_output", 
    "percentile_1_99"
)
```

#### **Create Custom Visualizations:**
```bash
# Generate additional diagrams
python create_upgraded_pipeline_diagram.py
```

### 10. **File Organization**

```
📁 Your Project/
├── 🎯 percentile_1_99_filtered_data/     # USE THIS for primary analysis
├── 🛡️ percentile_2_5_97_5_filtered_data/  # Conservative backup
├── ⚡ frequency_range_95_filtered_data/   # Energy-based alternative
├── 📊 preprocessing_analysis/            # All visualizations
├── 🔧 advanced_audio_preprocessing.py    # Main preprocessing script
├── 📈 create_upgraded_pipeline_diagram.py # Diagram generator
└── 📋 Usage_Guide.md                     # This guide
```

---

## 🎉 Summary

**Primary Workflow:**
1. `python advanced_audio_preprocessing.py` → Generates 3 filtered datasets
2. Use `percentile_1_99_filtered_data/` for main analysis
3. `python feature_extraction.py` on filtered data
4. Continue with your feature selection pipeline

**The 1st-99th percentile strategy is optimized for Parkinson's Disease voice analysis!**

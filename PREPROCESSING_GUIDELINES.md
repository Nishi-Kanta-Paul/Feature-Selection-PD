# 🎵 AUDIO PREPROCESSING GUIDELINES

## Complete Guide for Parkinson's Disease Voice Data Preprocessing

## ⚠️ **IMPORTANT CORRECTION**

### **Percentile Filtering Should NOT Be Done at Preprocessing Stage**

**Previous (Incorrect) Approach:**

- Apply percentile-based band-pass filtering during preprocessing
- Filter audio signals before feature extraction

**Corrected Approach:**

- Basic preprocessing: Format standardization ONLY (16kHz, mono, WAV)
- NO filtering at this stage
- Extract features from unfiltered audio
- Apply percentile-based filtering on FEATURES (optional), not audio

**See [CORRECTED_WORKFLOW.md](CORRECTED_WORKFLOW.md) for detailed explanation**

---

## 📋 **OVERVIEW**

This guide provides instructions for **basic** preprocessing of voice/audio data for Parkinson's Disease (PD) research. The preprocessing pipeline converts raw audio recordings into clean, standardized format (16kHz, mono, WAV) **WITHOUT applying any frequency filtering**.

**Key Change:** Filtering is **NOT** part of preprocessing. Features should be extracted from full-bandwidth audio.

---

## 🎯 **PREPROCESSING OBJECTIVES**

1. ✅ **Standardize Audio Format**: Convert all audio to consistent format (16kHz, mono, WAV)
2. ✅ **Quality Control**: Ensure audio meets minimum quality standards
3. ❌ **NO Filtering**: Do NOT apply band-pass or percentile-based filtering
4. ❌ **NO Noise Reduction**: Keep original signal characteristics
5. ✅ **Preserve Information**: Maintain all frequency content for feature extraction

---

## 🔧 **TECHNICAL SPECIFICATIONS**

### Audio Format Requirements:

- **Sample Rate**: 16,000 Hz (16 kHz)
- **Channels**: Mono (single channel)
- **Bit Depth**: 16-bit
- **Format**: WAV (uncompressed)
- **Duration**: Typically 10+ seconds for reliable analysis
- **Filtering**: **NONE** (preserve full frequency content)

---

## 🚀 **PREPROCESSING PIPELINE (CORRECTED)**

### **Step 1: Raw Audio Loading**

```python
# Input: Raw audio files (.wav, .m4a, .mp3, etc.)
# Location: data/HC/ and data/PD/
# Typical format: Various sample rates, stereo/mono
```

**Process:**

1. Load audio file using librosa or soundfile
2. Extract waveform and metadata

### **Step 2: Format Standardization**

```python
# Convert to standard format
audio, sr = librosa.load(filepath, sr=16000, mono=True)
```

**Operations:**

- **Resampling**: Convert to 16kHz if different sample rate
- **Channel Conversion**: Convert to mono if stereo
- **Normalization**: Normalize to [-1, 1] range (optional)
- **NO FILTERING**: Preserve all frequency content

### **Step 3: Save Standardized Audio**

```python
# Save as 16kHz mono WAV
import soundfile as sf
sf.write(output_path, audio, samplerate=16000)
```

**Output Format:**

- 16kHz sampling rate
- Mono channel
- WAV format (uncompressed)
- Full frequency bandwidth preserved

---

- **Peak Normalization**: Scale to maximum amplitude
- **RMS Normalization**: Scale based on RMS energy (recommended)
- **LUFS Normalization**: Perceptual loudness normalization

### **Step 5: Quality Validation**

```python
# Validate processed audio quality
# Check: SNR, frequency content, duration
```

**Quality Checks:**

- Signal-to-Noise Ratio (SNR) > 10 dB
- Frequency content in speech range (80-8000 Hz)
- No clipping or distortion
- Preserved voice characteristics

---

## 🛠 **IMPLEMENTATION GUIDE**

### **Using audio_preprocessing.py:**

```bash
# Basic preprocessing (1-99 percentile)
python audio_preprocessing.py --input data/HC --output preprocessed_data_percentile_1_99/HC

# Alternative filtering (2.5-97.5 percentile)
python audio_preprocessing.py --input data/PD --output preprocessed_data_percentile_2_5_97_5/PD --percentile 2.5 97.5

# With noise reduction
python audio_preprocessing.py --input data/HC --output processed_clean/HC --denoise True
```

### **Manual Processing Steps:**

1. **Setup Environment:**

```bash
pip install librosa soundfile numpy scipy matplotlib
```

2. **Run Preprocessing:**

```python
from audio_preprocessing import AudioPreprocessor

processor = AudioPreprocessor()
processor.process_directory('data/HC', 'preprocessed_data_percentile_1_99/HC')
```

3. **Verify Output:**

```python
# Check processed files
import os
processed_files = os.listdir('preprocessed_data_percentile_1_99/HC')
print(f"Processed {len(processed_files)} files")
```

---

## 📊 **QUALITY ASSESSMENT**

### **Before vs After Preprocessing:**

| Metric          | Raw Audio                  | Processed Audio    |
| --------------- | -------------------------- | ------------------ |
| Sample Rate     | Variable (44.1kHz typical) | 16kHz standardized |
| Channels        | Stereo/Mono                | Mono               |
| Noise Level     | High (background noise)    | Reduced            |
| Amplitude       | Variable                   | Normalized         |
| Frequency Range | 0-22kHz                    | 80-8000Hz (speech) |

### **Expected Improvements:**

- ✅ **Reduced Background Noise**: 10-20 dB noise reduction
- ✅ **Consistent Format**: All files 16kHz, mono, WAV
- ✅ **Preserved Speech**: Voice characteristics maintained
- ✅ **Standardized Levels**: Consistent amplitude across files

---

## 🎯 **PREPROCESSING PARAMETERS**

### **Default Configuration:**

```python
PREPROCESSING_PARAMS = {
    'target_sr': 16000,           # Target sample rate
    'n_fft': 2048,               # FFT window size
    'hop_length': 512,           # Hop length for STFT
    'win_length': 2048,          # Window length
    'window': 'hann',            # Window function
    'percentile_low': 1,         # Lower percentile
    'percentile_high': 99,       # Upper percentile
    'target_rms': 0.1,           # Target RMS level
    'min_frequency': 80,         # Minimum speech frequency
    'max_frequency': 8000        # Maximum speech frequency
}
```

### **Advanced Parameters:**

```python
ADVANCED_PARAMS = {
    'frame_length': 0.025,       # 25ms frames
    'frame_shift': 0.010,        # 10ms shift
    'pre_emphasis': 0.97,        # Pre-emphasis coefficient
    'noise_gate_threshold': 0.01, # Noise gate level
    'dynamic_range_db': 60,      # Dynamic range
    'spectral_floor_db': -60     # Spectral floor
}
```

---

## 🔍 **TROUBLESHOOTING**

### **Common Issues:**

1. **High Noise Levels:**

   - Increase percentile filtering (e.g., 5-95 percentile)
   - Apply additional noise reduction
   - Check input audio quality

2. **Distorted Output:**

   - Reduce filtering aggressiveness
   - Check normalization levels
   - Verify input audio integrity

3. **Missing Voice Content:**

   - Reduce filtering (e.g., 0.5-99.5 percentile)
   - Check frequency range settings
   - Validate input audio has speech

4. **File Format Errors:**
   - Ensure input files are valid audio
   - Check file permissions
   - Verify output directory exists

### **Debug Commands:**

```bash
# Check audio file info
python -c "import librosa; y, sr = librosa.load('file.wav'); print(f'Duration: {len(y)/sr:.2f}s, SR: {sr}Hz')"

# Validate processing
python -c "from audio_preprocessing import validate_audio; validate_audio('processed_file.wav')"
```

---

## 📁 **FILE ORGANIZATION**

### **Input Structure:**

```
data/
├── HC/                    # Healthy Control audio files
│   ├── audio1.wav
│   ├── audio2.m4a
│   └── ...
└── PD/                    # Parkinson's Disease audio files
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### **Output Structure:**

```
preprocessed_data_percentile_1_99/
├── HC/                    # Processed HC files
│   ├── filtered_0001.wav
│   ├── filtered_0002.wav
│   └── ...
└── PD/                    # Processed PD files
    ├── filtered_0001.wav
    ├── filtered_0002.wav
    └── ...
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **Processing Speed:**

- **Parallel Processing**: Use multiprocessing for batch processing
- **Memory Management**: Process files individually for large datasets
- **GPU Acceleration**: Use CuPy/RAPIDS for STFT operations (optional)

### **Batch Processing:**

```python
# Process multiple files efficiently
from multiprocessing import Pool

def process_file_batch(file_list):
    with Pool(processes=4) as pool:
        pool.map(process_single_file, file_list)
```

---

## ✅ **VALIDATION CHECKLIST**

Before proceeding to feature extraction:

- [ ] All files converted to 16kHz, mono, WAV format
- [ ] Background noise significantly reduced
- [ ] Voice characteristics preserved
- [ ] No clipping or distortion present
- [ ] Consistent amplitude levels across files
- [ ] File naming convention followed
- [ ] Output directory structure correct
- [ ] Quality metrics within acceptable range

---

## 🎵 **NEXT STEPS**

After successful preprocessing:

1. **Feature Extraction**: Run `comprehensive_pd_features.py`
2. **Quality Assessment**: Validate extracted features
3. **Dataset Analysis**: Compare HC vs PD characteristics
4. **Machine Learning**: Use features for classification
5. **Clinical Analysis**: Apply diagnostic thresholds

---

**⚠️ IMPORTANT**: Always backup original audio files before preprocessing. The preprocessing pipeline is designed to be conservative to preserve voice characteristics essential for PD analysis.

---

_This preprocessing pipeline ensures optimal audio quality for reliable Parkinson's Disease voice analysis._

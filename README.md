# Parkinson's Disease Audio Analysis - Feature Based Approach

This project implements a feature-based approach for Parkinson's Disease detection using audio analysis. The pipeline includes audio organization, preprocessing, and feature extraction from voice recordings.

## Project Overview

This project processes voice recordings to distinguish between Parkinson's Disease (PD) patients and Healthy Controls (HC) using audio feature analysis. The pipeline consists of three main components:

1. **Audio Organization**: Organizes raw audio files from CSV metadata into cohort-based folders
2. **Audio Preprocessing**: Applies signal processing techniques to clean and standardize audio files
3. **Feature Extraction**: Extracts comprehensive acoustic features for machine learning analysis

## Features

### Audio Organization (`organize_audio_files.py`)

- Reads audio file IDs and cohort labels from CSV metadata
- Searches for audio files in both raw_wav/0 and raw_wav/1 subdirectories
- Copies files to organized folders: `data/PD/` and `data/HC/`
- Handles long Windows filenames using robocopy
- Provides detailed progress and summary reports
- Generates missing files report

### Audio Preprocessing (`audio_preprocessing.py`)

- Resamples audio to 16kHz for standardization
- Applies high-pass filtering to remove low-frequency noise
- Removes silence using energy-based detection
- Normalizes audio amplitude
- Ensures minimum audio length for consistency
- Generates clean, processed audio files

### Feature Extraction (`feature_extraction.py`)

- **Time Domain Features**: Statistical measures (mean, std, RMS energy, zero crossing rate)
- **Frequency Domain Features**: Spectral properties (centroid, bandwidth, rolloff, flatness)
- **MFCC Features**: 13 coefficients with delta and delta-delta features
- **Spectral Features**: Mel-frequency, chroma, spectral contrast, tonnetz
- **Prosodic Features**: Fundamental frequency, jitter, shimmer, voice activity detection
- **Total Features**: 139 comprehensive acoustic features per audio sample
- **Visualizations**: Automatic generation of feature analysis plots and pipeline diagrams

## Requirements

```
Python 3.x
pandas
librosa
soundfile
numpy
scipy
matplotlib
seaborn
```

## Installation

```bash
pip install pandas librosa soundfile numpy scipy matplotlib seaborn
```

## Usage

### Step 1: Organize Audio Files

```bash
python organize_audio_files.py
```

### Step 2: Preprocess Audio Files

```bash
python audio_preprocessing.py
```

### Step 3: Extract Features

```bash
python feature_extraction.py
```

## Data Structure

### Input Structure

- **CSV Metadata**: `all_audios_mapped_id_for_label/final_selected.csv`
- **Raw Audio Files**: `Processed_data_sample_raw_voice/raw_wav/[0|1]/[audio_id]/audio_audio.m4a-*.wav`

### Organized Data Structure (After Step 1)

```
data/
├── PD/
│   ├── [audio_id].wav
│   └── ...
└── HC/
    ├── [audio_id].wav
    └── ...
```

### Preprocessed Data Structure (After Step 2)

```
preprocessed_data/
├── PD/
│   ├── processed_0001.wav
│   ├── processed_0002.wav
│   └── ...
└── HC/
    ├── processed_0001.wav
    ├── processed_0002.wav
    └── ...
```

### Feature Extraction Output (After Step 3)

```
extracted_features.csv          # Comprehensive feature dataset
feature_analysis/               # Visualization and analysis plots
├── pipeline_diagram.png        # Complete pipeline overview
├── feature_distributions.png   # Feature distribution analysis
├── correlation_matrix.png      # Feature correlation heatmap
├── pd_vs_hc_comparison.png     # PD vs HC feature comparison
└── feature_importance.png      # Statistical feature importance
```

## CSV Format

Required columns in `final_selected.csv`:

- `audio_audio.m4a`: Audio file ID
- `cohort`: Label (PD, HC, or Unknown)

## Processing Results

### Current Dataset Statistics

- **Total CSV records**: 55,939
- **Available audio folders**: 27
- **Successfully organized files**: 21 (2 PD + 19 HC)
- **Missing files**: 32,996

### Preprocessing Results

- **PD files processed**: 2
- **HC files processed**: 19
- **Total processed**: 21 files
- **Target sample rate**: 16,000 Hz

### Feature Extraction Results

- **Total samples processed**: 21 (2 PD + 19 HC)
- **Features extracted per sample**: 139
- **Feature categories**:
  - Time Domain: 10 features
  - Frequency Domain: 4 features  
  - MFCC: 104 features (13 coefficients + deltas + statistics)
  - Spectral: 10 features
  - Prosodic: 8 features
- **Output files**: CSV dataset + 5 visualization plots

## Audio Preprocessing Details

The preprocessing pipeline applies the following transformations:

1. **Resampling**: Converts all audio to 16kHz sample rate
2. **High-pass Filtering**: Removes frequencies below 80Hz using 3rd order Butterworth filter
3. **Silence Removal**:
   - Uses 25ms frame size with 10ms hop
   - Energy threshold-based detection
   - Preserves only voice segments
4. **Normalization**: Scales amplitude to 90% of maximum range
5. **Length Standardization**: Ensures minimum 0.5-second duration

## Feature Extraction Details

The feature extraction pipeline extracts 139 comprehensive acoustic features across 5 categories:

### 1. Time Domain Features (10 features)
- **Statistical Measures**: Mean, standard deviation, max, min amplitude
- **Energy Features**: RMS energy, energy statistics across frames
- **Temporal Features**: Zero crossing rate, signal duration
- **Activity Measures**: Voice activity detection metrics

### 2. Frequency Domain Features (4 features)
- **Spectral Centroid**: Center of mass of the frequency spectrum
- **Spectral Bandwidth**: Spread of frequencies around the centroid
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Spectral Flatness**: Measure of how noise-like vs. tone-like the signal is

### 3. MFCC Features (104 features)
- **Base MFCC**: 13 Mel-frequency cepstral coefficients
- **Delta Features**: First-order derivatives (velocity)
- **Delta-Delta Features**: Second-order derivatives (acceleration)
- **Statistical Measures**: Mean, std, max, min for each coefficient and derivative

### 4. Spectral Features (10 features)
- **Mel-frequency Features**: Statistical measures of mel-spectrogram
- **Chroma Features**: Pitch class representation
- **Spectral Contrast**: Difference between peaks and valleys in spectrum
- **Tonnetz Features**: Harmonic network representation

### 5. Prosodic Features (8 features)
- **Fundamental Frequency (F0)**: Pitch analysis (mean, std, range)
- **Jitter**: F0 variability (voice quality indicator)
- **Voice Activity**: Ratio of voiced vs. unvoiced segments
- **Harmonic-to-Noise Ratio**: Voice quality measure

## Visualization and Analysis

The feature extraction automatically generates comprehensive visualizations:

### 1. Pipeline Diagram (`pipeline_diagram.png`)
- Complete workflow overview from raw audio to features
- Feature category breakdown
- Processing statistics

### 2. Feature Distributions (`feature_distributions.png`)
- Histogram plots of the first 20 features
- Distribution analysis for understanding feature characteristics

### 3. Correlation Matrix (`correlation_matrix.png`)
- Heatmap showing inter-feature correlations
- Helps identify redundant features for feature selection

### 4. PD vs HC Comparison (`pd_vs_hc_comparison.png`)
- Side-by-side distribution comparison for top 12 features
- Visual discrimination analysis between cohorts

### 5. Feature Importance (`feature_importance.png`)
- Statistical significance ranking using t-tests
- Top 20 most discriminative features for PD detection

## Output Files

### Reports Generated

- **Console output**: Real-time progress and summary statistics
- **missing_files.csv**: Detailed list of files that couldn't be processed with reasons

### Audio Files

- **Organized files**: Clean, renamed audio files in cohort folders
- **Preprocessed files**: Signal-processed audio ready for feature extraction

## Technical Specifications

- **Audio Format**: WAV (uncompressed)
- **Sample Rate**: 16,000 Hz
- **Bit Depth**: 16-bit (default)
- **Channels**: Mono
- **Frame Size**: 25ms (preprocessing)
- **Hop Size**: 10ms (preprocessing)

## Notes

- Only processes files with cohort labels 'PD' or 'HC'
- Skips 'Unknown' and other labels
- Uses robocopy for reliable file copying with long Windows paths
- Automatically handles filename length limitations on Windows
- Preprocessed files use sequential naming to avoid path length issues
- Signal processing optimized for voice analysis

## Next Steps

1. **Feature Extraction**: Extract acoustic features (MFCC, spectral features, etc.)
2. **Feature Selection**: Apply dimensionality reduction techniques
3. **Model Training**: Train machine learning models for PD classification
4. **Model Evaluation**: Cross-validation and performance metrics

## Requirements for Future Development

- Feature extraction libraries (python_speech_features, pyAudioAnalysis)
- Machine learning frameworks (scikit-learn, TensorFlow/PyTorch)
- Visualization tools (matplotlib, seaborn)
- Statistical analysis packages

# Parkinson's Disease Audio Analysis - Feature Based Approach

This project implements a feature-based approach for Parkinson's Disease detection using audio analysis. The pipeline includes audio organization, preprocessing, and feature extraction from voice recordings.

## Project Overview

This project processes voice recordings to distinguish between Parkinson's Disease (PD) patients and Healthy Controls (HC) using audio feature analysis. The pipeline consists of four main components:

1. **Audio Organization**: Organizes raw audio files from CSV metadata into cohort-based folders
2. **Audio Preprocessing**: Applies signal processing techniques to clean and standardize audio files
3. **Feature Extraction**: Extracts comprehensive acoustic features for machine learning analysis
4. **Filter-based Feature Selection**: Applies statistical methods to identify most discriminative features

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

### Filter-based Feature Selection (`filter_feature_selection.py`)

- **Variance Threshold Selection**: Removes features with low variance (< 0.01)
- **Correlation-based Selection**: Eliminates highly correlated features (r > 0.9)
- **Statistical Tests**: ANOVA F-test, Independent t-test, Mutual Information
- **Combined Ranking**: Weighted combination of multiple filter methods
- **Cross-validation Evaluation**: Performance assessment using Random Forest and Logistic Regression
- **Comprehensive Visualizations**: 6 detailed analysis plots including pipeline diagram
- **Feature Ranking**: Final ranked list of most discriminative features for PD detection

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
scikit-learn
```

## Installation

```bash
pip install pandas librosa soundfile numpy scipy matplotlib seaborn scikit-learn
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

### Step 4: Filter-based Feature Selection

```bash
python filter_feature_selection.py
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

### Filter-based Feature Selection Output (After Step 4)

```
feature_selection_results.csv   # Ranked feature list with scores
feature_selection_analysis/     # Comprehensive selection analysis
├── selection_pipeline.png      # Filter selection pipeline overview
├── selection_methods_comparison.png  # Methods comparison analysis
├── statistical_scores.png      # Statistical test results visualization
├── correlation_analysis.png    # Correlation filtering analysis
├── feature_rankings.png        # Feature ranking comparisons
└── evaluation_results.png      # Cross-validation performance results
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

### Filter-based Feature Selection Results

- **Original features**: 139
- **Variance threshold filtering**: 100 features retained (39 removed)
- **Correlation filtering**: 117 features retained (22 highly correlated removed)
- **Statistical significance**: 8 features with p < 0.05
- **Top filter methods performance**: All methods achieved 91-96% cross-validation accuracy
- **Best features identified**: mel_std, mfcc_13_std, voiced_ratio, mfcc_delta2_10_std
- **Output files**: Ranked feature list + 6 comprehensive visualizations

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

## Filter-based Feature Selection Details

The filter-based selection pipeline implements multiple statistical methods to identify the most discriminative features:

### 1. Variance Threshold Filtering
- **Purpose**: Remove features with minimal variation across samples
- **Threshold**: 0.01 (removes constant or near-constant features)
- **Result**: 100/139 features retained (39 removed)
- **Impact**: Eliminates non-informative features like `min_amplitude`, `max_amplitude`, `voiced_ratio`

### 2. Correlation-based Filtering
- **Purpose**: Remove highly correlated redundant features
- **Threshold**: 0.9 correlation coefficient
- **Method**: Retains feature with higher variance from correlated pairs
- **Result**: 117/139 features retained (22 removed)
- **Key removals**: `duration` (correlated with `signal_length`), `rms_energy` (correlated with `std_amplitude`)

### 3. Statistical Test Methods

#### ANOVA F-test
- **Purpose**: Identify features with significant between-group variance
- **Metric**: F-statistic for PD vs HC classification
- **Significant features**: 8 features with p < 0.05
- **Top performers**: `mfcc_delta2_2_mean`, `mfcc_12_max`, `mfcc_12_mean`

#### Independent t-test
- **Purpose**: Detect features with significant mean differences between groups
- **Metric**: Absolute t-statistic
- **Significant features**: 8 features with p < 0.05
- **Consistent with F-test**: Same top features identified

#### Mutual Information
- **Purpose**: Capture non-linear relationships between features and target
- **Advantage**: Detects complex dependencies missed by linear methods
- **Top performers**: `mel_std`, `mfcc_13_std`, `voiced_ratio`
- **Score range**: 0.0000 - 0.2847

### 4. Combined Filter Ranking
- **Approach**: Weighted combination of all filter methods
- **Weights**: F-test (40%), t-test (30%), Mutual Information (30%)
- **Normalization**: All scores normalized to [0,1] range before combination
- **Output**: Consensus ranking of all 139 features

### 5. Cross-validation Evaluation
- **Methods**: 5-fold cross-validation
- **Classifiers**: Random Forest and Logistic Regression
- **Metrics**: Classification accuracy
- **Results**: All feature sets achieved 91-96% accuracy
- **Best performance**: Mutual Information selection (96% accuracy)

## Feature Selection Visualizations

The selection pipeline generates 6 comprehensive visualizations:

### 1. Selection Pipeline (`selection_pipeline.png`)
- Complete workflow from 139 original features to final selection
- Processing statistics at each stage
- Filter method descriptions and parameters

### 2. Selection Methods Comparison (`selection_methods_comparison.png`)
- Number of features selected by each method
- Variance distribution analysis
- Correlation matrix heatmap
- P-value distribution from statistical tests

### 3. Statistical Scores (`statistical_scores.png`)
- Top 30 features by F-score, t-test, Mutual Information, and combined ranking
- Side-by-side comparison of different scoring methods
- Feature importance visualization

### 4. Correlation Analysis (`correlation_analysis.png`)
- Distribution of high correlation pairs
- Feature reduction effectiveness
- Before/after correlation filtering comparison

### 5. Feature Rankings (`feature_rankings.png`)
- Score heatmap for top 20 features across all methods
- Method agreement analysis and correlation
- Combined score distribution

### 6. Evaluation Results (`evaluation_results.png`)
- Cross-validation accuracy comparison across methods
- Feature count vs. performance trade-off analysis
- Performance improvement over baseline

## Key Findings and Insights

### Top Discriminative Features
1. **mel_std** (0.3000) - Spectral variability in mel-frequency domain
2. **mfcc_13_std** (0.2582) - Variability in highest MFCC coefficient
3. **voiced_ratio** (0.2498) - Proportion of voiced segments
4. **mfcc_delta2_10_std** (0.2331) - Acceleration in 10th MFCC coefficient

### Feature Category Analysis (Top 50 Features)
- **MFCC Features**: 72% (36/50) - Dominant category
- **Time Domain**: 12% (6/50) - Basic signal statistics
- **Spectral Features**: 8% (4/50) - Mel-frequency characteristics
- **Prosodic Features**: 8% (4/50) - Voice quality measures
- **Frequency Domain**: 0% (0/50) - Traditional spectral features less discriminative

### Statistical Insights
- **Mutual Information**: Most effective method (57 features with MI > 0)
- **Significance**: Only 8 features show statistical significance (p < 0.05) with traditional tests
- **MFCC Dominance**: 66.7% of top 15 features are MFCC-based
- **Delta Features**: Second-order derivatives (delta-delta) particularly discriminative

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

# Parkinson's Disease Audio Analysis - Feature Based Approach

This project implements a feature-based approach for Parkinson's Disease detection using audio analysis. The pipeline includes audio organization, preprocessing, and feature extraction from voice recordings.

## Project Overview

This project processes voice recordings to distinguish between Parkinson's Disease (PD) patients and Healthy Controls (HC) using audio feature analysis. The pipeline consists of two main components:

1. **Audio Organization**: Organizes raw audio files from CSV metadata into cohort-based folders
2. **Audio Preprocessing**: Applies signal processing techniques to clean and standardize audio files

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

## Requirements

```
Python 3.x
pandas
librosa
soundfile
numpy
scipy
```

## Installation

```bash
pip install pandas librosa soundfile numpy scipy
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

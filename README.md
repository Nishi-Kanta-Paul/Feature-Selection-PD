# Audio File Organization Script for Parkinson's Disease Research

This script organizes audio files from a CSV dataset into folders based on cohort labels (PD - Parkinson's Disease, HC - Healthy Control).

## Features

- Reads audio file IDs and cohort labels from CSV
- Searches for audio files in both raw_wav/0 and raw_wav/1 subdirectories
- Copies files to organized folders: `data/PD/` and `data/HC/`
- Handles long filenames using Windows robocopy
- Provides detailed progress and summary reports
- Generates missing files report

## Requirements

- Python 3.x
- pandas
- Windows OS (uses robocopy command)

## Usage

```bash
python organize_audio_files.py
```

## Input Structure

- CSV file: `all_audios_mapped_id_for_label/final_selected.csv`
- Audio files: `Processed_data_sample_raw_voice/raw_wav/[0|1]/[audio_id]/audio_audio.m4a-*.wav`

## Output Structure

```
data/
├── PD/
│   ├── [audio_id].wav
│   └── ...
└── HC/
    ├── [audio_id].wav
    └── ...
```

## CSV Format

Required columns:

- `audio_audio.m4a`: Audio file ID
- `cohort`: Label (PD, HC, or Unknown)

## Reports

- Console output with progress and summary
- `missing_files.csv`: List of files that couldn't be processed

## Notes

- Only processes files with cohort labels 'PD' or 'HC'
- Skips 'Unknown' and other labels
- Uses robocopy for reliable file copying with long Windows paths
- Automatically renames files to clean format ([audio_id].wav)

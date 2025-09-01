# Enhanced Frequency Filtering for Parkinson's Disease Audio Analysis

This implementation provides advanced frequency filtering using dataset percentiles to determine optimal cutoff frequencies for high-pass and band-pass filters.

## Overview

The enhanced frequency filtering system analyzes the entire dataset to determine optimal filter parameters based on statistical percentiles, ensuring that the filtering is adapted to the specific characteristics of your audio data.

## Key Features

### 1. Dataset-Wide Frequency Analysis
- Analyzes spectral centroids, spectral rolloffs, and dominant frequencies across all audio files
- Calculates frequency ranges with significant energy content
- Determines optimal filter parameters using statistical percentiles

### 2. Percentile-Based Filter Design
- **1st-99th percentiles**: Broad filtering approach for maximum frequency range preservation
- **2.5th-97.5th percentiles**: Conservative filtering approach for balanced noise reduction
- **Voice-optimized filtering**: Focuses on typical voice frequency ranges (80-8000 Hz)

### 3. Multiple Filtering Strategies
- **Broad Filter**: Uses 1st-99th percentiles for minimal frequency content loss
- **Conservative Filter**: Uses 2.5th-97.5th percentiles for balanced performance (recommended)
- **Voice Optimized Filter**: Optimized for speech analysis with voice-specific constraints

## Implementation Details

### Filter Design Process

1. **Dataset Analysis Phase**:
   ```python
   extractor = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=True)
   filter_params = extractor.analyze_dataset_frequencies("preprocessed_data")
   ```

2. **Filter Parameter Calculation**:
   - Analyzes spectral centroids, rolloffs, and dominant frequencies
   - Calculates percentiles: [1, 2.5, 5, 10, 90, 95, 97.5, 99]
   - Determines cutoff frequencies based on percentile combinations

3. **Filter Application**:
   - High-pass filter: Removes low-frequency noise below the calculated cutoff
   - Band-pass filter: Focuses on the voice frequency range
   - Butterworth filters (3rd order) for smooth frequency response

### Filter Parameters Determined

For each filtering strategy, the system calculates:
- **Low cutoff frequency**: Based on spectral centroids and dominant frequency percentiles
- **High cutoff frequency**: Based on spectral rolloff percentiles
- **Filter description**: Human-readable summary of the frequency range

Example output:
```
Conservative Filter: 2.5th-97.5th percentile range: 120.5-6234.7 Hz
Voice Optimized Filter: Voice-optimized range: 95.3-4891.2 Hz
Broad Filter: 1st-99th percentile range: 85.1-7892.4 Hz
```

## Usage

### Basic Usage

1. **Run the enhanced feature extraction**:
   ```python
   python feature_extraction.py
   ```

2. **Run the demonstration script**:
   ```python
   python demo_enhanced_filtering.py
   ```

### Advanced Usage

```python
from feature_extraction import AudioFeatureExtractor

# Initialize with enhanced filtering
extractor = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=True)

# Analyze dataset frequencies
filter_params = extractor.analyze_dataset_frequencies("preprocessed_data")

# Extract features with specific filter type
features_df = extractor.process_dataset(
    "preprocessed_data", 
    filter_type="conservative_filter"
)

# Save results
extractor.save_features("extracted_features_filtered.csv")
```

### Available Filter Types

- `"conservative_filter"` (recommended): Balanced approach using 2.5th-97.5th percentiles
- `"voice_optimized_filter"`: Optimized for voice analysis
- `"broad_filter"`: Minimal filtering using 1st-99th percentiles

## File Outputs

### Feature Files
- `extracted_features_conservative_filter.csv`: Features with conservative filtering
- `extracted_features_voice_optimized_filter.csv`: Features with voice-optimized filtering  
- `extracted_features_broad_filter.csv`: Features with broad filtering

### Visualization Directories
- `feature_analysis_conservative_filter/`: Visualizations for conservative filtering
- `feature_analysis_voice_optimized_filter/`: Visualizations for voice-optimized filtering
- `filtering_comparison/`: Comparison plots between filtering approaches

## Technical Implementation

### Frequency Analysis Process

1. **Load audio files** from the dataset (with optional sampling)
2. **Calculate spectral features**:
   - Spectral centroid (center of mass of spectrum)
   - Spectral rolloff (frequency below which 85% of energy is contained)
   - Dominant frequency (peak frequency in voice range)
   - Frequency range with significant energy (above 80th percentile)

3. **Statistical analysis**:
   - Calculate percentiles for each spectral feature
   - Determine optimal cutoff frequencies
   - Apply voice-specific constraints (80-8000 Hz range)

### Filter Implementation

```python
def apply_enhanced_frequency_filtering(self, audio, filter_type="conservative_filter"):
    """Apply percentile-based frequency filtering"""
    params = self.filter_params[filter_type]
    
    # High-pass filter (3rd order Butterworth)
    b_hp, a_hp = butter(3, params['low_cutoff'] / (self.sr / 2), btype='high')
    audio_filtered = filtfilt(b_hp, a_hp, audio)
    
    # Band-pass filter (if high cutoff < Nyquist)
    if params['high_cutoff'] < self.sr / 2:
        b_bp, a_bp = butter(3, [params['low_cutoff'], params['high_cutoff']], 
                           btype='band', fs=self.sr)
        audio_filtered = filtfilt(b_bp, a_bp, audio_filtered)
    
    return audio_filtered
```

## Benefits of Percentile-Based Filtering

### 1. Dataset Adaptation
- Filters adapt to the specific frequency characteristics of your dataset
- No need for manual tuning of cutoff frequencies
- Robust to variations in recording conditions and equipment

### 2. Noise Reduction
- Removes low-frequency environmental noise
- Filters out high-frequency artifacts
- Preserves important voice characteristics

### 3. Feature Quality Improvement
- Enhances signal-to-noise ratio before feature extraction
- Reduces artifacts in spectral features
- Improves discriminative power of extracted features

### 4. Flexibility
- Multiple filtering strategies for different analysis needs
- Easy to experiment with different percentile ranges
- Maintains compatibility with existing preprocessing pipelines

## Recommendations

### For General Use
- Use `"conservative_filter"` for balanced performance
- Provides good noise reduction while preserving voice characteristics
- Suitable for most Parkinson's disease audio analysis tasks

### For Voice-Specific Analysis
- Use `"voice_optimized_filter"` for speech-focused features
- Optimized for fundamental frequency and formant analysis
- Best for prosodic and voice quality features

### For Research and Experimentation
- Use `"broad_filter"` to preserve maximum frequency content
- Minimal filtering for exploring frequency-domain features
- Good baseline for comparing different filtering approaches

## Validation and Quality Assurance

The implementation includes several validation mechanisms:

1. **Filter parameter validation**: Ensures cutoff frequencies are within valid ranges
2. **Audio quality checks**: Skips very short or corrupted audio files
3. **Error handling**: Graceful fallback to original audio if filtering fails
4. **Visualization tools**: Comprehensive plots to verify filter performance

## Integration with Existing Pipeline

The enhanced filtering is designed to integrate seamlessly with the existing feature extraction pipeline:

- **Backward compatible**: Can be disabled with `enable_enhanced_filtering=False`
- **Flexible**: Works with existing preprocessed data
- **Efficient**: Frequency analysis is performed once and reused
- **Documented**: Comprehensive logging and visualization output

## Example Results

Typical filter parameters for Parkinson's disease audio data:

```
Conservative Filter: 2.5th-97.5th percentile range: 118.3-5847.2 Hz
  - Removes low-frequency noise below 118 Hz
  - Removes high-frequency artifacts above 5.8 kHz
  - Preserves critical voice frequency range

Voice Optimized Filter: Voice-optimized range: 92.7-4523.8 Hz
  - Focused on speech frequency range
  - Optimal for fundamental frequency analysis
  - Enhanced prosodic feature extraction
```

This percentile-based approach ensures that your frequency filtering is optimally tuned to the specific characteristics of your Parkinson's disease audio dataset, improving the quality and discriminative power of extracted features for machine learning applications.

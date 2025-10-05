#!/usr/bin/env python3
"""
WINDOW-BASED FEATURE EXTRACTION FOR PARKINSON'S DISEASE DETECTION
================================================================
Implements temporal window-based feature analysis with multiple window sizes
for capturing local dynamics and temporal variations in voice signals.

Window sizes: 5ms, 10ms, 20ms
Features: All comprehensive features computed per window + temporal statistics
"""

import os
import wave
import struct
import numpy as np
import pandas as pd
import csv
import statistics
import math
from datetime import datetime
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Import existing feature extraction classes
from comprehensive_pd_features import AudioLoader, VoiceAnalyzer

class WindowBasedFeatureExtractor:
    """
    Advanced window-based feature extraction for temporal voice analysis
    """
    
    def __init__(self, window_sizes_ms=[5, 10, 20], overlap_ratio=0.5):
        """
        Initialize window-based feature extractor
        
        Args:
            window_sizes_ms: List of window sizes in milliseconds
            overlap_ratio: Overlap between consecutive windows (0-1)
        """
        self.window_sizes_ms = window_sizes_ms
        self.overlap_ratio = overlap_ratio
        self.base_analyzer = VoiceAnalyzer()
        self.loader = AudioLoader()
        
        print(f"🔧 Window-Based Feature Extractor Initialized")
        print(f"   Window sizes: {window_sizes_ms} ms")
        print(f"   Overlap ratio: {overlap_ratio}")
    
    def ms_to_samples(self, ms, sample_rate):
        """Convert milliseconds to number of samples"""
        return int(ms * sample_rate / 1000)
    
    def create_windows(self, signal, window_size_samples, overlap_samples):
        """
        Create overlapping windows from signal
        
        Args:
            signal: Input audio signal
            window_size_samples: Window size in samples
            overlap_samples: Overlap in samples
            
        Returns:
            List of windowed signal segments
        """
        windows = []
        step_size = window_size_samples - overlap_samples
        
        for start in range(0, len(signal) - window_size_samples + 1, step_size):
            end = start + window_size_samples
            window = signal[start:end]
            windows.append(window)
        
        return windows
    
    def extract_basic_features_from_window(self, window, sample_rate):
        """
        Extract basic time-domain and frequency-domain features from a single window
        
        Args:
            window: Audio signal window
            sample_rate: Sampling rate
            
        Returns:
            Dictionary of features for this window
        """
        features = {}
        
        if len(window) == 0:
            return features
        
        # Convert to numpy array
        window = np.array(window)
        
        try:
            # 1. Time-domain features
            features['energy'] = np.sum(window ** 2)
            features['rms'] = np.sqrt(np.mean(window ** 2))
            features['zcr'] = np.sum(np.diff(np.sign(window)) != 0) / (2 * len(window))
            features['mean_amplitude'] = np.mean(np.abs(window))
            features['std_amplitude'] = np.std(window)
            features['skewness'] = skew(window)
            features['kurtosis'] = kurtosis(window)
            
            # 2. Simple pitch estimation using autocorrelation
            features.update(self._estimate_pitch_autocorr(window, sample_rate))
            
            # 3. Spectral features
            features.update(self._extract_spectral_features(window, sample_rate))
            
            # 4. Voice quality indicators
            features.update(self._extract_voice_quality(window, sample_rate))
            
        except Exception as e:
            print(f"    Warning: Feature extraction failed for window: {e}")
            
        return features
    
    def _estimate_pitch_autocorr(self, window, sample_rate):
        """Simple pitch estimation using autocorrelation"""
        features = {}
        
        try:
            # Autocorrelation
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find fundamental frequency
            min_period = int(sample_rate / 500)  # 500 Hz max
            max_period = int(sample_rate / 50)   # 50 Hz min
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    f0 = sample_rate / peak_idx if peak_idx > 0 else 0
                    
                    features['f0_autocorr'] = f0
                    features['f0_confidence'] = search_range[peak_idx - min_period] / np.max(autocorr) if np.max(autocorr) > 0 else 0
                else:
                    features['f0_autocorr'] = 0
                    features['f0_confidence'] = 0
            else:
                features['f0_autocorr'] = 0
                features['f0_confidence'] = 0
                
        except Exception:
            features['f0_autocorr'] = 0
            features['f0_confidence'] = 0
            
        return features
    
    def _extract_spectral_features(self, window, sample_rate):
        """Extract spectral features from window"""
        features = {}
        
        try:
            # Apply window function
            windowed = window * np.hanning(len(window))
            
            # FFT
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)
            
            if len(magnitude) > 1:
                # Frequency bins
                freqs = np.fft.rfftfreq(len(window), 1/sample_rate)
                
                # Spectral centroid
                if np.sum(magnitude) > 0:
                    features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
                else:
                    features['spectral_centroid'] = 0
                
                # Spectral bandwidth
                centroid = features['spectral_centroid']
                if np.sum(magnitude) > 0:
                    features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
                else:
                    features['spectral_bandwidth'] = 0
                
                # Spectral rolloff (85% of energy)
                cumsum = np.cumsum(magnitude)
                if cumsum[-1] > 0:
                    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
                    if len(rolloff_idx) > 0:
                        features['spectral_rolloff'] = freqs[rolloff_idx[0]]
                    else:
                        features['spectral_rolloff'] = freqs[-1]
                else:
                    features['spectral_rolloff'] = 0
                
                # Spectral flux (change in spectrum)
                features['spectral_flux'] = np.sum(np.diff(magnitude) ** 2)
                
            else:
                features['spectral_centroid'] = 0
                features['spectral_bandwidth'] = 0
                features['spectral_rolloff'] = 0
                features['spectral_flux'] = 0
                
        except Exception:
            features['spectral_centroid'] = 0
            features['spectral_bandwidth'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_flux'] = 0
            
        return features
    
    def _extract_voice_quality(self, window, sample_rate):
        """Extract voice quality features"""
        features = {}
        
        try:
            # Simple jitter estimation (period-to-period variation)
            zero_crossings = np.where(np.diff(np.sign(window)))[0]
            
            if len(zero_crossings) > 2:
                periods = np.diff(zero_crossings) * 2  # Approximate periods
                if len(periods) > 1:
                    # Local jitter
                    period_diffs = np.abs(np.diff(periods))
                    if len(periods) > 0:
                        features['local_jitter'] = np.mean(period_diffs) / np.mean(periods) if np.mean(periods) > 0 else 0
                    else:
                        features['local_jitter'] = 0
                    
                    # Period variability
                    features['period_variability'] = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0
                else:
                    features['local_jitter'] = 0
                    features['period_variability'] = 0
            else:
                features['local_jitter'] = 0
                features['period_variability'] = 0
            
            # Simple shimmer estimation (amplitude variation)
            peaks = []
            for i in range(1, len(window) - 1):
                if window[i] > window[i-1] and window[i] > window[i+1] and window[i] > 0:
                    peaks.append(window[i])
            
            if len(peaks) > 1:
                peak_diffs = np.abs(np.diff(peaks))
                features['local_shimmer'] = np.mean(peak_diffs) / np.mean(peaks) if np.mean(peaks) > 0 else 0
                features['amplitude_variability'] = np.std(peaks) / np.mean(peaks) if np.mean(peaks) > 0 else 0
            else:
                features['local_shimmer'] = 0
                features['amplitude_variability'] = 0
                
        except Exception:
            features['local_jitter'] = 0
            features['period_variability'] = 0
            features['local_shimmer'] = 0
            features['amplitude_variability'] = 0
            
        return features
    
    def aggregate_window_features(self, window_features_list, prefix=""):
        """
        Aggregate features across all windows using statistical measures
        
        Args:
            window_features_list: List of feature dictionaries from windows
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of aggregated features
        """
        if not window_features_list:
            return {}
        
        aggregated = {}
        
        # Get all feature names from first window
        feature_names = list(window_features_list[0].keys())
        
        for feature_name in feature_names:
            # Collect values across windows
            values = []
            for window_features in window_features_list:
                if feature_name in window_features and not np.isnan(window_features[feature_name]):
                    values.append(window_features[feature_name])
            
            if values:
                values = np.array(values)
                
                # Statistical aggregations
                aggregated[f"{prefix}{feature_name}_mean"] = np.mean(values)
                aggregated[f"{prefix}{feature_name}_std"] = np.std(values)
                aggregated[f"{prefix}{feature_name}_min"] = np.min(values)
                aggregated[f"{prefix}{feature_name}_max"] = np.max(values)
                aggregated[f"{prefix}{feature_name}_median"] = np.median(values)
                aggregated[f"{prefix}{feature_name}_range"] = np.max(values) - np.min(values)
                
                # Additional temporal statistics
                if len(values) > 1:
                    aggregated[f"{prefix}{feature_name}_skew"] = skew(values)
                    aggregated[f"{prefix}{feature_name}_kurtosis"] = kurtosis(values)
                    
                    # Trend analysis (linear slope)
                    x = np.arange(len(values))
                    if np.std(x) > 0:
                        correlation = np.corrcoef(x, values)[0, 1]
                        aggregated[f"{prefix}{feature_name}_trend"] = correlation if not np.isnan(correlation) else 0
                    else:
                        aggregated[f"{prefix}{feature_name}_trend"] = 0
                else:
                    aggregated[f"{prefix}{feature_name}_skew"] = 0
                    aggregated[f"{prefix}{feature_name}_kurtosis"] = 0
                    aggregated[f"{prefix}{feature_name}_trend"] = 0
            else:
                # Default values if no valid data
                for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'skew', 'kurtosis', 'trend']:
                    aggregated[f"{prefix}{feature_name}_{stat}"] = 0
        
        return aggregated
    
    def extract_windowed_features(self, filepath):
        """
        Extract comprehensive windowed features from audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary containing features for all window sizes
        """
        print(f"\n🔍 Processing: {os.path.basename(filepath)}")
        
        # Load audio
        audio_data = self.loader.load_wav_file(filepath)
        if audio_data is None:
            print("  ❌ Failed to load audio")
            return None
        
        signal = np.array(audio_data['signal'])
        sr = audio_data['sample_rate']
        duration = audio_data['duration']
        
        print(f"  📊 Duration: {duration:.2f}s, Sample Rate: {sr}Hz, Samples: {len(signal)}")
        
        all_features = {
            'filename': os.path.basename(filepath),
            'duration': duration,
            'sample_rate': sr,
            'total_samples': len(signal)
        }
        
        # Process each window size
        for window_ms in self.window_sizes_ms:
            print(f"  🪟 Processing {window_ms}ms windows...")
            
            # Calculate window parameters
            window_samples = self.ms_to_samples(window_ms, sr)
            overlap_samples = int(window_samples * self.overlap_ratio)
            
            if window_samples > len(signal):
                print(f"    ⚠️  Window size ({window_ms}ms = {window_samples} samples) larger than signal ({len(signal)} samples)")
                continue
            
            # Create windows
            windows = self.create_windows(signal, window_samples, overlap_samples)
            print(f"    📦 Created {len(windows)} windows")
            
            if not windows:
                continue
            
            # Extract features from each window
            window_features_list = []
            for i, window in enumerate(windows):
                try:
                    window_features = self.extract_basic_features_from_window(window, sr)
                    if window_features:
                        window_features_list.append(window_features)
                except Exception as e:
                    print(f"    ⚠️  Window {i} failed: {e}")
                    continue
            
            if window_features_list:
                print(f"    ✅ Successfully processed {len(window_features_list)} windows")
                
                # Aggregate features across windows
                prefix = f"w{window_ms}ms_"
                aggregated_features = self.aggregate_window_features(window_features_list, prefix)
                all_features.update(aggregated_features)
                
                # Add window-specific metadata
                all_features[f"w{window_ms}ms_num_windows"] = len(window_features_list)
                all_features[f"w{window_ms}ms_window_samples"] = window_samples
                all_features[f"w{window_ms}ms_overlap_samples"] = overlap_samples
            else:
                print(f"    ❌ No valid features extracted for {window_ms}ms windows")
        
        print(f"  🎯 Total features extracted: {len(all_features)}")
        return all_features
    
    def process_dataset(self, output_file="window_based_features.csv"):
        """
        Process entire dataset and extract windowed features
        
        Args:
            output_file: Output CSV file name
        """
        print("🚀 WINDOW-BASED FEATURE EXTRACTION FOR PD DETECTION")
        print("=" * 60)
        print(f"Window sizes: {self.window_sizes_ms} ms")
        print(f"Overlap ratio: {self.overlap_ratio}")
        print()
        
        all_results = []
        
        # Process HC and PD files
        for group, dirname in [('HC', 'preprocessed_data_percentile_1_99/HC'), 
                              ('PD', 'preprocessed_data_percentile_1_99/PD')]:
            
            print(f"📁 Processing {group} files from {dirname}")
            print("-" * 40)
            
            if not os.path.exists(dirname):
                print(f"  ❌ Directory not found: {dirname}")
                continue
            
            files = [f for f in os.listdir(dirname) if f.endswith('.wav')][:5]  # Limit for testing
            print(f"  Found {len(files)} audio files")
            
            for i, filename in enumerate(files, 1):
                filepath = os.path.join(dirname, filename)
                print(f"\n  [{i}/{len(files)}] {filename}")
                
                try:
                    features = self.extract_windowed_features(filepath)
                    if features:
                        features['group'] = group
                        all_results.append(features)
                        print(f"  ✅ Successfully extracted features")
                    else:
                        print(f"  ❌ Feature extraction failed")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
                    continue
        
        # Save results
        if all_results:
            print(f"\n💾 Saving results to {output_file}...")
            
            # Create DataFrame
            df = pd.DataFrame(all_results)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            print(f"✅ Successfully saved {len(all_results)} samples with {len(df.columns)} features")
            print(f"📊 Feature categories per window size:")
            
            for window_ms in self.window_sizes_ms:
                prefix = f"w{window_ms}ms_"
                window_features = [col for col in df.columns if col.startswith(prefix)]
                if window_features:
                    print(f"  {window_ms}ms: {len(window_features)} features")
            
            print(f"\n📁 Output file: {output_file}")
            
            # Display sample statistics
            print(f"\n📈 Dataset Statistics:")
            print(f"  Total samples: {len(df)}")
            if 'group' in df.columns:
                print(f"  Class distribution: {dict(df['group'].value_counts())}")
            
            return df
        else:
            print("❌ No features extracted")
            return None

class WindowedFeatureAnalyzer:
    """
    Analyzer for comparing different window-based features
    """
    
    def __init__(self, csv_file):
        """Load windowed features from CSV"""
        self.df = pd.read_csv(csv_file)
        self.window_sizes = self._detect_window_sizes()
        
    def _detect_window_sizes(self):
        """Detect available window sizes from column names"""
        window_sizes = []
        for col in self.df.columns:
            if col.startswith('w') and 'ms_' in col:
                size_str = col.split('ms_')[0][1:]  # Remove 'w' and get size
                try:
                    size = int(size_str)
                    if size not in window_sizes:
                        window_sizes.append(size)
                except ValueError:
                    continue
        return sorted(window_sizes)
    
    def compare_window_features(self):
        """Compare feature statistics across different window sizes"""
        print("🔍 WINDOW SIZE COMPARISON")
        print("=" * 40)
        
        for window_ms in self.window_sizes:
            prefix = f"w{window_ms}ms_"
            window_cols = [col for col in self.df.columns if col.startswith(prefix)]
            
            print(f"\n📊 {window_ms}ms Windows:")
            print(f"  Features: {len(window_cols)}")
            
            if window_cols:
                # Get numeric columns only
                numeric_cols = []
                for col in window_cols:
                    if self.df[col].dtype in ['int64', 'float64']:
                        numeric_cols.append(col)
                
                if numeric_cols:
                    # Calculate feature diversity
                    non_zero_features = (self.df[numeric_cols] != 0).sum().sum()
                    total_values = len(self.df) * len(numeric_cols)
                    
                    print(f"  Non-zero values: {non_zero_features}/{total_values} ({100*non_zero_features/total_values:.1f}%)")
                    
                    # Sample statistics
                    if len(numeric_cols) > 0:
                        means = self.df[numeric_cols].mean().mean()
                        stds = self.df[numeric_cols].std().mean()
                        print(f"  Average feature mean: {means:.6f}")
                        print(f"  Average feature std: {stds:.6f}")

def main():
    """Main window-based feature extraction pipeline"""
    
    # Initialize extractor
    extractor = WindowBasedFeatureExtractor(
        window_sizes_ms=[5, 10, 20],  # Multiple window sizes
        overlap_ratio=0.5             # 50% overlap
    )
    
    # Extract features
    output_file = "comprehensive_features/windowed_pd_features.csv"
    
    # Create output directory
    os.makedirs("comprehensive_features", exist_ok=True)
    
    # Process dataset
    df = extractor.process_dataset(output_file)
    
    if df is not None:
        # Analyze results
        analyzer = WindowedFeatureAnalyzer(output_file)
        analyzer.compare_window_features()
        
        print(f"\n🎉 Window-based feature extraction completed!")
        print(f"📁 Results saved to: {output_file}")
    else:
        print("❌ Feature extraction failed")

if __name__ == "__main__":
    main()
import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt, welch
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """
    Comprehensive Audio Feature Extractor for Parkinson's Disease Detection
    
    Extracts multiple categories of features:
    1. Time-domain features
    2. Frequency-domain features  
    3. Cepstral features (MFCC)
    4. Spectral features
    5. Prosodic features
    6. Voice quality features
    
    Now includes enhanced frequency filtering using dataset percentiles.
    """
    
    def __init__(self, sr=16000, enable_enhanced_filtering=True):
        self.sr = sr
        self.features_df = None
        self.enable_enhanced_filtering = enable_enhanced_filtering
        self.frequency_analysis_complete = False
        self.filter_params = None
        
    def analyze_dataset_frequencies(self, data_dir="preprocessed_data", sample_size=None):
        """
        Analyze frequency content across the dataset to determine optimal filter parameters
        """
        print("Analyzing dataset frequency content for optimal filtering...")
        print("=" * 60)
        
        all_spectral_centroids = []
        all_spectral_rolloffs = []
        all_dominant_freqs = []
        all_frequency_ranges = []
        
        processed_files = 0
        
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join(data_dir, cohort)
            
            if not os.path.exists(cohort_dir):
                print(f"Directory not found: {cohort_dir}")
                continue
                
            audio_files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
            
            # Limit sample size if specified
            if sample_size and len(audio_files) > sample_size:
                audio_files = audio_files[:sample_size]
            
            print(f"Analyzing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                try:
                    file_path = os.path.join(cohort_dir, filename)
                    
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=self.sr)
                    
                    # Skip very short files
                    if len(audio) < self.sr * 0.5:  # Less than 0.5 seconds
                        continue
                    
                    # Spectral analysis
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
                    
                    all_spectral_centroids.append(spectral_centroid)
                    all_spectral_rolloffs.append(spectral_rolloff)
                    
                    # Dominant frequency analysis
                    dominant_freq = self._find_dominant_frequency(audio, sr)
                    all_dominant_freqs.append(dominant_freq)
                    
                    # Frequency range with significant energy
                    freq_range = self._analyze_frequency_range(audio, sr)
                    all_frequency_ranges.append(freq_range)
                    
                    processed_files += 1
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Processed: {i+1}/{len(audio_files)} files")
                        
                except Exception as e:
                    print(f"  Error analyzing {filename}: {e}")
        
        # Calculate percentile-based filter parameters
        if processed_files > 0:
            self._calculate_filter_parameters(
                all_spectral_centroids, all_spectral_rolloffs, 
                all_dominant_freqs, all_frequency_ranges
            )
            self.frequency_analysis_complete = True
            
            print(f"\n{'='*60}")
            print("FREQUENCY ANALYSIS COMPLETE!")
            print(f"{'='*60}")
            print(f"Total files analyzed: {processed_files}")
            print("Filter parameters calculated using dataset percentiles:")
            
            for filter_type, params in self.filter_params.items():
                if filter_type != 'raw_data':
                    print(f"  {filter_type.replace('_', ' ').title()}: {params['description']}")
        else:
            print("No files processed for frequency analysis!")
            
        return self.filter_params
    
    def _find_dominant_frequency(self, audio, sr):
        """Find the dominant frequency in the audio signal"""
        # FFT analysis
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]
        
        # Find dominant frequency in voice range (80-8000 Hz)
        voice_mask = (pos_freqs >= 80) & (pos_freqs <= 8000)
        if np.any(voice_mask):
            voice_freqs = pos_freqs[voice_mask]
            voice_magnitudes = pos_magnitude[voice_mask]
            if len(voice_magnitudes) > 0:
                dominant_idx = np.argmax(voice_magnitudes)
                return voice_freqs[dominant_idx]
        return 0
    
    def _analyze_frequency_range(self, audio, sr):
        """Analyze frequency range with significant energy"""
        # Power spectral density
        freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)//4))
        
        # Find frequency range with top 80% of energy
        energy_threshold = np.percentile(psd, 80)
        significant_freqs = freqs[psd > energy_threshold]
        
        if len(significant_freqs) > 0:
            return (np.min(significant_freqs), np.max(significant_freqs))
        return (0, sr/2)
    
    def _calculate_filter_parameters(self, centroids, rolloffs, dominants, freq_ranges):
        """Calculate filter parameters based on dataset percentiles"""
        
        # Convert to numpy arrays and remove invalid values
        centroids = np.array([c for c in centroids if c > 0])
        rolloffs = np.array([r for r in rolloffs if r > 0])
        dominants = np.array([d for d in dominants if d > 0])
        
        # Extract frequency ranges
        low_freqs = [fr[0] for fr in freq_ranges if fr[0] > 0]
        high_freqs = [fr[1] for fr in freq_ranges if fr[1] < self.sr/2]
        
        # Calculate percentiles
        percentiles = [1, 2.5, 5, 10, 90, 95, 97.5, 99]
        
        # Strategy 1: Use 1st-99th percentiles for broad filtering
        broad_low = max(50, np.percentile(centroids, 1)) if len(centroids) > 0 else 80
        broad_high = min(self.sr//2, np.percentile(rolloffs, 99)) if len(rolloffs) > 0 else 8000
        
        # Strategy 2: Use 2.5th-97.5th percentiles for conservative filtering  
        conservative_low = max(80, np.percentile(centroids, 2.5)) if len(centroids) > 0 else 100
        conservative_high = min(8000, np.percentile(rolloffs, 97.5)) if len(rolloffs) > 0 else 6000
        
        # Strategy 3: Voice-optimized filtering (5th-95th percentiles with voice constraints)
        voice_low = max(80, np.percentile(dominants, 5)) if len(dominants) > 0 else 85
        voice_high = min(8000, np.percentile(rolloffs, 95)) if len(rolloffs) > 0 else 5000
        
        self.filter_params = {
            'broad_filter': {
                'low_cutoff': broad_low,
                'high_cutoff': broad_high,
                'description': f'1st-99th percentile range: {broad_low:.1f}-{broad_high:.1f} Hz'
            },
            'conservative_filter': {
                'low_cutoff': conservative_low,
                'high_cutoff': conservative_high,
                'description': f'2.5th-97.5th percentile range: {conservative_low:.1f}-{conservative_high:.1f} Hz'
            },
            'voice_optimized_filter': {
                'low_cutoff': voice_low,
                'high_cutoff': voice_high,
                'description': f'Voice-optimized range: {voice_low:.1f}-{voice_high:.1f} Hz'
            },
            'raw_data': {
                'centroids_percentiles': np.percentile(centroids, percentiles) if len(centroids) > 0 else [0]*len(percentiles),
                'rolloffs_percentiles': np.percentile(rolloffs, percentiles) if len(rolloffs) > 0 else [0]*len(percentiles),
                'dominants_percentiles': np.percentile(dominants, percentiles) if len(dominants) > 0 else [0]*len(percentiles),
                'percentile_labels': percentiles
            }
        }
        
        return self.filter_params
    
    def apply_enhanced_frequency_filtering(self, audio, filter_type="conservative_filter"):
        """
        Apply enhanced frequency filtering using dataset-derived parameters
        """
        if not self.enable_enhanced_filtering or self.filter_params is None:
            return audio
        
        if filter_type not in self.filter_params:
            print(f"Warning: Filter type '{filter_type}' not available. Using conservative_filter.")
            filter_type = "conservative_filter"
        
        params = self.filter_params[filter_type]
        low_cutoff = params['low_cutoff']
        high_cutoff = params['high_cutoff']
        
        try:
            # Step 1: High-pass filter (removes low-frequency noise)
            b_hp, a_hp = butter(3, low_cutoff / (self.sr / 2), btype='high')
            audio_filtered = filtfilt(b_hp, a_hp, audio)
            
            # Step 2: Band-pass filter (focuses on voice frequency range)
            if high_cutoff < self.sr / 2:  # Only apply if high cutoff is below Nyquist
                b_bp, a_bp = butter(3, [low_cutoff, high_cutoff], btype='band', fs=self.sr)
                audio_filtered = filtfilt(b_bp, a_bp, audio_filtered)
            
            return audio_filtered
            
        except Exception as e:
            print(f"Warning: Enhanced filtering failed ({e}). Using original audio.")
            return audio
        
    def extract_time_domain_features(self, audio):
        """Extract time-domain features"""
        features = {}
        
        # Basic statistical features
        features['mean_amplitude'] = np.mean(np.abs(audio))
        features['std_amplitude'] = np.std(audio)
        features['max_amplitude'] = np.max(np.abs(audio))
        features['min_amplitude'] = np.min(np.abs(audio))
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        
        # Zero crossing rate
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        features['zcr_std'] = np.std(librosa.feature.zero_crossing_rate(audio))
        
        # Energy features
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.01 * self.sr)     # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energies = np.sum(frames**2, axis=0)
        
        features['energy_mean'] = np.mean(energies)
        features['energy_std'] = np.std(energies)
        features['energy_max'] = np.max(energies)
        features['energy_min'] = np.min(energies)
        
        # Temporal features
        features['signal_length'] = len(audio)
        features['duration'] = len(audio) / self.sr
        
        return features
    
    def extract_frequency_domain_features(self, audio):
        """Extract frequency-domain features"""
        features = {}
        
        # FFT-based features
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/self.sr)
        
        # Only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(pos_freqs * pos_magnitude) / np.sum(pos_magnitude)
        
        # Spectral bandwidth
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(np.sum(((pos_freqs - centroid)**2) * pos_magnitude) / np.sum(pos_magnitude))
        
        # Spectral rolloff
        cumsum_mag = np.cumsum(pos_magnitude)
        rolloff_threshold = 0.85 * cumsum_mag[-1]
        rolloff_idx = np.where(cumsum_mag >= rolloff_threshold)[0]
        features['spectral_rolloff'] = pos_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral flatness
        features['spectral_flatness'] = stats.gmean(pos_magnitude + 1e-10) / np.mean(pos_magnitude + 1e-10)
        
        return features
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features"""
        features = {}
        
        # Extract 13 MFCC coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # Statistical measures for each MFCC coefficient
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
        
        # Delta and Delta-Delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Statistical measures for delta features
        for i in range(13):
            features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta_{i+1}_std'] = np.std(mfcc_delta[i])
            features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_delta2[i])
            features[f'mfcc_delta2_{i+1}_std'] = np.std(mfcc_delta2[i])
        
        return features
    
    def extract_spectral_features(self, audio):
        """Extract advanced spectral features"""
        features = {}
        
        # Mel-frequency features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        
        features['mel_mean'] = np.mean(mel_spec)
        features['mel_std'] = np.std(mel_spec)
        features['mel_max'] = np.max(mel_spec)
        features['mel_min'] = np.min(mel_spec)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=self.sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        return features
    
    def extract_prosodic_features(self, audio):
        """Extract prosodic and voice quality features"""
        features = {}
        
        # Fundamental frequency (F0) estimation
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sr)
        f0_clean = f0[f0 > 0]  # Remove unvoiced segments
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_max'] = np.max(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_range'] = features['f0_max'] - features['f0_min']
            
            # Jitter and Shimmer approximations
            f0_diff = np.diff(f0_clean)
            features['jitter_approx'] = np.std(f0_diff) / features['f0_mean'] if features['f0_mean'] > 0 else 0
            
            # Voice activity detection
            features['voiced_ratio'] = len(f0_clean) / len(f0)
        else:
            # Default values if no F0 detected
            for key in ['f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_range', 'jitter_approx', 'voiced_ratio']:
                features[key] = 0
        
        # Harmonic-to-noise ratio approximation
        harmonic = librosa.effects.harmonic(audio)
        percussive = librosa.effects.percussive(audio)
        
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        
        features['hnr_approx'] = 10 * np.log10(harmonic_energy / (percussive_energy + 1e-10))
        
        return features
    
    def extract_all_features(self, audio_path, filter_type="conservative_filter"):
        """Extract all feature categories for a single audio file with enhanced filtering"""
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.sr)
            
            # Apply enhanced frequency filtering if enabled and parameters are available
            if self.enable_enhanced_filtering and self.filter_params is not None:
                audio = self.apply_enhanced_frequency_filtering(audio, filter_type)
            
            # Extract features from all categories
            features = {}
            features.update(self.extract_time_domain_features(audio))
            features.update(self.extract_frequency_domain_features(audio))
            features.update(self.extract_mfcc_features(audio))
            features.update(self.extract_spectral_features(audio))
            features.update(self.extract_prosodic_features(audio))
            
            # Add metadata
            features['file_path'] = audio_path
            features['file_name'] = os.path.basename(audio_path)
            features['filter_applied'] = filter_type if self.enable_enhanced_filtering and self.filter_params is not None else 'none'
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_dataset(self, data_dir="preprocessed_data", filter_type="conservative_filter", 
                       enable_frequency_analysis=True):
        """Process entire dataset and extract features with enhanced filtering"""
        
        # Step 1: Perform frequency analysis if enabled
        if self.enable_enhanced_filtering and enable_frequency_analysis:
            print("Step 1: Analyzing dataset frequencies...")
            self.analyze_dataset_frequencies(data_dir)
            print()
        
        all_features = []
        
        print("Step 2: Starting feature extraction...")
        if self.enable_enhanced_filtering and self.filter_params is not None:
            filter_desc = self.filter_params[filter_type]['description']
            print(f"Using enhanced filtering: {filter_desc}")
        print("=" * 70)
        
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join(data_dir, cohort)
            
            if not os.path.exists(cohort_dir):
                print(f"Directory not found: {cohort_dir}")
                continue
                
            audio_files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
            print(f"\nProcessing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                file_path = os.path.join(cohort_dir, filename)
                features = self.extract_all_features(file_path, filter_type)
                
                if features:
                    features['cohort'] = cohort
                    features['cohort_numeric'] = 1 if cohort == 'PD' else 0
                    all_features.append(features)
                    
                    print(f"  Processed: {filename} ({i+1}/{len(audio_files)})")
                else:
                    print(f"  Failed: {filename}")
        
        # Create DataFrame
        if all_features:
            self.features_df = pd.DataFrame(all_features)
            print(f"\n{'='*70}")
            print("FEATURE EXTRACTION COMPLETE!")
            print(f"{'='*70}")
            print(f"Total samples: {len(self.features_df)}")
            print(f"PD samples: {len(self.features_df[self.features_df['cohort'] == 'PD'])}")
            print(f"HC samples: {len(self.features_df[self.features_df['cohort'] == 'HC'])}")
            
            # Get numerical features only (exclude metadata)
            numerical_cols = self.features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col not in ['cohort_numeric']]
            print(f"Total features extracted: {len(feature_cols)}")
            
            if self.enable_enhanced_filtering and self.filter_params is not None:
                print(f"Enhanced filtering applied: {filter_type}")
                print(f"Filter range: {self.filter_params[filter_type]['description']}")
            
            return self.features_df
        else:
            print("No features extracted!")
            return None
    
    def save_features(self, output_path="extracted_features.csv"):
        """Save extracted features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_path, index=False)
            print(f"Features saved to: {os.path.abspath(output_path)}")
        else:
            print("No features to save!")
    
    def create_feature_visualization(self, output_dir="feature_analysis"):
        """Create comprehensive feature analysis visualizations"""
        if self.features_df is None:
            print("No features available for visualization!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Select numerical features only
        numerical_cols = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in ['cohort_numeric']]
        
        # 1. Feature Distribution Analysis
        self._plot_feature_distributions(feature_cols[:20], output_dir)  # Plot first 20 features
        
        # 2. Correlation Analysis
        self._plot_correlation_matrix(feature_cols, output_dir)
        
        # 3. PD vs HC Comparison
        self._plot_cohort_comparison(feature_cols[:12], output_dir)  # Top 12 features
        
        # 4. Feature Importance Analysis
        self._plot_feature_importance(feature_cols, output_dir)
        
        # 5. Pipeline Overview Diagram
        self._create_pipeline_diagram(output_dir)
        
        print(f"Visualizations saved to: {os.path.abspath(output_dir)}")
    
    def _plot_feature_distributions(self, features, output_dir):
        """Plot feature distributions"""
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, feature in enumerate(features):
            if i >= 20:
                break
                
            axes[i].hist(self.features_df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{feature}', fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(features), 20):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, features, output_dir):
        """Plot correlation matrix"""
        # Select subset of features for readability
        subset_features = features[:30] if len(features) > 30 else features
        corr_matrix = self.features_df[subset_features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cohort_comparison(self, features, output_dir):
        """Plot PD vs HC comparison for selected features"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features):
            if i >= 12:
                break
                
            pd_data = self.features_df[self.features_df['cohort'] == 'PD'][feature]
            hc_data = self.features_df[self.features_df['cohort'] == 'HC'][feature]
            
            axes[i].hist(pd_data, bins=15, alpha=0.6, label='PD', color='red', density=True)
            axes[i].hist(hc_data, bins=15, alpha=0.6, label='HC', color='blue', density=True)
            axes[i].set_title(f'{feature}', fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Feature Comparison: PD vs HC', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pd_vs_hc_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, features, output_dir):
        """Plot feature importance using statistical tests"""
        from scipy.stats import ttest_ind
        
        feature_scores = []
        feature_names = []
        
        for feature in features[:50]:  # Limit to first 50 features
            pd_data = self.features_df[self.features_df['cohort'] == 'PD'][feature]
            hc_data = self.features_df[self.features_df['cohort'] == 'HC'][feature]
            
            # Skip if not enough data
            if len(pd_data) < 2 or len(hc_data) < 2:
                continue
                
            # T-test for significance
            _, p_value = ttest_ind(pd_data, hc_data)
            
            # Use negative log p-value as importance score
            importance = -np.log10(p_value + 1e-10)
            
            feature_scores.append(importance)
            feature_names.append(feature)
        
        # Sort by importance
        sorted_indices = np.argsort(feature_scores)[-20:]  # Top 20
        top_scores = [feature_scores[i] for i in sorted_indices]
        top_names = [feature_names[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_scores)), top_scores, color='lightcoral')
        plt.yticks(range(len(top_scores)), top_names)
        plt.xlabel('-log10(p-value)')
        plt.title('Top 20 Most Discriminative Features (PD vs HC)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pipeline_diagram(self, output_dir):
        """Create pipeline overview diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Define pipeline stages
        stages = [
            "Raw Audio Files\n(55,939 records)",
            "Audio Organization\n(organize_audio_files.py)",
            "Organized Dataset\n(21 files: 2 PD + 19 HC)",
            "Audio Preprocessing\n(audio_preprocessing.py)",
            "Preprocessed Audio\n(16kHz, filtered, normalized)",
            "Feature Extraction\n(feature_extraction.py)",
            "Feature Dataset\n(Time, Frequency, MFCC, etc.)"
        ]
        
        # Define positions
        positions = [
            (2, 9), (2, 7.5), (2, 6),
            (2, 4.5), (2, 3), (2, 1.5), (2, 0)
        ]
        
        # Draw boxes and arrows
        box_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        arrow_props = dict(arrowstyle="->", connectionstyle="arc3", color="darkblue", lw=2)
        
        for i, (stage, (x, y)) in enumerate(zip(stages, positions)):
            # Draw box
            ax.text(x, y, stage, ha='center', va='center', fontsize=11, 
                   bbox=box_props, fontweight='bold')
            
            # Draw arrow to next stage
            if i < len(stages) - 1:
                next_y = positions[i + 1][1]
                ax.annotate('', xy=(x, next_y + 0.4), xytext=(x, y - 0.4),
                           arrowprops=arrow_props)
        
        # Add feature categories on the right
        feature_categories = [
            "Time Domain Features:\n• Mean, Std, RMS Energy\n• Zero Crossing Rate\n• Signal Statistics",
            "Frequency Domain Features:\n• Spectral Centroid\n• Spectral Bandwidth\n• Spectral Rolloff",
            "MFCC Features:\n• 13 MFCC Coefficients\n• Delta & Delta-Delta\n• Statistical Measures",
            "Prosodic Features:\n• Fundamental Frequency\n• Jitter & Shimmer\n• Voice Activity",
            "Spectral Features:\n• Mel-frequency\n• Chroma Features\n• Spectral Contrast"
        ]
        
        for i, category in enumerate(feature_categories):
            y_pos = 8 - i * 1.6
            ax.text(6, y_pos, category, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 10)
        ax.set_title('Parkinson\'s Disease Audio Analysis Pipeline', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pipeline_diagram.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run feature extraction with enhanced frequency filtering"""
    
    print("=" * 80)
    print("PARKINSON'S DISEASE AUDIO FEATURE EXTRACTION")
    print("Enhanced with Percentile-Based Frequency Filtering")
    print("=" * 80)
    
    # Initialize feature extractor with enhanced filtering
    extractor = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=True)
    
    # Option 1: Extract features with different filter types
    filter_types = ["conservative_filter", "voice_optimized_filter", "broad_filter"]
    
    for filter_type in filter_types:
        print(f"\n{'='*80}")
        print(f"PROCESSING WITH {filter_type.replace('_', ' ').upper()}")
        print(f"{'='*80}")
        
        # Extract features with current filter type
        features_df = extractor.process_dataset("preprocessed_data", filter_type=filter_type)
        
        if features_df is not None:
            # Save features with filter type suffix
            output_file = f"extracted_features_{filter_type}.csv"
            extractor.save_features(output_file)
            
            # Create visualizations with filter type suffix
            visualization_dir = f"feature_analysis_{filter_type}"
            extractor.create_feature_visualization(visualization_dir)
            
            # Print feature summary
            print(f"\nFeature extraction summary for {filter_type}:")
            print(f"- Total samples: {len(features_df)}")
            print(f"- PD samples: {len(features_df[features_df['cohort'] == 'PD'])}")
            print(f"- HC samples: {len(features_df[features_df['cohort'] == 'HC'])}")
            
            # Get numerical features only
            numerical_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col not in ['cohort_numeric']]
            print(f"- Total features extracted: {len(feature_cols)}")
            
            # Feature categories count
            categories = {
                'Time Domain': len([f for f in feature_cols if any(td in f for td in ['mean_amplitude', 'std_amplitude', 'rms_energy', 'zcr', 'energy', 'duration'])]),
                'Frequency Domain': len([f for f in feature_cols if any(fd in f for fd in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness'])]),
                'MFCC': len([f for f in feature_cols if 'mfcc' in f]),
                'Spectral': len([f for f in feature_cols if any(sp in f for sp in ['mel_', 'chroma_', 'contrast_', 'tonnetz_'])]),
                'Prosodic': len([f for f in feature_cols if any(pr in f for pr in ['f0_', 'jitter', 'voiced_ratio', 'hnr'])])
            }
            
            print("\nFeature categories:")
            for category, count in categories.items():
                print(f"- {category}: {count} features")
            
            if extractor.filter_params is not None:
                filter_info = extractor.filter_params[filter_type]
                print(f"\nFilter applied: {filter_info['description']}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ENHANCED FEATURE EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    
    if extractor.filter_params is not None:
        print("Filter parameters determined using dataset percentiles:")
        for ftype, params in extractor.filter_params.items():
            if ftype != 'raw_data':
                print(f"  {ftype.replace('_', ' ').title()}: {params['description']}")
        
        print("\nOutput files generated:")
        for filter_type in filter_types:
            print(f"  - extracted_features_{filter_type}.csv")
            print(f"  - feature_analysis_{filter_type}/ (visualizations)")
    
    print("\nRecommendation: Use 'conservative_filter' for balanced performance")
    print("or 'voice_optimized_filter' for voice-specific analysis.")


def run_comparison_analysis():
    """
    Run comparison analysis between different filtering approaches
    """
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: FILTERED vs UNFILTERED")
    print("="*80)
    
    results_comparison = {}
    
    # Process without enhanced filtering
    print("\n1. Processing WITHOUT enhanced filtering...")
    extractor_basic = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=False)
    features_basic = extractor_basic.process_dataset("preprocessed_data")
    
    if features_basic is not None:
        extractor_basic.save_features("extracted_features_basic.csv")
        results_comparison['basic'] = {
            'total_samples': len(features_basic),
            'pd_samples': len(features_basic[features_basic['cohort'] == 'PD']),
            'hc_samples': len(features_basic[features_basic['cohort'] == 'HC']),
            'total_features': len([col for col in features_basic.select_dtypes(include=[np.number]).columns if col != 'cohort_numeric'])
        }
    
    # Process with enhanced filtering
    print("\n2. Processing WITH enhanced filtering (conservative)...")
    extractor_enhanced = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=True)
    features_enhanced = extractor_enhanced.process_dataset("preprocessed_data", filter_type="conservative_filter")
    
    if features_enhanced is not None:
        extractor_enhanced.save_features("extracted_features_enhanced.csv")
        results_comparison['enhanced'] = {
            'total_samples': len(features_enhanced),
            'pd_samples': len(features_enhanced[features_enhanced['cohort'] == 'PD']),
            'hc_samples': len(features_enhanced[features_enhanced['cohort'] == 'HC']),
            'total_features': len([col for col in features_enhanced.select_dtypes(include=[np.number]).columns if col != 'cohort_numeric']),
            'filter_params': extractor_enhanced.filter_params['conservative_filter'] if extractor_enhanced.filter_params else None
        }
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for approach, stats in results_comparison.items():
        print(f"\n{approach.upper()} APPROACH:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  PD samples: {stats['pd_samples']}")
        print(f"  HC samples: {stats['hc_samples']}")
        print(f"  Features extracted: {stats['total_features']}")
        
        if approach == 'enhanced' and 'filter_params' in stats and stats['filter_params']:
            print(f"  Filter applied: {stats['filter_params']['description']}")
    
    return results_comparison


if __name__ == "__main__":
    # Run main feature extraction with enhanced filtering
    main()
    
    # Optionally run comparison analysis
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        run_comparison_analysis()

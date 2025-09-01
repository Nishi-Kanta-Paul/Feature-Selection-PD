import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
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
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.features_df = None
        
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
    
    def extract_all_features(self, audio_path):
        """Extract all feature categories for a single audio file"""
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.sr)
            
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
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_dataset(self, data_dir="preprocessed_data"):
        """Process entire dataset and extract features"""
        all_features = []
        
        print("Starting feature extraction...")
        print("=" * 50)
        
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join(data_dir, cohort)
            
            if not os.path.exists(cohort_dir):
                print(f"Directory not found: {cohort_dir}")
                continue
                
            audio_files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
            print(f"\nProcessing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                file_path = os.path.join(cohort_dir, filename)
                features = self.extract_all_features(file_path)
                
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
            print(f"\n{'='*50}")
            print("FEATURE EXTRACTION COMPLETE!")
            print(f"{'='*50}")
            print(f"Total samples: {len(self.features_df)}")
            print(f"PD samples: {len(self.features_df[self.features_df['cohort'] == 'PD'])}")
            print(f"HC samples: {len(self.features_df[self.features_df['cohort'] == 'HC'])}")
            print(f"Total features: {len(self.features_df.columns) - 4}")  # Exclude metadata columns
            
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
    """Main function to run feature extraction"""
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(sr=16000)
    
    # Extract features from preprocessed data
    features_df = extractor.process_dataset("preprocessed_data")
    
    if features_df is not None:
        # Save features to CSV
        extractor.save_features("extracted_features.csv")
        
        # Create visualizations
        extractor.create_feature_visualization("feature_analysis")
        
        # Print feature summary
        print(f"\nFeature extraction summary:")
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


if __name__ == "__main__":
    main()

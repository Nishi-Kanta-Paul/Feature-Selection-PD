import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioPreprocessor:
    """
    Advanced Audio Preprocessor for Parkinson's Disease Analysis
    
    Implements:
    1. Frequency Filtering with percentile-based cutoffs (high-pass + band-pass)
    2. Silence Analysis (measure silence ratio in PD vs HC)
    3. No amplitude normalization or length standardization
    4. Audio length distribution analysis
    5. Signal comparison and frequency profile analysis
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.analysis_results = {}
        self.filter_params = None
        
    def analyze_dataset_for_preprocessing(self, data_dir="data"):
        """
        Step 1: Comprehensive dataset analysis for preprocessing decisions
        """
        print("="*80)
        print("DATASET ANALYSIS FOR PREPROCESSING")
        print("="*80)
        
        analysis_data = {
            'PD': {'files': [], 'durations': [], 'silence_ratios': [], 'freq_profiles': [], 
                   'below_80hz_ratios': [], 'std_devs': []},
            'HC': {'files': [], 'durations': [], 'silence_ratios': [], 'freq_profiles': [], 
                   'below_80hz_ratios': [], 'std_devs': []}
        }
        
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join(data_dir, cohort)
            
            if not os.path.exists(cohort_dir):
                print(f"Directory not found: {cohort_dir}")
                continue
                
            audio_files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
            print(f"\nAnalyzing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                try:
                    file_path = os.path.join(cohort_dir, filename)
                    
                    # Load audio at original sample rate first
                    audio_orig, sr_orig = librosa.load(file_path, sr=None)
                    # Then resample to target rate
                    audio, sr = librosa.load(file_path, sr=self.target_sr)
                    
                    # 1. Audio length analysis
                    duration = len(audio) / sr
                    analysis_data[cohort]['durations'].append(duration)
                    analysis_data[cohort]['files'].append(filename)
                    
                    # 2. Silence analysis
                    silence_ratio = self._calculate_silence_ratio(audio, sr)
                    analysis_data[cohort]['silence_ratios'].append(silence_ratio)
                    
                    # 3. Frequency profile analysis
                    freq_profile = self._analyze_frequency_profile(audio, sr)
                    analysis_data[cohort]['freq_profiles'].append(freq_profile)
                    
                    # 4. Signal below 80 Hz analysis
                    below_80hz_ratio = self._calculate_below_80hz_ratio(audio, sr)
                    analysis_data[cohort]['below_80hz_ratios'].append(below_80hz_ratio)
                    
                    # 5. Signal standard deviation
                    std_dev = np.std(audio)
                    analysis_data[cohort]['std_devs'].append(std_dev)
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Analyzed: {i+1}/{len(audio_files)} files")
                        
                except Exception as e:
                    print(f"  Error analyzing {filename}: {e}")
        
        self.analysis_results = analysis_data
        
        # Calculate filter parameters using percentiles
        self._calculate_percentile_based_filters()
        
        # Create analysis visualizations
        self._create_preprocessing_analysis_plots()
        
        # Print analysis summary
        self._print_analysis_summary()
        
        return self.analysis_results
    
    def _calculate_silence_ratio(self, audio, sr, threshold_percentile=10):
        """Calculate silence ratio using energy-based detection"""
        # Frame-based energy calculation
        frame_size = int(0.025 * sr)  # 25ms frames
        hop_size = int(0.01 * sr)     # 10ms hop
        
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)
        
        if len(energies) == 0:
            return 1.0  # All silence if no frames
        
        energies = np.array(energies)
        
        # Use percentile-based threshold (adaptive to each file)
        threshold = np.percentile(energies, threshold_percentile)
        
        # Calculate silence ratio
        silence_frames = energies <= threshold
        silence_ratio = np.sum(silence_frames) / len(silence_frames)
        
        return silence_ratio
    
    def _analyze_frequency_profile(self, audio, sr):
        """Analyze frequency profile and content distribution"""
        # Power spectral density
        freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)//4))
        
        # Calculate key frequency statistics
        profile = {
            'spectral_centroid': np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0,
            'spectral_bandwidth': np.sqrt(np.sum(((freqs - np.sum(freqs * psd) / np.sum(psd))**2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0,
            'spectral_rolloff': self._calculate_spectral_rolloff(freqs, psd),
            'dominant_freq': freqs[np.argmax(psd)] if len(psd) > 0 else 0,
            'freq_range_95': self._calculate_frequency_range(freqs, psd, 0.95),
            'freq_range_99': self._calculate_frequency_range(freqs, psd, 0.99)
        }
        
        return profile
    
    def _calculate_spectral_rolloff(self, freqs, psd, rolloff=0.85):
        """Calculate spectral rolloff frequency"""
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_energy = rolloff * total_energy
        
        rolloff_idx = np.where(cumsum_psd >= rolloff_energy)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return freqs[-1]
    
    def _calculate_frequency_range(self, freqs, psd, energy_percentage):
        """Calculate frequency range containing specified percentage of energy"""
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        
        # Find frequency range containing the specified percentage of energy
        lower_energy = (1 - energy_percentage) / 2 * total_energy
        upper_energy = (1 + energy_percentage) / 2 * total_energy
        
        lower_idx = np.where(cumsum_psd >= lower_energy)[0]
        upper_idx = np.where(cumsum_psd >= upper_energy)[0]
        
        lower_freq = freqs[lower_idx[0]] if len(lower_idx) > 0 else freqs[0]
        upper_freq = freqs[upper_idx[0]] if len(upper_idx) > 0 else freqs[-1]
        
        return (lower_freq, upper_freq)
    
    def _calculate_below_80hz_ratio(self, audio, sr):
        """Calculate percentage of signal energy below 80 Hz"""
        # FFT analysis
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = np.abs(fft[:len(fft)//2])
        
        # Calculate energy below 80 Hz
        below_80hz_mask = pos_freqs < 80
        energy_below_80hz = np.sum(pos_magnitude[below_80hz_mask]**2)
        total_energy = np.sum(pos_magnitude**2)
        
        if total_energy > 0:
            return energy_below_80hz / total_energy
        return 0
    
    def _calculate_percentile_based_filters(self):
        """Calculate filter parameters using dataset percentiles"""
        print("\nCalculating percentile-based filter parameters...")
        
        # Collect frequency data from all files
        all_centroids = []
        all_rolloffs = []
        all_dominant_freqs = []
        all_freq_ranges_95 = []
        all_freq_ranges_99 = []
        
        for cohort in ['PD', 'HC']:
            for profile in self.analysis_results[cohort]['freq_profiles']:
                all_centroids.append(profile['spectral_centroid'])
                all_rolloffs.append(profile['spectral_rolloff'])
                all_dominant_freqs.append(profile['dominant_freq'])
                all_freq_ranges_95.append(profile['freq_range_95'])
                all_freq_ranges_99.append(profile['freq_range_99'])
        
        # Convert to numpy arrays and remove invalid values
        centroids = np.array([c for c in all_centroids if c > 0])
        rolloffs = np.array([r for r in all_rolloffs if r > 0])
        dominants = np.array([d for d in all_dominant_freqs if d > 0])
        
        # Calculate percentiles
        percentiles = [1, 2.5, 5, 10, 90, 95, 97.5, 99]
        
        if len(centroids) > 0 and len(rolloffs) > 0:
            # Strategy 1: 1st-99th percentiles (broad filtering)
            broad_low = max(50, np.percentile(centroids, 1))  # Minimum 50 Hz
            broad_high = min(self.target_sr//2, np.percentile(rolloffs, 99))
            
            # Strategy 2: 2.5th-97.5th percentiles (conservative filtering)
            conservative_low = max(80, np.percentile(centroids, 2.5))  # Minimum 80 Hz
            conservative_high = min(8000, np.percentile(rolloffs, 97.5))  # Maximum 8 kHz
            
            # Strategy 3: Based on frequency ranges
            freq_95_lows = [fr[0] for fr in all_freq_ranges_95 if fr[0] > 0]
            freq_95_highs = [fr[1] for fr in all_freq_ranges_95 if fr[1] < self.target_sr//2]
            
            range_95_low = max(70, np.percentile(freq_95_lows, 5)) if freq_95_lows else 80
            range_95_high = min(6000, np.percentile(freq_95_highs, 95)) if freq_95_highs else 5000
            
            self.filter_params = {
                'percentile_1_99': {
                    'low_cutoff': broad_low,
                    'high_cutoff': broad_high,
                    'description': f'1st-99th percentile: {broad_low:.1f}-{broad_high:.1f} Hz'
                },
                'percentile_2_5_97_5': {
                    'low_cutoff': conservative_low,
                    'high_cutoff': conservative_high,
                    'description': f'2.5th-97.5th percentile: {conservative_low:.1f}-{conservative_high:.1f} Hz'
                },
                'frequency_range_95': {
                    'low_cutoff': range_95_low,
                    'high_cutoff': range_95_high,
                    'description': f'95% energy range: {range_95_low:.1f}-{range_95_high:.1f} Hz'
                },
                'raw_percentiles': {
                    'centroids': np.percentile(centroids, percentiles).tolist(),
                    'rolloffs': np.percentile(rolloffs, percentiles).tolist(),
                    'dominants': np.percentile(dominants, percentiles).tolist() if len(dominants) > 0 else [0]*len(percentiles),
                    'percentile_labels': percentiles
                }
            }
            
            print("Filter parameters calculated:")
            for filter_type, params in self.filter_params.items():
                if filter_type != 'raw_percentiles':
                    print(f"  {params['description']}")
        else:
            print("Warning: Insufficient data for filter parameter calculation")
            self.filter_params = None
    
    def _create_preprocessing_analysis_plots(self):
        """Create comprehensive analysis plots"""
        print("\nCreating preprocessing analysis visualizations...")
        
        os.makedirs("preprocessing_analysis", exist_ok=True)
        
        # 1. Audio length distribution
        self._plot_audio_length_distribution()
        
        # 2. Silence ratio comparison
        self._plot_silence_analysis()
        
        # 3. Frequency profile comparison
        self._plot_frequency_profiles()
        
        # 4. Signal statistics comparison
        self._plot_signal_statistics()
        
        # 5. Below 80 Hz analysis
        self._plot_below_80hz_analysis()
        
        print("Analysis plots saved to: preprocessing_analysis/")
    
    def _plot_audio_length_distribution(self):
        """Plot audio length variability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pd_durations = self.analysis_results['PD']['durations']
        hc_durations = self.analysis_results['HC']['durations']
        
        # Duration histograms
        axes[0,0].hist(pd_durations, bins=15, alpha=0.7, label='PD', color='red', density=True)
        axes[0,0].hist(hc_durations, bins=15, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,0].set_xlabel('Duration (seconds)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Audio Length Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0,1].boxplot([pd_durations, hc_durations], labels=['PD', 'HC'])
        axes[0,1].set_ylabel('Duration (seconds)')
        axes[0,1].set_title('Audio Length Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Duration statistics
        stats_data = [
            ['Group', 'Count', 'Mean (s)', 'Std (s)', 'Min (s)', 'Max (s)'],
            ['PD', len(pd_durations), f'{np.mean(pd_durations):.2f}', 
             f'{np.std(pd_durations):.2f}', f'{np.min(pd_durations):.2f}', f'{np.max(pd_durations):.2f}'],
            ['HC', len(hc_durations), f'{np.mean(hc_durations):.2f}', 
             f'{np.std(hc_durations):.2f}', f'{np.min(hc_durations):.2f}', f'{np.max(hc_durations):.2f}']
        ]
        
        axes[1,0].axis('tight')
        axes[1,0].axis('off')
        table = axes[1,0].table(cellText=stats_data[1:], colLabels=stats_data[0], 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1,0].set_title('Duration Statistics')
        
        # Individual file durations
        x_pd = range(len(pd_durations))
        x_hc = range(len(pd_durations), len(pd_durations) + len(hc_durations))
        
        axes[1,1].scatter(x_pd, pd_durations, color='red', alpha=0.7, label='PD')
        axes[1,1].scatter(x_hc, hc_durations, color='blue', alpha=0.7, label='HC')
        axes[1,1].set_xlabel('File Index')
        axes[1,1].set_ylabel('Duration (seconds)')
        axes[1,1].set_title('Individual File Durations')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Audio Length Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_analysis/audio_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_silence_analysis(self):
        """Plot silence ratio analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pd_silence = self.analysis_results['PD']['silence_ratios']
        hc_silence = self.analysis_results['HC']['silence_ratios']
        
        # Silence ratio histograms
        axes[0,0].hist(pd_silence, bins=15, alpha=0.7, label='PD', color='red', density=True)
        axes[0,0].hist(hc_silence, bins=15, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,0].set_xlabel('Silence Ratio')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Silence Ratio Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0,1].boxplot([pd_silence, hc_silence], labels=['PD', 'HC'])
        axes[0,1].set_ylabel('Silence Ratio')
        axes[0,1].set_title('Silence Ratio Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(pd_silence, hc_silence)
        
        axes[1,0].text(0.1, 0.8, f'Statistical Test Results:', fontsize=12, fontweight='bold', transform=axes[1,0].transAxes)
        axes[1,0].text(0.1, 0.7, f'PD mean: {np.mean(pd_silence):.3f} ± {np.std(pd_silence):.3f}', fontsize=10, transform=axes[1,0].transAxes)
        axes[1,0].text(0.1, 0.6, f'HC mean: {np.mean(hc_silence):.3f} ± {np.std(hc_silence):.3f}', fontsize=10, transform=axes[1,0].transAxes)
        axes[1,0].text(0.1, 0.5, f't-statistic: {t_stat:.3f}', fontsize=10, transform=axes[1,0].transAxes)
        axes[1,0].text(0.1, 0.4, f'p-value: {p_value:.3f}', fontsize=10, transform=axes[1,0].transAxes)
        significance = 'Significant' if p_value < 0.05 else 'Not significant'
        axes[1,0].text(0.1, 0.3, f'Difference: {significance}', fontsize=10, fontweight='bold', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Statistical Analysis')
        axes[1,0].axis('off')
        
        # Individual file silence ratios
        x_pd = range(len(pd_silence))
        x_hc = range(len(pd_silence), len(pd_silence) + len(hc_silence))
        
        axes[1,1].scatter(x_pd, pd_silence, color='red', alpha=0.7, label='PD')
        axes[1,1].scatter(x_hc, hc_silence, color='blue', alpha=0.7, label='HC')
        axes[1,1].set_xlabel('File Index')
        axes[1,1].set_ylabel('Silence Ratio')
        axes[1,1].set_title('Individual File Silence Ratios')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Silence Analysis: PD vs HC', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_analysis/silence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frequency_profiles(self):
        """Plot frequency profile comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract frequency statistics
        pd_profiles = self.analysis_results['PD']['freq_profiles']
        hc_profiles = self.analysis_results['HC']['freq_profiles']
        
        pd_centroids = [p['spectral_centroid'] for p in pd_profiles]
        hc_centroids = [p['spectral_centroid'] for p in hc_profiles]
        
        pd_rolloffs = [p['spectral_rolloff'] for p in pd_profiles]
        hc_rolloffs = [p['spectral_rolloff'] for p in hc_profiles]
        
        # Spectral centroids
        axes[0,0].hist(pd_centroids, bins=10, alpha=0.7, label='PD', color='red', density=True)
        axes[0,0].hist(hc_centroids, bins=10, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,0].set_xlabel('Spectral Centroid (Hz)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Spectral Centroid Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Spectral rolloffs
        axes[0,1].hist(pd_rolloffs, bins=10, alpha=0.7, label='PD', color='red', density=True)
        axes[0,1].hist(hc_rolloffs, bins=10, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,1].set_xlabel('Spectral Rolloff (Hz)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Spectral Rolloff Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Frequency percentiles plot
        if self.filter_params and 'raw_percentiles' in self.filter_params:
            percentiles = self.filter_params['raw_percentiles']['percentile_labels']
            centroid_percentiles = self.filter_params['raw_percentiles']['centroids']
            rolloff_percentiles = self.filter_params['raw_percentiles']['rolloffs']
            
            axes[1,0].plot(percentiles, centroid_percentiles, 'o-', label='Centroids', linewidth=2, markersize=8)
            axes[1,0].plot(percentiles, rolloff_percentiles, 's-', label='Rolloffs', linewidth=2, markersize=8)
            axes[1,0].set_xlabel('Percentile')
            axes[1,0].set_ylabel('Frequency (Hz)')
            axes[1,0].set_title('Frequency Percentiles')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Highlight key percentiles
            for p in [1, 2.5, 97.5, 99]:
                if p in percentiles:
                    idx = percentiles.index(p)
                    axes[1,0].axvline(p, color='red', linestyle='--', alpha=0.7)
                    axes[1,0].text(p, max(centroid_percentiles + rolloff_percentiles) * 0.9, 
                                  f'{p}%', ha='center', fontweight='bold')
        
        # Filter cutoff visualization
        if self.filter_params:
            filter_info = []
            for filter_name, params in self.filter_params.items():
                if filter_name != 'raw_percentiles':
                    filter_info.append([filter_name.replace('_', ' ').title(), 
                                      f"{params['low_cutoff']:.1f}", 
                                      f"{params['high_cutoff']:.1f}",
                                      params['description']])
            
            if filter_info:
                axes[1,1].axis('tight')
                axes[1,1].axis('off')
                table = axes[1,1].table(cellText=filter_info, 
                                       colLabels=['Filter Type', 'Low (Hz)', 'High (Hz)', 'Description'],
                                       cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                axes[1,1].set_title('Calculated Filter Parameters')
        
        plt.suptitle('Frequency Profile Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_analysis/frequency_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_signal_statistics(self):
        """Plot signal statistics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pd_std = self.analysis_results['PD']['std_devs']
        hc_std = self.analysis_results['HC']['std_devs']
        
        # Standard deviation comparison
        axes[0,0].hist(pd_std, bins=10, alpha=0.7, label='PD', color='red', density=True)
        axes[0,0].hist(hc_std, bins=10, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,0].set_xlabel('Signal Standard Deviation')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Signal Variability Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0,1].boxplot([pd_std, hc_std], labels=['PD', 'HC'])
        axes[0,1].set_ylabel('Signal Standard Deviation')
        axes[0,1].set_title('Signal Variability Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Correlation between silence and std dev
        all_silence = self.analysis_results['PD']['silence_ratios'] + self.analysis_results['HC']['silence_ratios']
        all_std = pd_std + hc_std
        
        axes[1,0].scatter(all_silence, all_std, alpha=0.6)
        axes[1,0].set_xlabel('Silence Ratio')
        axes[1,0].set_ylabel('Signal Standard Deviation')
        axes[1,0].set_title('Silence vs Signal Variability')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(all_silence, all_std)[0,1]
        axes[1,0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                      transform=axes[1,0].transAxes, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Summary statistics
        stats_data = [
            ['Metric', 'PD Mean', 'PD Std', 'HC Mean', 'HC Std'],
            ['Signal Std Dev', f'{np.mean(pd_std):.4f}', f'{np.std(pd_std):.4f}', 
             f'{np.mean(hc_std):.4f}', f'{np.std(hc_std):.4f}'],
            ['Silence Ratio', f'{np.mean(self.analysis_results["PD"]["silence_ratios"]):.3f}', 
             f'{np.std(self.analysis_results["PD"]["silence_ratios"]):.3f}',
             f'{np.mean(self.analysis_results["HC"]["silence_ratios"]):.3f}', 
             f'{np.std(self.analysis_results["HC"]["silence_ratios"]):.3f}']
        ]
        
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        table = axes[1,1].table(cellText=stats_data[1:], colLabels=stats_data[0], 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1,1].set_title('Summary Statistics')
        
        plt.suptitle('Signal Statistics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_analysis/signal_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_below_80hz_analysis(self):
        """Plot analysis of signal below 80 Hz"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        pd_below_80hz = self.analysis_results['PD']['below_80hz_ratios']
        hc_below_80hz = self.analysis_results['HC']['below_80hz_ratios']
        
        # Below 80 Hz ratio distribution
        axes[0,0].hist(pd_below_80hz, bins=10, alpha=0.7, label='PD', color='red', density=True)
        axes[0,0].hist(hc_below_80hz, bins=10, alpha=0.7, label='HC', color='blue', density=True)
        axes[0,0].set_xlabel('Signal Energy Below 80 Hz (ratio)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Low Frequency Content Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0,1].boxplot([pd_below_80hz, hc_below_80hz], labels=['PD', 'HC'])
        axes[0,1].set_ylabel('Signal Energy Below 80 Hz (ratio)')
        axes[0,1].set_title('Low Frequency Content Comparison')
        axes[0,1].grid(True, alpha=0.3)
        
        # Filter effect simulation
        filter_types = ['Original', 'High-pass 80Hz']
        original_ratios = pd_below_80hz + hc_below_80hz
        filtered_ratios = [0] * len(original_ratios)  # After high-pass filtering
        
        x = np.arange(len(filter_types))
        width = 0.35
        
        axes[1,0].bar(x - width/2, [np.mean(original_ratios), np.mean(filtered_ratios)], 
                     width, label='Mean', alpha=0.7)
        axes[1,0].bar(x + width/2, [np.std(original_ratios), np.std(filtered_ratios)], 
                     width, label='Std Dev', alpha=0.7)
        axes[1,0].set_xlabel('Processing Stage')
        axes[1,0].set_ylabel('Signal Below 80 Hz Ratio')
        axes[1,0].set_title('Filter Effect on Low Frequencies')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(filter_types)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Statistical significance
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(pd_below_80hz, hc_below_80hz)
        
        axes[1,1].text(0.1, 0.8, 'Low Frequency Analysis:', fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.7, f'PD below 80Hz: {np.mean(pd_below_80hz):.3f} ± {np.std(pd_below_80hz):.3f}', 
                      fontsize=10, transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.6, f'HC below 80Hz: {np.mean(hc_below_80hz):.3f} ± {np.std(hc_below_80hz):.3f}', 
                      fontsize=10, transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.5, f'Statistical test p-value: {p_value:.3f}', fontsize=10, transform=axes[1,1].transAxes)
        
        filter_impact = f"High-pass filtering at 80 Hz will remove\n{np.mean(original_ratios)*100:.1f}% of signal energy on average"
        axes[1,1].text(0.1, 0.3, filter_impact, fontsize=10, fontweight='bold', 
                      transform=axes[1,1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1,1].set_title('Filter Impact Assessment')
        axes[1,1].axis('off')
        
        plt.suptitle('Below 80 Hz Frequency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_analysis/below_80hz_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print(f"\n{'='*80}")
        print("PREPROCESSING ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        pd_data = self.analysis_results['PD']
        hc_data = self.analysis_results['HC']
        
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   PD files: {len(pd_data['files'])}")
        print(f"   HC files: {len(hc_data['files'])}")
        print(f"   Total files: {len(pd_data['files']) + len(hc_data['files'])}")
        
        print(f"\n2. AUDIO LENGTH ANALYSIS:")
        print(f"   PD duration: {np.mean(pd_data['durations']):.2f} ± {np.std(pd_data['durations']):.2f} seconds")
        print(f"   HC duration: {np.mean(hc_data['durations']):.2f} ± {np.std(hc_data['durations']):.2f} seconds")
        print(f"   Duration variability: {'High' if np.std(pd_data['durations'] + hc_data['durations']) > 5 else 'Low'}")
        
        print(f"\n3. SILENCE ANALYSIS:")
        print(f"   PD silence ratio: {np.mean(pd_data['silence_ratios']):.3f} ± {np.std(pd_data['silence_ratios']):.3f}")
        print(f"   HC silence ratio: {np.mean(hc_data['silence_ratios']):.3f} ± {np.std(hc_data['silence_ratios']):.3f}")
        
        from scipy.stats import ttest_ind
        _, p_value = ttest_ind(pd_data['silence_ratios'], hc_data['silence_ratios'])
        print(f"   Silence difference significance: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.3f})")
        
        print(f"\n4. FREQUENCY PROFILE:")
        if self.filter_params:
            print(f"   Recommended filters calculated:")
            for filter_name, params in self.filter_params.items():
                if filter_name != 'raw_percentiles':
                    print(f"   - {params['description']}")
        
        print(f"\n5. LOW FREQUENCY CONTENT (<80 Hz):")
        print(f"   PD below 80Hz: {np.mean(pd_data['below_80hz_ratios']):.3f} ± {np.std(pd_data['below_80hz_ratios']):.3f}")
        print(f"   HC below 80Hz: {np.mean(hc_data['below_80hz_ratios']):.3f} ± {np.std(hc_data['below_80hz_ratios']):.3f}")
        
        avg_below_80hz = np.mean(pd_data['below_80hz_ratios'] + hc_data['below_80hz_ratios'])
        print(f"   Filter impact: High-pass at 80Hz removes {avg_below_80hz*100:.1f}% of signal energy")
        
        print(f"\n6. PREPROCESSING RECOMMENDATIONS:")
        avg_silence = np.mean(pd_data['silence_ratios'] + hc_data['silence_ratios'])
        
        if avg_silence > 0.3:
            print("   - Consider silence removal (high silence ratio detected)")
        else:
            print("   - Minimal silence removal recommended (low silence ratio)")
            
        if avg_below_80hz > 0.1:
            print("   - High-pass filtering strongly recommended (significant low-freq noise)")
        else:
            print("   - High-pass filtering optional (minimal low-freq content)")
            
        print("   - NO amplitude normalization (preserves original signal characteristics)")
        print("   - NO length standardization (preserves natural duration variability)")
        
        duration_std = np.std(pd_data['durations'] + hc_data['durations'])
        if duration_std > 5:
            print("   - High duration variability detected - good for natural analysis")
        
        return self.analysis_results
    
    def preprocess_with_analysis_based_settings(self, data_dir="data", output_dir="analysis_based_preprocessed_data"):
        """
        Preprocess audio files using analysis-based settings
        NO amplitude normalization, NO length standardization
        """
        if self.filter_params is None:
            print("Error: Must run dataset analysis first!")
            return
        
        print(f"\n{'='*80}")
        print("ANALYSIS-BASED AUDIO PREPROCESSING")
        print(f"{'='*80}")
        
        # Create output directories
        for cohort in ["PD", "HC"]:
            os.makedirs(os.path.join(output_dir, cohort), exist_ok=True)
        
        # Use conservative filter (2.5th-97.5th percentiles)
        filter_params = self.filter_params['percentile_2_5_97_5']
        low_cutoff = filter_params['low_cutoff']
        high_cutoff = filter_params['high_cutoff']
        
        print(f"Filter settings: {filter_params['description']}")
        print("Processing settings:")
        print("- Frequency filtering: High-pass + Band-pass")
        print("- Silence handling: Based on analysis results")
        print("- NO amplitude normalization")
        print("- NO length standardization")
        print()
        
        processed_count = {"PD": 0, "HC": 0}
        processing_stats = {"PD": [], "HC": []}
        
        for cohort in ["PD", "HC"]:
            input_dir = os.path.join(data_dir, cohort)
            output_cohort_dir = os.path.join(output_dir, cohort)
            
            if not os.path.exists(input_dir):
                continue
                
            audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
            print(f"Processing {len(audio_files)} {cohort} files...")
            
            # Determine silence removal threshold based on analysis
            cohort_silence_ratios = self.analysis_results[cohort]['silence_ratios']
            silence_threshold = np.percentile(cohort_silence_ratios, 25)  # Remove only excessive silence
            
            for i, filename in enumerate(audio_files):
                try:
                    input_path = os.path.join(input_dir, filename)
                    output_filename = f"preprocessed_{i+1:04d}.wav"
                    output_path = os.path.join(output_cohort_dir, output_filename)
                    
                    # Load audio
                    audio, sr = librosa.load(input_path, sr=self.target_sr)
                    original_length = len(audio)
                    
                    # Apply frequency filtering
                    processed_audio = self._apply_frequency_filtering(audio, sr, low_cutoff, high_cutoff)
                    
                    # Selective silence removal (only if excessive)
                    current_silence_ratio = self._calculate_silence_ratio(processed_audio, sr)
                    if current_silence_ratio > silence_threshold * 1.5:  # Only if significantly above threshold
                        processed_audio = self._remove_excessive_silence(processed_audio, sr, silence_threshold)
                    
                    final_length = len(processed_audio)
                    
                    # NO amplitude normalization - preserve original dynamics
                    # NO length standardization - preserve natural duration
                    
                    # Save processed audio
                    sf.write(output_path, processed_audio, sr)
                    processed_count[cohort] += 1
                    
                    # Track processing statistics
                    processing_stats[cohort].append({
                        'original_length': original_length / sr,
                        'final_length': final_length / sr,
                        'length_change_ratio': final_length / original_length,
                        'original_silence_ratio': current_silence_ratio
                    })
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Processed: {i+1}/{len(audio_files)} files")
                        
                except Exception as e:
                    print(f"  Error with {filename}: {e}")
        
        # Print processing summary
        self._print_processing_summary(processed_count, processing_stats, filter_params, output_dir)
        
        return processed_count, processing_stats
    
    def _apply_frequency_filtering(self, audio, sr, low_cutoff, high_cutoff):
        """Apply high-pass followed by band-pass filtering"""
        try:
            # Step 1: High-pass filter
            b_hp, a_hp = butter(3, low_cutoff / (sr / 2), btype='high')
            audio_filtered = filtfilt(b_hp, a_hp, audio)
            
            # Step 2: Band-pass filter (if high cutoff is below Nyquist)
            if high_cutoff < sr / 2:
                b_bp, a_bp = butter(3, [low_cutoff, high_cutoff], btype='band', fs=sr)
                audio_filtered = filtfilt(b_bp, a_bp, audio_filtered)
            
            return audio_filtered
            
        except Exception as e:
            print(f"Warning: Filtering failed ({e}). Using original audio.")
            return audio
    
    def _remove_excessive_silence(self, audio, sr, threshold):
        """Remove only excessive silence while preserving natural pauses"""
        frame_size = int(0.025 * sr)  # 25ms
        hop_size = int(0.01 * sr)     # 10ms
        
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)
        
        if len(energies) == 0:
            return audio
        
        energies = np.array(energies)
        max_energy = np.max(energies)
        
        if max_energy > 0:
            energies = energies / max_energy
        
        # Keep frames above threshold
        voice_frames = energies > threshold
        
        # Apply some smoothing to avoid choppy removal
        kernel = np.ones(5) / 5  # 5-frame smoothing
        voice_frames_smooth = np.convolve(voice_frames.astype(float), kernel, mode='same') > 0.3
        
        # Extract voice segments
        voice_audio = []
        for i, is_voice in enumerate(voice_frames_smooth):
            if is_voice:
                start = i * hop_size
                end = min(start + frame_size, len(audio))
                voice_audio.extend(audio[start:end])
        
        return np.array(voice_audio) if voice_audio else audio
    
    def _print_processing_summary(self, processed_count, processing_stats, filter_params, output_dir):
        """Print detailed processing summary"""
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        
        print(f"Filter applied: {filter_params['description']}")
        print(f"Files processed:")
        print(f"  PD: {processed_count['PD']}")
        print(f"  HC: {processed_count['HC']}")
        print(f"  Total: {sum(processed_count.values())}")
        
        print(f"\nProcessing statistics:")
        for cohort in ['PD', 'HC']:
            if processing_stats[cohort]:
                stats = processing_stats[cohort]
                avg_original_length = np.mean([s['original_length'] for s in stats])
                avg_final_length = np.mean([s['final_length'] for s in stats])
                avg_length_change = np.mean([s['length_change_ratio'] for s in stats])
                
                print(f"  {cohort}:")
                print(f"    Average original length: {avg_original_length:.2f} seconds")
                print(f"    Average final length: {avg_final_length:.2f} seconds")
                print(f"    Average length retention: {avg_length_change:.3f}")
        
        print(f"\nKey preprocessing principles followed:")
        print(f"  ✓ Frequency filtering based on dataset percentiles")
        print(f"  ✓ Minimal silence removal (preserve natural pauses)")
        print(f"  ✓ NO amplitude normalization (preserve dynamics)")
        print(f"  ✓ NO length standardization (preserve duration variability)")
        
        print(f"\nOutput directory: {os.path.abspath(output_dir)}")


def main():
    """Main function for advanced audio preprocessing"""
    print("ADVANCED AUDIO PREPROCESSING FOR PARKINSON'S DISEASE ANALYSIS")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = AdvancedAudioPreprocessor(target_sr=16000)
    
    # Step 1: Comprehensive dataset analysis
    print("\nSTEP 1: Dataset Analysis")
    analysis_results = preprocessor.analyze_dataset_for_preprocessing("data")
    
    if not analysis_results:
        print("Analysis failed. Please check your data directory.")
        return
    
    # Step 2: Analysis-based preprocessing
    print("\nSTEP 2: Analysis-Based Preprocessing")
    processed_count, processing_stats = preprocessor.preprocess_with_analysis_based_settings(
        "data", "analysis_based_preprocessed_data"
    )
    
    print(f"\n{'='*80}")
    print("ALL PREPROCESSING TASKS COMPLETED!")
    print(f"{'='*80}")
    print("\nGenerated outputs:")
    print("1. preprocessing_analysis/ - Comprehensive analysis visualizations")
    print("2. analysis_based_preprocessed_data/ - Preprocessed audio files")
    print("\nKey achievements:")
    print("✓ Frequency filtering with percentile-based cutoffs")
    print("✓ Silence analysis and selective removal")
    print("✓ Audio length distribution analysis")
    print("✓ Signal comparison and low-frequency assessment")
    print("✓ Preserved original amplitude and length characteristics")


if __name__ == "__main__":
    main()

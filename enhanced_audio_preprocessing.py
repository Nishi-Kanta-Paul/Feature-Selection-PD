import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class EnhancedAudioPreprocessor:
    """
    Enhanced Audio Preprocessor with Advanced Frequency Filtering
    
    Features:
    - Dataset-wide frequency analysis
    - Percentile-based filter cutoff determination
    - High-pass and band-pass filtering
    - Comprehensive preprocessing pipeline
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.frequency_stats = None
        self.filter_params = None
        
    def analyze_dataset_frequencies(self, data_dir="data", sample_size=None):
        """
        Analyze frequency content across the entire dataset to determine optimal filter parameters
        """
        print("Analyzing dataset frequency content...")
        print("=" * 50)
        
        all_frequency_data = []
        all_spectral_centroids = []
        all_spectral_rolloffs = []
        all_dominant_freqs = []
        
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
            
            print(f"\nAnalyzing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                try:
                    file_path = os.path.join(cohort_dir, filename)
                    
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=self.target_sr)
                    
                    # Skip very short files
                    if len(audio) < self.target_sr * 0.5:  # Less than 0.5 seconds
                        continue
                    
                    # Frequency analysis
                    freq_data = self._analyze_audio_frequencies(audio, sr)
                    all_frequency_data.append(freq_data)
                    
                    # Spectral features
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
                    
                    all_spectral_centroids.append(spectral_centroid)
                    all_spectral_rolloffs.append(spectral_rolloff)
                    
                    # Dominant frequency
                    dominant_freq = self._find_dominant_frequency(audio, sr)
                    all_dominant_freqs.append(dominant_freq)
                    
                    processed_files += 1
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Processed: {i+1}/{len(audio_files)} files")
                        
                except Exception as e:
                    print(f"  Error analyzing {filename}: {e}")
        
        # Compile frequency statistics
        self.frequency_stats = {
            'spectral_centroids': np.array(all_spectral_centroids),
            'spectral_rolloffs': np.array(all_spectral_rolloffs),
            'dominant_frequencies': np.array(all_dominant_freqs),
            'frequency_distributions': all_frequency_data,
            'total_files_analyzed': processed_files
        }
        
        # Calculate percentile-based filter parameters
        self._calculate_filter_parameters()
        
        print(f"\n{'='*50}")
        print("FREQUENCY ANALYSIS COMPLETE!")
        print(f"{'='*50}")
        print(f"Total files analyzed: {processed_files}")
        print(f"Filter parameters calculated using dataset percentiles")
        
        return self.frequency_stats
    
    def _analyze_audio_frequencies(self, audio, sr):
        """Analyze frequency content of a single audio file"""
        # Calculate power spectral density
        freqs, psd = welch(audio, fs=sr, nperseg=min(2048, len(audio)//4))
        
        # Find frequency ranges with significant energy
        energy_threshold = np.percentile(psd, 75)  # Top 25% energy
        significant_freqs = freqs[psd > energy_threshold]
        
        return {
            'freqs': freqs,
            'psd': psd,
            'significant_freqs': significant_freqs,
            'freq_range': (np.min(significant_freqs), np.max(significant_freqs)) if len(significant_freqs) > 0 else (0, sr/2)
        }
    
    def _find_dominant_frequency(self, audio, sr):
        """Find the dominant frequency in the audio signal"""
        # FFT analysis
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]
        
        # Find dominant frequency
        if len(pos_magnitude) > 0:
            dominant_idx = np.argmax(pos_magnitude)
            return pos_freqs[dominant_idx]
        return 0
    
    def _calculate_filter_parameters(self):
        """Calculate filter parameters based on dataset percentiles"""
        if self.frequency_stats is None:
            raise ValueError("Must run frequency analysis first")
        
        centroids = self.frequency_stats['spectral_centroids']
        rolloffs = self.frequency_stats['spectral_rolloffs']
        dominants = self.frequency_stats['dominant_frequencies']
        
        # Remove outliers and invalid values
        centroids = centroids[centroids > 0]
        rolloffs = rolloffs[rolloffs > 0]
        dominants = dominants[dominants > 0]
        
        # Calculate percentiles for different filtering strategies
        percentiles = [1, 2.5, 5, 10, 90, 95, 97.5, 99]
        
        centroid_percentiles = np.percentile(centroids, percentiles)
        rolloff_percentiles = np.percentile(rolloffs, percentiles)
        dominant_percentiles = np.percentile(dominants, percentiles)
        
        # Strategy 1: Use 1st-99th percentiles for broad filtering
        broad_low_cutoff = max(50, np.min([centroid_percentiles[0], dominant_percentiles[0]]))  # Minimum 50 Hz
        broad_high_cutoff = min(self.target_sr // 2, rolloff_percentiles[-1])  # Maximum Nyquist
        
        # Strategy 2: Use 2.5th-97.5th percentiles for conservative filtering
        conservative_low_cutoff = max(80, np.min([centroid_percentiles[1], dominant_percentiles[1]]))  # Minimum 80 Hz
        conservative_high_cutoff = min(self.target_sr // 2, rolloff_percentiles[-2])  # Maximum Nyquist
        
        # Voice-specific filtering (typical voice range: 80-8000 Hz)
        voice_low_cutoff = max(80, centroid_percentiles[2])  # 5th percentile, min 80 Hz
        voice_high_cutoff = min(8000, rolloff_percentiles[-3])  # 95th percentile, max 8000 Hz
        
        self.filter_params = {
            'broad_filter': {
                'low_cutoff': broad_low_cutoff,
                'high_cutoff': broad_high_cutoff,
                'description': f'1st-99th percentile range: {broad_low_cutoff:.1f}-{broad_high_cutoff:.1f} Hz'
            },
            'conservative_filter': {
                'low_cutoff': conservative_low_cutoff,
                'high_cutoff': conservative_high_cutoff,
                'description': f'2.5th-97.5th percentile range: {conservative_low_cutoff:.1f}-{conservative_high_cutoff:.1f} Hz'
            },
            'voice_optimized_filter': {
                'low_cutoff': voice_low_cutoff,
                'high_cutoff': voice_high_cutoff,
                'description': f'Voice-optimized range: {voice_low_cutoff:.1f}-{voice_high_cutoff:.1f} Hz'
            },
            'percentiles': {
                'spectral_centroids': dict(zip(percentiles, centroid_percentiles)),
                'spectral_rolloffs': dict(zip(percentiles, rolloff_percentiles)),
                'dominant_frequencies': dict(zip(percentiles, dominant_percentiles))
            }
        }
        
        # Print filter parameters
        print(f"\nCalculated Filter Parameters:")
        print(f"{'='*50}")
        for filter_type, params in self.filter_params.items():
            if filter_type != 'percentiles':
                print(f"{filter_type.replace('_', ' ').title()}: {params['description']}")
        
        return self.filter_params
    
    def create_frequency_analysis_visualizations(self, output_dir="frequency_analysis"):
        """Create visualizations of frequency analysis results"""
        if self.frequency_stats is None or self.filter_params is None:
            print("Must run frequency analysis first")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Spectral statistics distribution
        self._plot_spectral_distributions(output_dir)
        
        # 2. Filter frequency response
        self._plot_filter_responses(output_dir)
        
        # 3. Dataset frequency content overview
        self._plot_frequency_content_overview(output_dir)
        
        # 4. Percentile analysis
        self._plot_percentile_analysis(output_dir)
        
        print(f"Frequency analysis visualizations saved to: {os.path.abspath(output_dir)}")
    
    def _plot_spectral_distributions(self, output_dir):
        """Plot distributions of spectral features"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        stats_data = self.frequency_stats
        
        # Spectral centroids
        axes[0,0].hist(stats_data['spectral_centroids'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_title('Distribution of Spectral Centroids')
        axes[0,0].set_xlabel('Frequency (Hz)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].grid(True, alpha=0.3)
        
        # Spectral rolloffs
        axes[0,1].hist(stats_data['spectral_rolloffs'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_title('Distribution of Spectral Rolloffs')
        axes[0,1].set_xlabel('Frequency (Hz)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].grid(True, alpha=0.3)
        
        # Dominant frequencies
        axes[1,0].hist(stats_data['dominant_frequencies'], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1,0].set_title('Distribution of Dominant Frequencies')
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Count')
        axes[1,0].grid(True, alpha=0.3)
        
        # Box plot comparison
        box_data = [
            stats_data['spectral_centroids'],
            stats_data['spectral_rolloffs'],
            stats_data['dominant_frequencies']
        ]
        axes[1,1].boxplot(box_data, labels=['Centroids', 'Rolloffs', 'Dominant'])
        axes[1,1].set_title('Spectral Features Comparison')
        axes[1,1].set_ylabel('Frequency (Hz)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Dataset Spectral Feature Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectral_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_filter_responses(self, output_dir):
        """Plot frequency responses of designed filters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Frequency range for plotting
        freqs = np.logspace(1, np.log10(self.target_sr/2), 1000)  # 10 Hz to Nyquist
        
        filter_types = ['broad_filter', 'conservative_filter', 'voice_optimized_filter']
        colors = ['blue', 'green', 'red']
        
        for i, (filter_type, color) in enumerate(zip(filter_types, colors)):
            params = self.filter_params[filter_type]
            
            # High-pass filter response
            if i < 3:  # First 3 subplots
                row, col = i // 2, i % 2
                ax = axes[row, col] if i < 2 else axes[1, 0]
                
                # High-pass filter
                b_hp, a_hp = butter(3, params['low_cutoff'] / (self.target_sr / 2), btype='high')
                w_hp, h_hp = self._get_filter_response(b_hp, a_hp, freqs, self.target_sr)
                
                # Band-pass filter
                b_bp, a_bp = butter(3, [params['low_cutoff'], params['high_cutoff']], 
                                  btype='band', fs=self.target_sr)
                w_bp, h_bp = self._get_filter_response(b_bp, a_bp, freqs, self.target_sr)
                
                ax.semilogx(w_hp, 20 * np.log10(np.abs(h_hp)), color=color, linestyle='--', 
                           label=f'High-pass ({params["low_cutoff"]:.1f} Hz)', linewidth=2)
                ax.semilogx(w_bp, 20 * np.log10(np.abs(h_bp)), color=color, 
                           label=f'Band-pass ({params["low_cutoff"]:.1f}-{params["high_cutoff"]:.1f} Hz)', linewidth=2)
                
                ax.set_title(f'{filter_type.replace("_", " ").title()} Response')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude (dB)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylim([-60, 5])
        
        # Combined comparison in the last subplot
        ax = axes[1, 1]
        for filter_type, color in zip(filter_types, colors):
            params = self.filter_params[filter_type]
            b_bp, a_bp = butter(3, [params['low_cutoff'], params['high_cutoff']], 
                              btype='band', fs=self.target_sr)
            w_bp, h_bp = self._get_filter_response(b_bp, a_bp, freqs, self.target_sr)
            
            ax.semilogx(w_bp, 20 * np.log10(np.abs(h_bp)), color=color, linewidth=2,
                       label=f'{filter_type.replace("_", " ").title()}')
        
        ax.set_title('Filter Comparison')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([-60, 5])
        
        plt.suptitle('Filter Frequency Responses', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'filter_responses.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_filter_response(self, b, a, freqs, fs):
        """Calculate filter frequency response"""
        from scipy.signal import freqs as scipy_freqs
        w, h = scipy_freqs(b, a, worN=freqs, fs=fs)
        return w, h
    
    def _plot_frequency_content_overview(self, output_dir):
        """Plot overview of dataset frequency content"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Aggregate frequency data
        all_freqs = []
        all_psds = []
        
        for freq_data in self.frequency_stats['frequency_distributions'][:50]:  # Sample first 50
            all_freqs.extend(freq_data['freqs'])
            all_psds.extend(freq_data['psd'])
        
        # 2D histogram of frequency content
        axes[0].hist2d(all_freqs, all_psds, bins=50, cmap='viridis')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power Spectral Density')
        axes[0].set_title('Dataset Frequency Content Distribution')
        axes[0].set_xlim([0, self.target_sr/2])
        
        # Percentile lines
        percentiles = self.filter_params['percentiles']
        centroid_percentiles = percentiles['spectral_centroids']
        
        for p, freq in centroid_percentiles.items():
            if p in [1, 2.5, 97.5, 99]:
                axes[0].axvline(freq, color='red', linestyle='--', alpha=0.7, 
                               label=f'{p}th percentile' if p in [1, 99] else '')
        
        axes[0].legend()
        
        # Average power spectrum
        avg_freqs = np.linspace(0, self.target_sr/2, 1000)
        avg_psd = np.zeros_like(avg_freqs)
        
        for freq_data in self.frequency_stats['frequency_distributions'][:20]:  # Sample first 20
            interp_psd = np.interp(avg_freqs, freq_data['freqs'], freq_data['psd'])
            avg_psd += interp_psd
        
        avg_psd /= len(self.frequency_stats['frequency_distributions'][:20])
        
        axes[1].semilogx(avg_freqs, 10 * np.log10(avg_psd + 1e-10))
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power (dB)')
        axes[1].set_title('Average Power Spectrum')
        axes[1].grid(True, alpha=0.3)
        
        # Add filter boundaries
        for filter_type, color in zip(['conservative_filter', 'voice_optimized_filter'], ['green', 'red']):
            params = self.filter_params[filter_type]
            axes[1].axvline(params['low_cutoff'], color=color, linestyle='--', alpha=0.7,
                           label=f'{filter_type.replace("_", " ").title()} low')
            axes[1].axvline(params['high_cutoff'], color=color, linestyle=':', alpha=0.7,
                           label=f'{filter_type.replace("_", " ").title()} high')
        
        axes[1].legend()
        
        plt.suptitle('Dataset Frequency Analysis Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_content_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_percentile_analysis(self, output_dir):
        """Plot detailed percentile analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        percentiles = self.filter_params['percentiles']
        percentile_values = list(percentiles['spectral_centroids'].keys())
        
        features = ['spectral_centroids', 'spectral_rolloffs', 'dominant_frequencies']
        colors = ['blue', 'green', 'red']
        titles = ['Spectral Centroids', 'Spectral Rolloffs', 'Dominant Frequencies']
        
        for i, (feature, color, title) in enumerate(zip(features, colors, titles)):
            values = [percentiles[feature][p] for p in percentile_values]
            
            axes[i].plot(percentile_values, values, 'o-', color=color, linewidth=2, markersize=8)
            axes[i].set_xlabel('Percentile')
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
            
            # Highlight key percentiles
            for p in [1, 2.5, 97.5, 99]:
                if p in percentiles[feature]:
                    idx = percentile_values.index(p)
                    axes[i].plot(p, values[idx], 'ro', markersize=10, alpha=0.7)
                    axes[i].text(p, values[idx], f'{values[idx]:.0f} Hz', 
                                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Percentile Analysis for Filter Design', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'percentile_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_dataset_with_enhanced_filtering(self, data_dir="data", output_dir="enhanced_preprocessed_data", 
                                              filter_type="conservative_filter"):
        """
        Process entire dataset with enhanced frequency filtering
        """
        if self.filter_params is None:
            print("Must run frequency analysis first to determine filter parameters")
            return
        
        # Create output directories
        for cohort in ["PD", "HC"]:
            os.makedirs(os.path.join(output_dir, cohort), exist_ok=True)
        
        # Get filter parameters
        params = self.filter_params[filter_type]
        
        print(f"Starting enhanced audio preprocessing with {filter_type}...")
        print(f"Filter: {params['description']}")
        print("=" * 60)
        
        processed_count = {"PD": 0, "HC": 0}
        
        for cohort in ["PD", "HC"]:
            input_dir = os.path.join(data_dir, cohort)
            output_cohort_dir = os.path.join(output_dir, cohort)
            
            if not os.path.exists(input_dir):
                continue
                
            audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
            print(f"\nProcessing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                try:
                    input_path = os.path.join(input_dir, filename)
                    output_filename = f"enhanced_{i+1:04d}.wav"
                    output_path = os.path.join(output_cohort_dir, output_filename)
                    
                    # Load audio
                    audio, sr = librosa.load(input_path, sr=self.target_sr)
                    
                    # Enhanced preprocessing with percentile-based filtering
                    processed_audio = self._preprocess_with_enhanced_filtering(
                        audio, sr, params['low_cutoff'], params['high_cutoff'])
                    
                    # Save
                    sf.write(output_path, processed_audio, sr)
                    processed_count[cohort] += 1
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Processed: {i+1}/{len(audio_files)} files")
                        
                except Exception as e:
                    print(f"  Error with {filename}: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print("ENHANCED PREPROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Filter type: {filter_type}")
        print(f"Filter range: {params['description']}")
        print(f"PD files: {processed_count['PD']}")
        print(f"HC files: {processed_count['HC']}")
        print(f"Total: {sum(processed_count.values())}")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        return processed_count
    
    def _preprocess_with_enhanced_filtering(self, audio, sr, low_cutoff, high_cutoff, 
                                          silence_threshold=0.02):
        """
        Enhanced preprocessing with percentile-based frequency filtering
        """
        # Step 1: High-pass filter (removes low-frequency noise)
        b_hp, a_hp = butter(3, low_cutoff / (sr / 2), btype='high')
        audio = filtfilt(b_hp, a_hp, audio)
        
        # Step 2: Band-pass filter (focuses on voice frequency range)
        if high_cutoff < sr / 2:  # Only apply if high cutoff is below Nyquist
            b_bp, a_bp = butter(3, [low_cutoff, high_cutoff], btype='band', fs=sr)
            audio = filtfilt(b_bp, a_bp, audio)
        
        # Step 3: Remove silence
        audio = self._remove_silence_enhanced(audio, sr, silence_threshold)
        
        # Step 4: Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Step 5: Ensure minimum length
        min_length = int(0.5 * sr)  # 0.5 seconds
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))
        
        return audio
    
    def _remove_silence_enhanced(self, audio, sr, threshold=0.02):
        """
        Enhanced silence removal with voice activity detection
        """
        # Frame parameters
        frame_size = int(0.025 * sr)  # 25ms
        hop_size = int(0.01 * sr)     # 10ms
        
        # Calculate multiple energy measures
        energies = []
        spectral_centroids = []
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            
            # Energy measures
            energy = np.mean(frame ** 2)
            energies.append(energy)
            
            # Spectral centroid for voice activity
            if len(frame) > 0:
                fft = np.fft.fft(frame)
                magnitude = np.abs(fft)
                freqs = np.fft.fftfreq(len(frame), 1/sr)
                pos_freqs = freqs[:len(freqs)//2]
                pos_magnitude = magnitude[:len(magnitude)//2]
                
                if np.sum(pos_magnitude) > 0:
                    centroid = np.sum(pos_freqs * pos_magnitude) / np.sum(pos_magnitude)
                else:
                    centroid = 0
                spectral_centroids.append(centroid)
            else:
                spectral_centroids.append(0)
        
        if not energies:
            return audio
        
        energies = np.array(energies)
        spectral_centroids = np.array(spectral_centroids)
        
        # Normalize energies
        max_energy = np.max(energies)
        if max_energy > 0:
            energies = energies / max_energy
        
        # Voice activity detection using both energy and spectral centroid
        voice_frames = (energies > threshold) & (spectral_centroids > 80) & (spectral_centroids < 8000)
        
        # Apply some smoothing to avoid choppy removal
        kernel = np.ones(3) / 3  # Simple moving average
        voice_frames_smooth = np.convolve(voice_frames.astype(float), kernel, mode='same') > 0.5
        
        # Extract voice segments
        voice_audio = []
        for i, is_voice in enumerate(voice_frames_smooth):
            if is_voice:
                start = i * hop_size
                end = min(start + frame_size, len(audio))
                voice_audio.extend(audio[start:end])
        
        return np.array(voice_audio) if voice_audio else audio


def main():
    """Main function to run enhanced audio preprocessing"""
    # Initialize preprocessor
    preprocessor = EnhancedAudioPreprocessor(target_sr=16000)
    
    # Step 1: Analyze dataset frequencies
    print("Step 1: Analyzing dataset frequency content...")
    frequency_stats = preprocessor.analyze_dataset_frequencies(data_dir="data", sample_size=10)
    
    if frequency_stats is None:
        print("Failed to analyze frequencies. Exiting.")
        return
    
    # Step 2: Create frequency analysis visualizations
    print("\nStep 2: Creating frequency analysis visualizations...")
    preprocessor.create_frequency_analysis_visualizations("frequency_analysis")
    
    # Step 3: Process dataset with different filter types
    filter_types = ["conservative_filter", "voice_optimized_filter", "broad_filter"]
    
    for filter_type in filter_types:
        print(f"\nStep 3.{filter_types.index(filter_type)+1}: Processing with {filter_type}...")
        output_dir = f"enhanced_preprocessed_data_{filter_type}"
        processed_count = preprocessor.process_dataset_with_enhanced_filtering(
            data_dir="data", 
            output_dir=output_dir, 
            filter_type=filter_type
        )
    
    # Print final summary
    print(f"\n{'='*80}")
    print("ENHANCED AUDIO PREPROCESSING COMPLETE!")
    print(f"{'='*80}")
    print("Filter parameters determined using dataset percentiles:")
    
    for filter_type, params in preprocessor.filter_params.items():
        if filter_type != 'percentiles':
            print(f"- {filter_type.replace('_', ' ').title()}: {params['description']}")
    
    print(f"\nOutput directories created:")
    for filter_type in filter_types:
        print(f"- enhanced_preprocessed_data_{filter_type}")
    
    print(f"\nVisualizations saved to: frequency_analysis/")


if __name__ == "__main__":
    main()

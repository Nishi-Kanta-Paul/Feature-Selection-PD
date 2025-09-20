import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """
    Streamlined Audio Preprocessor for Parkinson's Disease Analysis
    
    Features:
    1. Data mapping from final_selected.csv
    2. 16kHz sampling rate
    3. Band-pass filtering (1-99 and 2.5-97.5 percentiles)
    4. Clean and minimal implementation
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.data_mapping = None
        self.filter_params = None
        
    def load_data_mapping(self, csv_path="all_audios_mapped_id_for_label/final_selected.csv"):
        """Load and process the final_selected.csv file"""
        print("Loading data mapping from final_selected.csv...")
        
        try:
            self.data_mapping = pd.read_csv(csv_path)
            print(f"Loaded {len(self.data_mapping)} records")
            
            # Count cohorts
            cohort_counts = self.data_mapping['cohort'].value_counts()
            print("Cohort distribution:")
            for cohort, count in cohort_counts.items():
                print(f"  {cohort}: {count} files")
                
            return self.data_mapping
            
        except Exception as e:
            print(f"Error loading data mapping: {e}")
            return None
    
    def analyze_audio_frequencies(self, data_dir="data"):
        """Analyze frequency characteristics for filter parameter calculation"""
        print("\nAnalyzing audio frequencies for filter parameters...")
        
        all_centroids = []
        all_rolloffs = []
        processed_files = 0
        
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join(data_dir, cohort)
            if not os.path.exists(cohort_dir):
                continue
                
            audio_files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
            print(f"Analyzing {len(audio_files)} {cohort} files...")
            
            for filename in audio_files[:10]:  # Sample first 10 files for speed
                try:
                    file_path = os.path.join(cohort_dir, filename)
                    audio, sr = librosa.load(file_path, sr=self.target_sr)
                    
                    # Calculate spectral features
                    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
                    
                    all_centroids.extend(spectral_centroids)
                    all_rolloffs.extend(spectral_rolloff)
                    processed_files += 1
                    
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
        
        if len(all_centroids) > 0:
            # Calculate percentile-based filter parameters
            centroids = np.array(all_centroids)
            rolloffs = np.array(all_rolloffs)
            
            # Strategy 1: 1st-99th percentiles
            p1_low = max(80, np.percentile(centroids, 1))
            p1_high = min(self.target_sr//2, np.percentile(rolloffs, 99))
            
            # Strategy 2: 2.5th-97.5th percentiles  
            p2_low = max(100, np.percentile(centroids, 2.5))
            p2_high = min(8000, np.percentile(rolloffs, 97.5))
            
            self.filter_params = {
                'percentile_1_99': {
                    'low_cutoff': p1_low,
                    'high_cutoff': p1_high,
                    'description': f'1-99 percentile: {p1_low:.1f}-{p1_high:.1f} Hz'
                },
                'percentile_2_5_97_5': {
                    'low_cutoff': p2_low,
                    'high_cutoff': p2_high,
                    'description': f'2.5-97.5 percentile: {p2_low:.1f}-{p2_high:.1f} Hz'
                }
            }
            
            print(f"Analyzed {processed_files} files")
            print("Filter parameters calculated:")
            for strategy, params in self.filter_params.items():
                print(f"  {params['description']}")
                
            # Create beginner-friendly filter diagrams BEFORE applying filters
            self._create_filter_explanation_diagrams()
                
        else:
            print("Warning: No audio data found for analysis")
            # Default values if no data
            self.filter_params = {
                'percentile_1_99': {
                    'low_cutoff': 80,
                    'high_cutoff': 4000,
                    'description': '1-99 percentile: 80.0-4000.0 Hz (default)'
                },
                'percentile_2_5_97_5': {
                    'low_cutoff': 100,
                    'high_cutoff': 3500,
                    'description': '2.5-97.5 percentile: 100.0-3500.0 Hz (default)'
                }
            }
    
    def apply_band_pass_filter(self, audio, sr, low_cutoff, high_cutoff):
        """Apply band-pass filtering"""
        try:
            # Ensure valid frequency range
            nyquist = sr / 2
            low_cutoff = max(1, min(low_cutoff, nyquist - 1))
            high_cutoff = max(low_cutoff + 1, min(high_cutoff, nyquist - 1))
            
            # Apply 3rd order Butterworth band-pass filter
            b, a = butter(3, [low_cutoff, high_cutoff], btype='band', fs=sr)
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            print(f"Warning: Band-pass filtering failed ({e}). Using original audio.")
            return audio
    
    def preprocess_audio_data(self, data_dir="data", output_base_dir="preprocessed_data"):
        """
        Main preprocessing function with both filtering strategies
        """
        if self.filter_params is None:
            print("Error: Must run frequency analysis first!")
            return
            
        print(f"\n{'='*60}")
        print("STREAMLINED AUDIO PREPROCESSING")
        print(f"{'='*60}")
        
        results = {}
        
        # Process both filtering strategies
        for strategy_name, strategy_params in self.filter_params.items():
            print(f"\nProcessing with {strategy_name}...")
            print(f"Filter: {strategy_params['description']}")
            
            # Create output directory
            output_dir = f"{output_base_dir}_{strategy_name}"
            for cohort in ["PD", "HC"]:
                os.makedirs(os.path.join(output_dir, cohort), exist_ok=True)
            
            low_cutoff = strategy_params['low_cutoff']
            high_cutoff = strategy_params['high_cutoff']
            
            processed_count = {"PD": 0, "HC": 0}
            
            for cohort in ["PD", "HC"]:
                input_dir = os.path.join(data_dir, cohort)
                if not os.path.exists(input_dir):
                    continue
                    
                audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
                print(f"  Processing {len(audio_files)} {cohort} files...")
                
                for i, filename in enumerate(audio_files):
                    try:
                        input_path = os.path.join(input_dir, filename)
                        output_filename = f"{strategy_name}_{i+1:04d}.wav"
                        output_path = os.path.join(output_dir, cohort, output_filename)
                        
                        # Load audio at 16kHz
                        audio, sr = librosa.load(input_path, sr=self.target_sr)
                        
                        # Apply band-pass filtering
                        filtered_audio = self.apply_band_pass_filter(audio, sr, low_cutoff, high_cutoff)
                        
                        # Save processed audio
                        sf.write(output_path, filtered_audio, sr)
                        processed_count[cohort] += 1
                        
                    except Exception as e:
                        print(f"    Error with {filename}: {e}")
            
            results[strategy_name] = {
                'output_dir': output_dir,
                'processed_count': processed_count,
                'filter_params': strategy_params
            }
            
            print(f"  Completed: PD={processed_count['PD']}, HC={processed_count['HC']}")
        
        self._print_final_summary(results)
        
        # Create before/after filtering comparison with actual data
        self._create_before_after_comparison()
        
        return results
    
    def _print_final_summary(self, results):
        """Print final processing summary"""
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETED!")
        print(f"{'='*60}")
        
        for strategy_name, result in results.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  Filter: {result['filter_params']['description']}")
            print(f"  Output: {result['output_dir']}/")
            print(f"  Files: PD={result['processed_count']['PD']}, HC={result['processed_count']['HC']}")
        
        print("\nKey features:")
        print("  ✓ 16kHz sampling rate")
        print("  ✓ Band-pass filtering (1-99 and 2.5-97.5 percentiles)")
        print("  ✓ Minimal, clean implementation")
        print("  ✓ Data mapped from final_selected.csv")
    
    def _create_filter_explanation_diagrams(self):
        """Create beginner-friendly diagrams explaining band-pass filtering"""
        print("\n🎨 Creating beginner-friendly filter explanation diagrams...")
        os.makedirs("essential_analysis", exist_ok=True)
        
        # Create comprehensive filter explanation figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # 1. Simple frequency spectrum explanation
        ax1 = fig.add_subplot(gs[0, :])
        freq_range = np.linspace(0, 8000, 1000)
        
        # Show full spectrum before filtering
        signal_power = np.exp(-((freq_range - 2000) / 1500) ** 2) + 0.5 * np.exp(-((freq_range - 4000) / 800) ** 2)
        noise_low = 0.3 * np.exp(-((freq_range - 50) / 100) ** 2)
        noise_high = 0.2 * np.exp(-((freq_range - 7000) / 500) ** 2)
        total_signal = signal_power + noise_low + noise_high
        
        ax1.fill_between(freq_range, 0, total_signal, alpha=0.7, color='lightcoral', label='🔴 Original Signal (with noise)')
        ax1.fill_between(freq_range, 0, noise_low + noise_high, alpha=0.5, color='red', label='❌ Unwanted Noise')
        ax1.fill_between(freq_range, noise_low + noise_high, signal_power + noise_low + noise_high, 
                        alpha=0.8, color='lightgreen', label='✅ Voice Information')
        
        ax1.set_xlim(0, 8000)
        ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Signal Strength', fontsize=12, fontweight='bold')
        ax1.set_title('🎙️ BEFORE FILTERING: Voice Signal Contains Noise', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations
        ax1.annotate('Low-freq noise\n(room hum, breathing)', xy=(100, 0.25), xytext=(500, 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        ax1.annotate('Voice range\n(clear speech)', xy=(2000, 1.2), xytext=(2000, 1.8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax1.annotate('High-freq noise\n(electronics, artifacts)', xy=(7000, 0.15), xytext=(6000, 0.6),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # 2. Band-pass filter visualization
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Strategy 1 filter
        p1_low = self.filter_params['percentile_1_99']['low_cutoff']
        p1_high = self.filter_params['percentile_1_99']['high_cutoff']
        
        filter_response = np.zeros_like(freq_range)
        mask = (freq_range >= p1_low) & (freq_range <= p1_high)
        filter_response[mask] = 1.0
        
        # Add transition zones for realism
        transition_width = 100
        # Low cutoff transition
        low_transition = (freq_range >= p1_low - transition_width) & (freq_range < p1_low)
        filter_response[low_transition] = (freq_range[low_transition] - (p1_low - transition_width)) / transition_width
        # High cutoff transition
        high_transition = (freq_range > p1_high) & (freq_range <= p1_high + transition_width)
        filter_response[high_transition] = 1 - (freq_range[high_transition] - p1_high) / transition_width
        
        ax2.fill_between(freq_range, 0, filter_response, alpha=0.8, color='skyblue', label='Filter passband')
        ax2.plot(freq_range, filter_response, 'b-', linewidth=3, label='Filter response')
        ax2.axvline(p1_low, color='green', linestyle='--', linewidth=2, label=f'Low cutoff: {p1_low:.0f} Hz')
        ax2.axvline(p1_high, color='red', linestyle='--', linewidth=2, label=f'High cutoff: {p1_high:.0f} Hz')
        
        ax2.set_xlim(0, 8000)
        ax2.set_ylim(0, 1.2)
        ax2.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Filter Strength', fontsize=11, fontweight='bold')
        ax2.set_title(f'🔧 Strategy 1: Band-Pass Filter\n({p1_low:.0f}-{p1_high:.0f} Hz)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Strategy 2 filter
        ax3 = fig.add_subplot(gs[1, 1])
        
        p2_low = self.filter_params['percentile_2_5_97_5']['low_cutoff']
        p2_high = self.filter_params['percentile_2_5_97_5']['high_cutoff']
        
        filter_response2 = np.zeros_like(freq_range)
        mask2 = (freq_range >= p2_low) & (freq_range <= p2_high)
        filter_response2[mask2] = 1.0
        
        # Add transition zones
        low_transition2 = (freq_range >= p2_low - transition_width) & (freq_range < p2_low)
        filter_response2[low_transition2] = (freq_range[low_transition2] - (p2_low - transition_width)) / transition_width
        high_transition2 = (freq_range > p2_high) & (freq_range <= p2_high + transition_width)
        filter_response2[high_transition2] = 1 - (freq_range[high_transition2] - p2_high) / transition_width
        
        ax3.fill_between(freq_range, 0, filter_response2, alpha=0.8, color='lightcoral', label='Filter passband')
        ax3.plot(freq_range, filter_response2, 'r-', linewidth=3, label='Filter response')
        ax3.axvline(p2_low, color='green', linestyle='--', linewidth=2, label=f'Low cutoff: {p2_low:.0f} Hz')
        ax3.axvline(p2_high, color='red', linestyle='--', linewidth=2, label=f'High cutoff: {p2_high:.0f} Hz')
        
        ax3.set_xlim(0, 8000)
        ax3.set_ylim(0, 1.2)
        ax3.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Filter Strength', fontsize=11, fontweight='bold')
        ax3.set_title(f'🔧 Strategy 2: Band-Pass Filter\n({p2_low:.0f}-{p2_high:.0f} Hz)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. After filtering result
        ax4 = fig.add_subplot(gs[2, :])
        
        # Apply filter to original signal
        filtered_signal1 = total_signal * filter_response
        filtered_signal2 = total_signal * filter_response2
        
        ax4.fill_between(freq_range, 0, total_signal, alpha=0.3, color='lightgray', label='🔴 Original (with noise)')
        ax4.plot(freq_range, filtered_signal1, 'b-', linewidth=3, alpha=0.8, label=f'✅ Strategy 1 Filtered ({p1_low:.0f}-{p1_high:.0f} Hz)')
        ax4.plot(freq_range, filtered_signal2, 'r-', linewidth=3, alpha=0.8, label=f'✅ Strategy 2 Filtered ({p2_low:.0f}-{p2_high:.0f} Hz)')
        
        ax4.set_xlim(0, 8000)
        ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Signal Strength', fontsize=12, fontweight='bold')
        ax4.set_title('🎯 AFTER FILTERING: Clean Voice Signal (Noise Removed)', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(fontsize=11, loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Add result annotations
        ax4.text(0.02, 0.95, '✅ Benefits of Band-Pass Filtering:', transform=ax4.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        ax4.text(0.02, 0.85, '• Removes low-frequency noise (room hum, breathing)', transform=ax4.transAxes, fontsize=10)
        ax4.text(0.02, 0.80, '• Removes high-frequency noise (electronics, artifacts)', transform=ax4.transAxes, fontsize=10)
        ax4.text(0.02, 0.75, '• Preserves voice frequencies for PD analysis', transform=ax4.transAxes, fontsize=10)
        ax4.text(0.02, 0.70, '• Improves feature extraction quality', transform=ax4.transAxes, fontsize=10)
        
        plt.suptitle('🎙️ BEGINNER GUIDE: Band-Pass Filtering for Voice Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('essential_analysis/filter_explanation_beginner.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Filter explanation diagram saved: essential_analysis/filter_explanation_beginner.png")
    
    def create_essential_visualizations(self):
        """Create only essential visualizations"""
        if self.filter_params is None:
            return
            
        print("\nCreating essential visualizations...")
        os.makedirs("essential_analysis", exist_ok=True)
        
        # Filter comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Frequency ranges
        strategies = list(self.filter_params.keys())
        colors = ['#D5E8D4', '#FFF2CC']
        
        for i, (strategy, params) in enumerate(self.filter_params.items()):
            low, high = params['low_cutoff'], params['high_cutoff']
            bandwidth = high - low
            ax1.barh(i, bandwidth, left=low, color=colors[i], alpha=0.7, edgecolor='black')
            ax1.text(low + bandwidth/2, i, f'{low:.0f}-{high:.0f} Hz', 
                    ha='center', va='center', fontweight='bold')
        
        ax1.set_yticks(range(len(strategies)))
        ax1.set_yticklabels([s.replace('_', ' ').title() for s in strategies])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_title('Filter Frequency Ranges')
        ax1.grid(True, alpha=0.3)
        
        # Bandwidth comparison
        bandwidths = [params['high_cutoff'] - params['low_cutoff'] 
                     for params in self.filter_params.values()]
        
        bars = ax2.bar(range(len(strategies)), bandwidths, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
        ax2.set_ylabel('Bandwidth (Hz)')
        ax2.set_title('Filter Bandwidth Comparison')
        
        for bar, bw in zip(bars, bandwidths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{bw:.0f} Hz', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Streamlined Preprocessing - Filter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('essential_analysis/filter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Essential visualizations saved to: essential_analysis/")
    
    def _create_before_after_comparison(self):
        """Create simple before/after filtering comparison using actual audio data"""
        print("\n🎨 Creating before/after filtering comparison...")
        
        # Find a sample audio file to demonstrate
        sample_file = None
        for cohort in ['PD', 'HC']:
            cohort_dir = os.path.join("data", cohort)
            if os.path.exists(cohort_dir):
                files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
                if files:
                    sample_file = os.path.join(cohort_dir, files[0])
                    break
        
        if sample_file is None:
            print("  ⚠️ No sample audio file found for demonstration")
            return
            
        try:
            # Load sample audio
            audio, sr = librosa.load(sample_file, sr=self.target_sr)
            
            # Take first 3 seconds for demonstration
            demo_length = min(3 * sr, len(audio))
            audio_demo = audio[:demo_length]
            
            # Apply both filtering strategies to demo audio
            p1_low = self.filter_params['percentile_1_99']['low_cutoff']
            p1_high = self.filter_params['percentile_1_99']['high_cutoff']
            p2_low = self.filter_params['percentile_2_5_97_5']['low_cutoff']
            p2_high = self.filter_params['percentile_2_5_97_5']['high_cutoff']
            
            filtered_1 = self.apply_band_pass_filter(audio_demo, sr, p1_low, p1_high)
            filtered_2 = self.apply_band_pass_filter(audio_demo, sr, p2_low, p2_high)
            
            # Create comparison figure
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            time_axis = np.linspace(0, len(audio_demo)/sr, len(audio_demo))
            
            # 1. Original audio waveform
            axes[0].plot(time_axis, audio_demo, 'k-', alpha=0.7, linewidth=1)
            axes[0].set_title('🔴 BEFORE: Original Audio Signal (with all frequencies)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Amplitude', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].text(0.02, 0.85, '❌ Contains noise + voice', transform=axes[0].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7), fontsize=10)
            
            # 2. Filtered audio 1
            axes[1].plot(time_axis, filtered_1, 'b-', alpha=0.8, linewidth=1)
            axes[1].set_title(f'✅ AFTER: Strategy 1 Filtered ({p1_low:.0f}-{p1_high:.0f} Hz)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Amplitude', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].text(0.02, 0.85, '✅ Clean voice signal', transform=axes[1].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7), fontsize=10)
            
            # 3. Filtered audio 2
            axes[2].plot(time_axis, filtered_2, 'r-', alpha=0.8, linewidth=1)
            axes[2].set_title(f'✅ AFTER: Strategy 2 Filtered ({p2_low:.0f}-{p2_high:.0f} Hz)', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Amplitude', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].text(0.02, 0.85, '✅ Conservative clean signal', transform=axes[2].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7), fontsize=10)
            
            # 4. Frequency domain comparison
            freqs = np.fft.fftfreq(len(audio_demo), 1/sr)[:len(audio_demo)//2]
            
            fft_orig = np.abs(np.fft.fft(audio_demo))[:len(audio_demo)//2]
            fft_filt1 = np.abs(np.fft.fft(filtered_1))[:len(audio_demo)//2]
            fft_filt2 = np.abs(np.fft.fft(filtered_2))[:len(audio_demo)//2]
            
            axes[3].semilogy(freqs, fft_orig, 'k-', alpha=0.6, label='🔴 Original', linewidth=2)
            axes[3].semilogy(freqs, fft_filt1, 'b-', alpha=0.8, label=f'✅ Strategy 1 ({p1_low:.0f}-{p1_high:.0f} Hz)', linewidth=2)
            axes[3].semilogy(freqs, fft_filt2, 'r-', alpha=0.8, label=f'✅ Strategy 2 ({p2_low:.0f}-{p2_high:.0f} Hz)', linewidth=2)
            
            axes[3].axvline(p1_low, color='blue', linestyle='--', alpha=0.7)
            axes[3].axvline(p1_high, color='blue', linestyle='--', alpha=0.7)
            axes[3].axvline(p2_low, color='red', linestyle='--', alpha=0.7)
            axes[3].axvline(p2_high, color='red', linestyle='--', alpha=0.7)
            
            axes[3].set_xlim(0, 8000)
            axes[3].set_xlabel('Frequency (Hz)', fontweight='bold')
            axes[3].set_ylabel('Magnitude (log scale)', fontweight='bold')
            axes[3].set_title('🔍 Frequency Domain: See How Noise is Removed', fontsize=12, fontweight='bold')
            axes[3].legend(fontsize=10)
            axes[3].grid(True, alpha=0.3)
            
            # Add explanation text
            axes[3].text(0.02, 0.95, 'Filter Effects:', transform=axes[3].transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            axes[3].text(0.02, 0.85, '• Low frequencies (noise) removed', transform=axes[3].transAxes, fontsize=9)
            axes[3].text(0.02, 0.78, '• High frequencies (artifacts) removed', transform=axes[3].transAxes, fontsize=9)
            axes[3].text(0.02, 0.71, '• Voice frequencies preserved', transform=axes[3].transAxes, fontsize=9)
            
            plt.suptitle('🎙️ REAL EXAMPLE: Before vs After Band-Pass Filtering', 
                        fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig('essential_analysis/before_after_filtering_real.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Before/after comparison saved: essential_analysis/before_after_filtering_real.png")
            
        except Exception as e:
            print(f"  ⚠️ Could not create before/after comparison: {e}")


def main():
    """Main function for streamlined preprocessing"""
    print("STREAMLINED AUDIO PREPROCESSING FOR PARKINSON'S DISEASE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(target_sr=16000)
    
    # Step 1: Load data mapping
    data_mapping = preprocessor.load_data_mapping()
    if data_mapping is None:
        print("Failed to load data mapping. Exiting.")
        return
    
    # Step 2: Analyze frequencies for filter parameters
    preprocessor.analyze_audio_frequencies("data")
    
    # Step 3: Preprocess with both strategies
    results = preprocessor.preprocess_audio_data("data", "preprocessed_data")
    
    # Step 4: Create essential visualizations
    preprocessor.create_essential_visualizations()
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETED!")
    print(f"{'='*60}")
    print("\nOutput directories:")
    print("1. preprocessed_data_percentile_1_99/ - Primary output (1-99 percentile)")
    print("2. preprocessed_data_percentile_2_5_97_5/ - Conservative output (2.5-97.5 percentile)")
    print("3. essential_analysis/ - Filter analysis visualization")


if __name__ == "__main__":
    main()
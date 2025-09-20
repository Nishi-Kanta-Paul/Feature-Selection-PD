import matplotlib.pyplot as plt
import numpy as np
import os
import librosa

class FilterVisualization:
    """
    Separate class for creating filter diagrams and visualizations
    """
    
    def __init__(self):
        self.output_dir = "essential_analysis"
        
    def create_beginner_filter_explanation(self, filter_params):
        """Create beginner-friendly diagrams explaining band-pass filtering"""
        print("\n🎨 Creating beginner-friendly filter explanation diagrams...")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        p1_low = filter_params['percentile_1_99']['low_cutoff']
        p1_high = filter_params['percentile_1_99']['high_cutoff']
        
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
        
        p2_low = filter_params['percentile_2_5_97_5']['low_cutoff']
        p2_high = filter_params['percentile_2_5_97_5']['high_cutoff']
        
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
        
        plt.savefig(f'{self.output_dir}/filter_explanation_beginner.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Filter explanation diagram saved: {self.output_dir}/filter_explanation_beginner.png")
    
    def create_before_after_comparison(self, filter_params, sample_audio_path=None, target_sr=16000):
        """Create simple before/after filtering comparison using actual audio data"""
        print("\n🎨 Creating before/after filtering comparison...")
        
        # Find a sample audio file to demonstrate
        if sample_audio_path is None:
            sample_file = None
            for cohort in ['PD', 'HC']:
                cohort_dir = os.path.join("data", cohort)
                if os.path.exists(cohort_dir):
                    files = [f for f in os.listdir(cohort_dir) if f.endswith('.wav')]
                    if files:
                        sample_file = os.path.join(cohort_dir, files[0])
                        break
        else:
            sample_file = sample_audio_path
        
        if sample_file is None:
            print("  ⚠️ No sample audio file found for demonstration")
            return
            
        try:
            # Load sample audio
            audio, sr = librosa.load(sample_file, sr=target_sr)
            
            # Take first 3 seconds for demonstration
            demo_length = min(3 * sr, len(audio))
            audio_demo = audio[:demo_length]
            
            # Apply both filtering strategies to demo audio
            p1_low = filter_params['percentile_1_99']['low_cutoff']
            p1_high = filter_params['percentile_1_99']['high_cutoff']
            p2_low = filter_params['percentile_2_5_97_5']['low_cutoff']
            p2_high = filter_params['percentile_2_5_97_5']['high_cutoff']
            
            # Import filter function
            from scipy.signal import butter, filtfilt
            
            def apply_bandpass_filter(audio, sr, low_cutoff, high_cutoff):
                nyquist = sr / 2
                low_cutoff = max(1, min(low_cutoff, nyquist - 1))
                high_cutoff = max(low_cutoff + 1, min(high_cutoff, nyquist - 1))
                b, a = butter(3, [low_cutoff, high_cutoff], btype='band', fs=sr)
                return filtfilt(b, a, audio)
            
            filtered_1 = apply_bandpass_filter(audio_demo, sr, p1_low, p1_high)
            filtered_2 = apply_bandpass_filter(audio_demo, sr, p2_low, p2_high)
            
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
            plt.savefig(f'{self.output_dir}/before_after_filtering_real.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Before/after comparison saved: {self.output_dir}/before_after_filtering_real.png")
            
        except Exception as e:
            print(f"  ⚠️ Could not create before/after comparison: {e}")
    
    def create_filter_analysis(self, filter_params):
        """Create technical filter analysis visualization"""
        print("\nCreating filter analysis visualization...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Filter comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Frequency ranges
        strategies = list(filter_params.keys())
        colors = ['#D5E8D4', '#FFF2CC']
        
        for i, (strategy, params) in enumerate(filter_params.items()):
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
                     for params in filter_params.values()]
        
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
        
        plt.suptitle('Audio Preprocessing - Filter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/filter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Filter analysis visualization saved: {self.output_dir}/filter_analysis.png")
    
    def create_all_visualizations(self, filter_params, sample_audio_path=None, target_sr=16000):
        """Create all visualizations at once"""
        self.create_beginner_filter_explanation(filter_params)
        self.create_before_after_comparison(filter_params, sample_audio_path, target_sr)
        self.create_filter_analysis(filter_params)
        print(f"\n✅ All visualizations created in: {self.output_dir}/")
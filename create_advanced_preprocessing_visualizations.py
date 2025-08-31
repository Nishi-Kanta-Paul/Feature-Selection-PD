import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from scipy.signal import butter, filtfilt
import soundfile as sf

def create_advanced_preprocessing_visualizations():
    """
    Create advanced visualizations showing detailed effects of each preprocessing step:
    1. Spectrogram comparison (time-frequency analysis)
    2. Energy plot (detailed VAD analysis)
    3. Histogram/Amplitude distribution (normalization effects)
    """
    
    # Create visualization directory
    viz_dir = "preprocessing_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Creating advanced preprocessing visualizations...")
    
    # Try to find an audio file for real analysis
    audio_file = None
    for cohort in ['PD', 'HC']:
        data_dir = f"data/{cohort}"
        if os.path.exists(data_dir):
            wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
            if wav_files:
                audio_file = os.path.join(data_dir, wav_files[0])
                break
    
    if audio_file:
        print(f"Using real audio file: {audio_file}")
        create_real_audio_advanced_visualizations(audio_file, viz_dir)
    else:
        print("No audio files found, creating synthetic demonstrations...")
        create_synthetic_advanced_visualizations(viz_dir)
    
    print(f"Advanced visualizations saved to: {os.path.abspath(viz_dir)}")

def create_real_audio_advanced_visualizations(audio_file, viz_dir):
    """
    Create advanced visualizations using real audio data
    """
    try:
        # Load original audio
        audio_orig, sr_orig = librosa.load(audio_file, sr=None)
        
        # Step-by-step processing to collect intermediate results
        print("Processing audio through pipeline...")
        
        # Step 1: Resample to 16 kHz
        audio_resampled, sr_target = librosa.load(audio_file, sr=16000)
        
        # Step 2: High-pass filter
        b, a = butter(3, 80 / (sr_target / 2), btype='high')
        audio_filtered = filtfilt(b, a, audio_resampled)
        
        # Step 3: Voice activity detection (get intermediate results)
        audio_vad, energy_data = apply_advanced_vad(audio_filtered, sr_target)
        
        # Step 4: Normalization
        if np.max(np.abs(audio_vad)) > 0:
            audio_normalized = audio_vad / np.max(np.abs(audio_vad)) * 0.9
        else:
            audio_normalized = audio_vad
        
        # Create the three advanced visualizations
        create_spectrogram_comparison(
            [(audio_orig, sr_orig, "Original"),
             (audio_resampled, sr_target, "Resampled"),
             (audio_filtered, sr_target, "Filtered"),
             (audio_normalized, sr_target, "Final")],
            viz_dir
        )
        
        create_energy_plot_analysis(
            audio_filtered, sr_target, energy_data, viz_dir
        )
        
        create_amplitude_distribution_analysis(
            [(audio_orig, "Original"),
             (audio_resampled, "Resampled"), 
             (audio_filtered, "Filtered"),
             (audio_normalized, "Normalized")],
            viz_dir
        )
        
    except Exception as e:
        print(f"Error processing real audio: {e}")
        create_synthetic_advanced_visualizations(viz_dir)

def create_synthetic_advanced_visualizations(viz_dir):
    """
    Create advanced visualizations using synthetic data
    """
    print("Creating synthetic audio for demonstration...")
    
    # Generate realistic synthetic speech signal
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create synthetic speech with multiple harmonics and silence periods
    speech = np.zeros_like(t)
    
    # Speech segments with different characteristics
    speech_segments = [(0.3, 1.0, 120), (1.5, 2.2, 180), (2.5, 2.9, 150)]  # (start, end, f0)
    
    for start, end, f0 in speech_segments:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        t_seg = t[start_idx:end_idx]
        
        # Generate harmonic complex
        speech_seg = (np.sin(2 * np.pi * f0 * t_seg) +
                     0.5 * np.sin(2 * np.pi * f0 * 2 * t_seg) +
                     0.3 * np.sin(2 * np.pi * f0 * 3 * t_seg) +
                     0.2 * np.sin(2 * np.pi * f0 * 4 * t_seg))
        
        # Add envelope
        envelope = np.exp(-3 * ((t_seg - (start + end)/2) / (end - start))**2)
        speech_seg *= envelope
        
        speech[start_idx:end_idx] = speech_seg
    
    # Add different types of noise
    noise_low = 0.15 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
    noise_high = 0.05 * np.random.randn(len(t))     # White noise
    
    # Create original noisy signal
    audio_orig = speech + noise_low + noise_high
    
    # Process through pipeline
    audio_resampled = audio_orig  # Already at 16 kHz
    
    # High-pass filter
    b, a = butter(3, 80 / (sr / 2), btype='high')
    audio_filtered = filtfilt(b, a, audio_resampled)
    
    # VAD processing
    audio_vad, energy_data = apply_advanced_vad(audio_filtered, sr)
    
    # Normalization
    if np.max(np.abs(audio_vad)) > 0:
        audio_normalized = audio_vad / np.max(np.abs(audio_vad)) * 0.9
    else:
        audio_normalized = audio_vad
    
    # Create visualizations
    create_spectrogram_comparison(
        [(audio_orig, sr, "Original with Noise"),
         (audio_resampled, sr, "Resampled (16kHz)"),
         (audio_filtered, sr, "High-Pass Filtered"),
         (audio_normalized, sr, "Final Processed")],
        viz_dir
    )
    
    create_energy_plot_analysis(
        audio_filtered, sr, energy_data, viz_dir
    )
    
    create_amplitude_distribution_analysis(
        [(audio_orig, "Original"),
         (audio_resampled, "Resampled"), 
         (audio_filtered, "Filtered"),
         (audio_normalized, "Normalized")],
        viz_dir
    )

def apply_advanced_vad(audio, sr, threshold=0.02):
    """
    Apply VAD and return both processed audio and detailed energy data
    """
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.01 * sr)     # 10ms
    
    energies = []
    frame_times = []
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)
        frame_times.append(i / sr)
    
    energies = np.array(energies)
    frame_times = np.array(frame_times)
    
    # Normalize energies
    max_energy = np.max(energies)
    if max_energy > 0:
        energies_norm = energies / max_energy
    else:
        energies_norm = energies
    
    # Apply threshold
    voice_frames = energies_norm > threshold
    
    # Extract voice segments
    voice_audio = []
    for i, is_voice in enumerate(voice_frames):
        if is_voice:
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            voice_audio.extend(audio[start:end])
    
    processed_audio = np.array(voice_audio) if voice_audio else audio
    
    # Return both processed audio and energy analysis data
    energy_data = {
        'frame_times': frame_times,
        'energies_raw': energies,
        'energies_norm': energies_norm,
        'voice_frames': voice_frames,
        'threshold': threshold,
        'original_audio': audio
    }
    
    return processed_audio, energy_data

def create_spectrogram_comparison(audio_stages, viz_dir):
    """
    Create spectrogram comparison showing frequency content changes
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (audio, sr, title) in enumerate(audio_stages):
        if i >= 4:
            break
            
        # Compute spectrogram
        n_fft = 1024
        hop_length = 256
        
        # Limit audio length for visualization (first 3 seconds)
        max_samples = int(3 * sr)
        audio_viz = audio[:max_samples] if len(audio) > max_samples else audio
        
        # Compute mel-spectrogram for better visualization
        S = librosa.feature.melspectrogram(
            y=audio_viz, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=128, fmax=sr//2
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Plot spectrogram
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i],
            cmap='viridis'
        )
        
        axes[i].set_title(f'{title}\nFreq Content Analysis', fontweight='bold', fontsize=12)
        axes[i].set_ylabel('Mel Frequency (Hz)')
        
        # Add colorbar
        plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
        
        # Add frequency annotations
        if i == 0:  # Original
            axes[i].axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=2)
            axes[i].text(0.1, 90, '80 Hz cutoff', color='red', fontweight='bold')
        elif i == 2:  # Filtered
            axes[i].axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2)
            axes[i].text(0.1, 90, 'Frequencies below 80 Hz removed', color='green', fontweight='bold')
    
    plt.suptitle('Spectrogram Comparison: Time-Frequency Analysis of Preprocessing Effects', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/advanced_spectrogram_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_energy_plot_analysis(audio, sr, energy_data, viz_dir):
    """
    Create detailed energy analysis showing VAD processing
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Time vector for original audio
    t_audio = np.arange(len(audio)) / sr
    
    # Panel 1: Original filtered audio
    axes[0].plot(t_audio, audio, color='#2196F3', alpha=0.7, linewidth=0.8)
    axes[0].set_title('1. High-Pass Filtered Audio Signal', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max(t_audio))
    
    # Panel 2: Frame-by-frame energy calculation
    frame_times = energy_data['frame_times']
    energies_raw = energy_data['energies_raw']
    
    axes[1].plot(frame_times, energies_raw, color='#FF9800', linewidth=2, label='Raw Energy')
    axes[1].set_title('2. Frame-by-Frame Energy Calculation (RMS Energy)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Energy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, max(frame_times))
    
    # Panel 3: Normalized energy with threshold
    energies_norm = energy_data['energies_norm']
    voice_frames = energy_data['voice_frames']
    threshold = energy_data['threshold']
    
    axes[2].plot(frame_times, energies_norm, color='#FF9800', linewidth=2, label='Normalized Energy')
    axes[2].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    axes[2].fill_between(frame_times, 0, energies_norm, where=voice_frames, 
                        alpha=0.3, color='green', label='Voice-active frames')
    axes[2].set_title('3. Energy Normalization and Threshold Application', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Normalized Energy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, max(frame_times))
    
    # Panel 4: VAD decision and statistics
    vad_signal = np.zeros_like(voice_frames, dtype=float)
    vad_signal[voice_frames] = 1.0
    
    axes[3].plot(frame_times, vad_signal, color='#4CAF50', linewidth=3, drawstyle='steps-post')
    axes[3].set_title('4. Voice Activity Detection Result', fontweight='bold', fontsize=12)
    axes[3].set_ylabel('Voice Activity')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, max(frame_times))
    
    # Add detailed statistics
    total_frames = len(voice_frames)
    voice_frame_count = np.sum(voice_frames)
    voice_ratio = voice_frame_count / total_frames
    total_duration = max(frame_times)
    voice_duration = voice_frame_count * 0.01  # 10ms hop size
    
    stats_text = f"""VAD Performance Statistics:
Total Duration: {total_duration:.2f} seconds
Voice Duration: {voice_duration:.2f} seconds
Silence Removed: {total_duration - voice_duration:.2f} seconds
Total Frames: {total_frames}
Voice Frames: {voice_frame_count}
Voice Activity Ratio: {voice_ratio:.3f}
Energy Threshold: {threshold}
Max Energy: {np.max(energies_raw):.6f}"""
    
    axes[3].text(0.02, 0.98, stats_text, transform=axes[3].transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('Detailed Energy Analysis: Voice Activity Detection Processing', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/advanced_energy_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_amplitude_distribution_analysis(audio_stages, viz_dir):
    """
    Create amplitude distribution analysis showing normalization effects
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#FF5722', '#FF9800', '#FFC107', '#2196F3']
    
    for i, (audio, stage_name) in enumerate(audio_stages):
        # Histogram of amplitude values
        axes[i].hist(audio, bins=100, alpha=0.7, color=colors[i], density=True, edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_amp = np.mean(audio)
        std_amp = np.std(audio)
        max_amp = np.max(np.abs(audio))
        rms_amp = np.sqrt(np.mean(audio**2))
        
        # Add vertical lines for key statistics
        axes[i].axvline(mean_amp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_amp:.4f}')
        axes[i].axvline(mean_amp + std_amp, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean+STD: {mean_amp + std_amp:.4f}')
        axes[i].axvline(mean_amp - std_amp, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean-STD: {mean_amp - std_amp:.4f}')
        
        axes[i].set_title(f'{stage_name} Signal\nAmplitude Distribution', fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Amplitude')
        axes[i].set_ylabel('Probability Density')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=9)
        
        # Add statistics text box
        stats_text = f"""Statistics:
Mean: {mean_amp:.4f}
STD: {std_amp:.4f}
Max: {max_amp:.4f}
RMS: {rms_amp:.4f}
Range: [{np.min(audio):.4f}, {np.max(audio):.4f}]
Dynamic Range: {20*np.log10(max_amp/rms_amp) if rms_amp > 0 else 0:.1f} dB"""
        
        axes[i].text(0.65, 0.95, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Amplitude Distribution Analysis: Signal Normalization Effects', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/advanced_amplitude_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_advanced_preprocessing_visualizations()
    print("Advanced preprocessing visualizations created successfully!")
    print("\nNew visualizations:")
    print("1. advanced_spectrogram_comparison.png - Time-frequency analysis showing filtering effects")
    print("2. advanced_energy_analysis.png - Detailed VAD processing with statistics")
    print("3. advanced_amplitude_distribution.png - Histogram analysis showing normalization effects")

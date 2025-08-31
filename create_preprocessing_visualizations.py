import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from scipy.signal import butter, filtfilt
import soundfile as sf

def create_preprocessing_visualizations():
    """
    Create comprehensive visualizations for the audio preprocessing pipeline
    """
    
    # Create visualization directory
    viz_dir = "preprocessing_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Creating audio preprocessing visualizations...")
    
    # 1. Create pipeline overview diagram
    create_pipeline_diagram(viz_dir)
    
    # 2. Create sample audio processing demonstration
    if os.path.exists("data"):
        create_audio_processing_demo(viz_dir)
    else:
        create_synthetic_audio_demo(viz_dir)
    
    # 3. Create filter response visualization
    create_filter_response_plot(viz_dir)
    
    # 4. Create energy-based voice activity detection demo
    create_vad_demonstration(viz_dir)
    
    print(f"Visualizations saved to: {os.path.abspath(viz_dir)}")

def create_pipeline_diagram(viz_dir):
    """
    Create a flowchart-style diagram of the preprocessing pipeline
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    input_color = '#E3F2FD'
    process_color = '#BBDEFB' 
    output_color = '#90CAF9'
    
    # Title
    ax.text(5, 11.5, 'Audio Preprocessing Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Step boxes with detailed information
    steps = [
        {
            'title': 'Input Audio Files',
            'details': ['Raw WAV files from data/PD and data/HC',
                       'Various sample rates and quality levels',
                       'Potential noise and silence segments'],
            'pos': (2, 9.5),
            'color': input_color
        },
        {
            'title': '1. Load & Resample',
            'details': ['Target: 16 kHz sample rate',
                       'librosa.load() with sr=16000',
                       'Optimal balance: quality vs efficiency'],
            'pos': (2, 8),
            'color': process_color
        },
        {
            'title': '2. High-Pass Filter',
            'details': ['3rd-order Butterworth filter',
                       'Cutoff: 80 Hz',
                       'Removes low-frequency noise & rumble'],
            'pos': (2, 6.5),
            'color': process_color
        },
        {
            'title': '3. Voice Activity Detection',
            'details': ['Frame size: 25ms, Hop: 10ms',
                       'Energy threshold: 2% of max',
                       'Removes silence & non-speech'],
            'pos': (2, 5),
            'color': process_color
        },
        {
            'title': '4. Amplitude Normalization',
            'details': ['Scale to 90% of max amplitude',
                       'Prevents clipping',
                       'Consistent signal levels'],
            'pos': (2, 3.5),
            'color': process_color
        },
        {
            'title': '5. Length Standardization',
            'details': ['Minimum duration: 0.5 seconds',
                       'Zero-padding if needed',
                       'Ensures reliable feature extraction'],
            'pos': (2, 2),
            'color': process_color
        },
        {
            'title': 'Output: Preprocessed Audio',
            'details': ['Clean, normalized WAV files',
                       'preprocessed_data/PD/ and /HC/',
                       'Ready for feature extraction'],
            'pos': (8, 2),
            'color': output_color
        }
    ]
    
    # Draw boxes and text
    for step in steps:
        x, y = step['pos']
        
        # Main box
        rect = plt.Rectangle((x-1.8, y-0.6), 3.6, 1.2, 
                           facecolor=step['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # Title
        ax.text(x, y+0.2, step['title'], fontsize=12, fontweight='bold', 
                ha='center', va='center')
        
        # Details
        for i, detail in enumerate(step['details']):
            ax.text(x, y-0.1-i*0.15, detail, fontsize=9, 
                    ha='center', va='center')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#1976D2')
    
    # Vertical arrows for main pipeline
    for i in range(len(steps)-2):
        y_start = steps[i]['pos'][1] - 0.6
        y_end = steps[i+1]['pos'][1] + 0.6
        ax.annotate('', xy=(2, y_end), xytext=(2, y_start), arrowprops=arrow_props)
    
    # Horizontal arrow to output
    ax.annotate('', xy=(6.2, 2), xytext=(3.8, 2), arrowprops=arrow_props)
    
    # Add technical specifications
    specs_text = [
        'Technical Specifications:',
        '• Sample Rate: 16,000 Hz',
        '• Frame Analysis: 25ms windows, 10ms hop',
        '• Filter: Butterworth 3rd order, 80 Hz cutoff',
        '• Energy Threshold: Adaptive (2% of max)',
        '• Normalization: 90% of amplitude range',
        '• Min Duration: 0.5 seconds (8,000 samples)'
    ]
    
    for i, spec in enumerate(specs_text):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(6.5, 9-i*0.3, spec, fontsize=10, fontweight=weight, va='top')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/preprocessing_pipeline_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_audio_processing_demo(viz_dir):
    """
    Demonstrate preprocessing steps on actual audio file
    """
    # Try to find an audio file
    audio_file = None
    for cohort in ['PD', 'HC']:
        data_dir = f"data/{cohort}"
        if os.path.exists(data_dir):
            wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
            if wav_files:
                audio_file = os.path.join(data_dir, wav_files[0])
                break
    
    if not audio_file:
        print("No audio files found, creating synthetic demo...")
        create_synthetic_audio_demo(viz_dir)
        return
    
    try:
        # Load original audio
        audio_orig, sr_orig = librosa.load(audio_file, sr=None)
        
        # Step-by-step processing
        audio_resampled, sr_target = librosa.load(audio_file, sr=16000)
        
        # High-pass filter
        b, a = butter(3, 80 / (sr_target / 2), btype='high')
        audio_filtered = filtfilt(b, a, audio_resampled)
        
        # Voice activity detection (simplified)
        audio_vad = apply_simple_vad(audio_filtered, sr_target)
        
        # Normalization
        if np.max(np.abs(audio_vad)) > 0:
            audio_normalized = audio_vad / np.max(np.abs(audio_vad)) * 0.9
        else:
            audio_normalized = audio_vad
        
        # Create visualization
        fig, axes = plt.subplots(5, 1, figsize=(15, 12))
        
        # Time vectors
        t_orig = np.arange(len(audio_orig)) / sr_orig
        t_processed = np.arange(len(audio_resampled)) / sr_target
        t_filtered = np.arange(len(audio_filtered)) / sr_target
        t_vad = np.arange(len(audio_vad)) / sr_target
        t_norm = np.arange(len(audio_normalized)) / sr_target
        
        # Plot each step
        axes[0].plot(t_orig, audio_orig, color='#FF5722', alpha=0.7)
        axes[0].set_title(f'1. Original Audio (SR: {sr_orig} Hz)', fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_processed, audio_resampled, color='#FF9800', alpha=0.7)
        axes[1].set_title('2. Resampled to 16 kHz', fontweight='bold')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_filtered, audio_filtered, color='#FFC107', alpha=0.7)
        axes[2].set_title('3. High-Pass Filtered (80 Hz cutoff)', fontweight='bold')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(t_vad, audio_vad, color='#4CAF50', alpha=0.7)
        axes[3].set_title('4. Voice Activity Detection (Silence Removed)', fontweight='bold')
        axes[3].set_ylabel('Amplitude')
        axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(t_norm, audio_normalized, color='#2196F3', alpha=0.7)
        axes[4].set_title('5. Normalized & Length Standardized', fontweight='bold')
        axes[4].set_ylabel('Amplitude')
        axes[4].set_xlabel('Time (seconds)')
        axes[4].grid(True, alpha=0.3)
        
        plt.suptitle('Audio Preprocessing Pipeline - Step-by-Step Demonstration', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/audio_processing_steps_demo.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        create_synthetic_audio_demo(viz_dir)

def create_synthetic_audio_demo(viz_dir):
    """
    Create demonstration using synthetic audio signal
    """
    # Generate synthetic speech-like signal with noise
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create synthetic speech signal
    speech = np.sin(2 * np.pi * 150 * t) * np.exp(-t * 0.5)  # Fundamental
    speech += 0.3 * np.sin(2 * np.pi * 300 * t) * np.exp(-t * 0.5)  # First harmonic
    speech += 0.2 * np.sin(2 * np.pi * 450 * t) * np.exp(-t * 0.5)  # Second harmonic
    
    # Add silence periods
    speech[int(0.5*sr):int(0.8*sr)] = 0  # Silence
    speech[int(1.2*sr):int(1.5*sr)] = 0  # Silence
    
    # Add low-frequency noise
    noise_low = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz hum
    noise_high = 0.05 * np.random.randn(len(t))   # White noise
    
    audio_orig = speech + noise_low + noise_high
    
    # Process step by step
    # 1. Resample to 16kHz
    audio_resampled = librosa.resample(audio_orig, orig_sr=sr, target_sr=16000)
    sr_target = 16000
    
    # 2. High-pass filter
    b, a = butter(3, 80 / (sr_target / 2), btype='high')
    audio_filtered = filtfilt(b, a, audio_resampled)
    
    # 3. Voice activity detection
    audio_vad = apply_simple_vad(audio_filtered, sr_target)
    
    # 4. Normalization
    if np.max(np.abs(audio_vad)) > 0:
        audio_normalized = audio_vad / np.max(np.abs(audio_vad)) * 0.9
    else:
        audio_normalized = audio_vad
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    
    # Time vectors
    t_orig = np.arange(len(audio_orig)) / sr
    t_resampled = np.arange(len(audio_resampled)) / sr_target
    t_filtered = np.arange(len(audio_filtered)) / sr_target
    t_vad = np.arange(len(audio_vad)) / sr_target
    t_norm = np.arange(len(audio_normalized)) / sr_target
    
    # Plot each step
    axes[0].plot(t_orig, audio_orig, color='#FF5722', alpha=0.7)
    axes[0].set_title(f'1. Original Audio with Noise (SR: {sr} Hz)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_resampled, audio_resampled, color='#FF9800', alpha=0.7)
    axes[1].set_title('2. Resampled to 16 kHz', fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t_filtered, audio_filtered, color='#FFC107', alpha=0.7)
    axes[2].set_title('3. High-Pass Filtered (80 Hz cutoff) - Noise Removed', fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(t_vad, audio_vad, color='#4CAF50', alpha=0.7)
    axes[3].set_title('4. Voice Activity Detection - Silence Removed', fontweight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].grid(True, alpha=0.3)
    
    axes[4].plot(t_norm, audio_normalized, color='#2196F3', alpha=0.7)
    axes[4].set_title('5. Normalized & Length Standardized', fontweight='bold')
    axes[4].set_ylabel('Amplitude')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].grid(True, alpha=0.3)
    
    plt.suptitle('Audio Preprocessing Pipeline - Synthetic Signal Demonstration', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/synthetic_audio_processing_demo.png", dpi=300, bbox_inches='tight')
    plt.close()

def apply_simple_vad(audio, sr, threshold=0.02):
    """
    Simple voice activity detection for demonstration
    """
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.01 * sr)     # 10ms
    
    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    if not energies:
        return audio
    
    energies = np.array(energies)
    max_energy = np.max(energies)
    if max_energy > 0:
        energies = energies / max_energy
    
    voice_frames = energies > threshold
    
    voice_audio = []
    for i, is_voice in enumerate(voice_frames):
        if is_voice:
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            voice_audio.extend(audio[start:end])
    
    return np.array(voice_audio) if voice_audio else audio

def create_filter_response_plot(viz_dir):
    """
    Visualize the frequency response of the high-pass filter
    """
    # Filter parameters
    sr = 16000
    cutoff = 80
    order = 3
    
    # Create filter
    b, a = butter(order, cutoff / (sr / 2), btype='high')
    
    # Frequency response
    from scipy.signal import freqz
    w, h = freqz(b, a, worN=8000)
    freq = w * sr / (2 * np.pi)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnitude response
    ax1.plot(freq, 20 * np.log10(abs(h)), 'b', linewidth=2)
    ax1.axvline(cutoff, color='r', linestyle='--', linewidth=2, label=f'Cutoff: {cutoff} Hz')
    ax1.axhline(-3, color='g', linestyle='--', alpha=0.7, label='-3 dB line')
    ax1.set_title('High-Pass Filter Frequency Response - Magnitude', fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(10, 1000)
    ax1.set_ylim(-60, 5)
    
    # Add frequency regions
    ax1.axvspan(10, 80, alpha=0.2, color='red', label='Removed frequencies')
    ax1.axvspan(85, 255, alpha=0.2, color='green', label='Male speech F0')
    ax1.axvspan(165, 265, alpha=0.2, color='blue', label='Female speech F0')
    
    # Phase response
    ax2.plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, 'b', linewidth=2)
    ax2.axvline(cutoff, color='r', linestyle='--', linewidth=2)
    ax2.set_title('High-Pass Filter Frequency Response - Phase', fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 1000)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/filter_frequency_response.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_vad_demonstration(viz_dir):
    """
    Demonstrate voice activity detection algorithm
    """
    # Create synthetic signal with speech and silence
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create signal with alternating speech and silence
    signal = np.zeros_like(t)
    
    # Speech segments
    speech_segments = [(0.2, 0.8), (1.2, 1.8), (2.2, 2.8)]
    
    for start, end in speech_segments:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        t_seg = t[start_idx:end_idx]
        
        # Synthetic speech with multiple harmonics
        speech_seg = (np.sin(2 * np.pi * 150 * t_seg) + 
                     0.5 * np.sin(2 * np.pi * 300 * t_seg) +
                     0.3 * np.sin(2 * np.pi * 450 * t_seg))
        speech_seg *= np.exp(-2 * (t_seg - (start + end)/2)**2)  # Envelope
        
        signal[start_idx:end_idx] = speech_seg
    
    # Add low-level noise throughout
    signal += 0.05 * np.random.randn(len(signal))
    
    # Apply VAD algorithm
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.01 * sr)     # 10ms
    
    energies = []
    frame_times = []
    
    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i:i + frame_size]
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
    threshold = 0.02
    voice_frames = energies_norm > threshold
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Original signal
    axes[0].plot(t, signal, color='#2196F3', alpha=0.7, linewidth=1)
    axes[0].set_title('Original Audio Signal with Speech and Silence Segments', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Mark speech segments
    for start, end in speech_segments:
        axes[0].axvspan(start, end, alpha=0.2, color='green', label='Speech segments' if start == speech_segments[0][0] else "")
    axes[0].legend()
    
    # Frame energies
    axes[1].plot(frame_times, energies_norm, color='#FF9800', linewidth=2, label='Normalized Energy')
    axes[1].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    axes[1].fill_between(frame_times, 0, energies_norm, where=voice_frames, 
                        alpha=0.3, color='green', label='Voice frames')
    axes[1].set_title('Frame-by-Frame Energy Analysis', fontweight='bold')
    axes[1].set_ylabel('Normalized Energy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Voice activity detection result
    vad_signal = np.zeros_like(voice_frames, dtype=float)
    vad_signal[voice_frames] = 1.0
    
    axes[2].plot(frame_times, vad_signal, color='#4CAF50', linewidth=3, drawstyle='steps-post')
    axes[2].set_title('Voice Activity Detection Result', fontweight='bold')
    axes[2].set_ylabel('Voice Activity')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics
    total_frames = len(voice_frames)
    voice_frame_count = np.sum(voice_frames)
    voice_ratio = voice_frame_count / total_frames
    
    stats_text = f'VAD Statistics:\nTotal frames: {total_frames}\nVoice frames: {voice_frame_count}\nVoice ratio: {voice_ratio:.2f}'
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Voice Activity Detection Algorithm Demonstration', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/voice_activity_detection_demo.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_preprocessing_visualizations()
    print("Audio preprocessing visualizations created successfully!")

import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt

def simple_audio_preprocessing():
    """
    Simple and fast audio preprocessing:
    - Load and resample to standard rate
    - Basic noise reduction using spectral subtraction
    - Remove silence using energy threshold
    - Normalize amplitude
    """
    
    # Directories
    input_base = "data"
    output_base = "preprocessed_data"
    
    # Create output directories
    for cohort in ["PD", "HC"]:
        os.makedirs(os.path.join(output_base, cohort), exist_ok=True)
    
    # Parameters
    target_sr = 16000  # Lower sample rate for faster processing
    silence_threshold = 0.02
    
    print("Starting simple audio preprocessing...")
    print(f"Target sample rate: {target_sr} Hz")
    
    processed_count = {"PD": 0, "HC": 0}
    
    for cohort in ["PD", "HC"]:
        input_dir = os.path.join(input_base, cohort)
        output_dir = os.path.join(output_base, cohort)
        
        if not os.path.exists(input_dir):
            continue
            
        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        print(f"\nProcessing {len(audio_files)} {cohort} files...")
        
        for i, filename in enumerate(audio_files):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"clean_{filename}")
                
                # Load audio
                audio, sr = librosa.load(input_path, sr=target_sr)
                
                # Simple preprocessing
                processed_audio = preprocess_single_audio(audio, sr, silence_threshold)
                
                # Save
                sf.write(output_path, processed_audio, sr)
                processed_count[cohort] += 1
                
                print(f"  Processed: {filename} ({i+1}/{len(audio_files)})")
                
            except Exception as e:
                print(f"  Error with {filename}: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"PD files: {processed_count['PD']}")
    print(f"HC files: {processed_count['HC']}")
    print(f"Total: {sum(processed_count.values())}")
    print(f"Output directory: {os.path.abspath(output_base)}")

def preprocess_single_audio(audio, sr, silence_threshold=0.02):
    """
    Preprocess a single audio file
    """
    # Step 1: High-pass filter to remove low frequency noise
    b, a = butter(3, 80 / (sr / 2), btype='high')
    audio = filtfilt(b, a, audio)
    
    # Step 2: Remove silence
    audio = remove_silence_simple(audio, sr, silence_threshold)
    
    # Step 3: Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Step 4: Ensure minimum length
    min_length = int(0.5 * sr)  # 0.5 seconds
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)))
    
    return audio

def remove_silence_simple(audio, sr, threshold=0.02):
    """
    Simple silence removal using moving average energy
    """
    # Frame parameters
    frame_size = int(0.025 * sr)  # 25ms
    hop_size = int(0.01 * sr)     # 10ms
    
    # Calculate energy for each frame
    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    if not energies:
        return audio
    
    energies = np.array(energies)
    
    # Normalize energies
    max_energy = np.max(energies)
    if max_energy > 0:
        energies = energies / max_energy
    
    # Find voice regions
    voice_frames = energies > threshold
    
    # Extract voice segments
    voice_audio = []
    for i, is_voice in enumerate(voice_frames):
        if is_voice:
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            voice_audio.extend(audio[start:end])
    
    return np.array(voice_audio) if voice_audio else audio

if __name__ == "__main__":
    simple_audio_preprocessing()

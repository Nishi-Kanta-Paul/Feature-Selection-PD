#!/usr/bin/env python3
"""
COMPREHENSIVE PD VOICE FEATURE EXTRACTION
=========================================
Implements all key features for Parkinson's Disease detection from audio
Based on research literature and MDVP standards
"""

import os
import wave
import struct
import numpy as np
import csv
import statistics
import math
from datetime import datetime

class AudioLoader:
    """Load 16kHz preprocessed audio files"""
    
    def load_wav_file(self, filepath):
        """Load WAV file and return signal + metadata"""
        try:
            with wave.open(filepath, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                raw_data = wav_file.readframes(frames)
                
                if wav_file.getsampwidth() == 2:  # 16-bit
                    signal = struct.unpack(f'<{frames}h', raw_data)
                    signal = [sample / 32768.0 for sample in signal]
                else:
                    return None
                
                return {
                    'signal': signal,
                    'sample_rate': sample_rate,
                    'duration': frames / sample_rate
                }
                
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            return None

class VoiceAnalyzer:
    """Extract comprehensive voice features for PD detection"""
    
    def __init__(self):
        self.loader = AudioLoader()
    
    def get_voiced_segments(self, signal, sr, frame_length=400, hop_length=160):
        """
        Extract voiced segments using energy and ZCR thresholds
        frame_length=25ms at 16kHz, hop_length=10ms
        """
        # Frame the signal
        frames = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            frames.append(frame)
        
        voiced_frames = []
        periods = []
        f0_values = []
        
        for frame in frames:
            # Calculate energy
            energy = sum(x*x for x in frame)
            
            # Calculate zero crossing rate
            zcr = sum(1 for i in range(len(frame)-1) if frame[i]*frame[i+1] < 0) / len(frame)
            
            # Simple voiced detection (energy > threshold, low ZCR)
            if energy > 0.01 and zcr < 0.3:
                voiced_frames.append(frame)
                
                # Pitch detection using autocorrelation
                f0, period = self.estimate_pitch(frame, sr)
                if f0 > 0:
                    f0_values.append(f0)
                    periods.append(period)
        
        return voiced_frames, periods, f0_values
    
    def estimate_pitch(self, frame, sr, min_f0=70, max_f0=400):
        """Estimate pitch using autocorrelation"""
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Search in valid pitch range
        min_lag = int(sr / max_f0)
        max_lag = int(sr / min_f0)
        
        if max_lag < len(autocorr):
            search_range = autocorr[min_lag:max_lag]
            if len(search_range) > 0 and max(search_range) > 0.3 * autocorr[0]:
                peak_idx = np.argmax(search_range) + min_lag
                f0 = sr / peak_idx
                period = peak_idx / sr
                return f0, period
        
        return 0, 0
    
    def calculate_jitter_features(self, periods):
        """Calculate all jitter-based features (MDVP standards)"""
        if len(periods) < 3:
            return {
                'jitter_percent': 0, 'jitter_abs': 0, 'jitter_rap': 0,
                'jitter_ppq': 0, 'jitter_ddp': 0
            }
        
        periods = np.array(periods)
        mean_period = np.mean(periods)
        
        # MDVP: Jitter (%)
        period_diffs = np.abs(np.diff(periods))
        jitter_percent = (np.mean(period_diffs) / mean_period) * 100
        
        # MDVP: Jitter (Abs) - in microseconds
        jitter_abs = np.mean(period_diffs) * 1000000
        
        # MDVP: RAP (Relative Average Perturbation)
        rap_values = []
        for i in range(1, len(periods)-1):
            local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
            if local_mean > 0:
                rap_values.append(abs(periods[i] - local_mean) / local_mean)
        jitter_rap = (np.mean(rap_values) * 100) if rap_values else 0
        
        # MDVP: PPQ (5-point Period Perturbation Quotient)
        ppq_values = []
        for i in range(2, len(periods)-2):
            local_mean = np.mean(periods[i-2:i+3])
            if local_mean > 0:
                ppq_values.append(abs(periods[i] - local_mean) / local_mean)
        jitter_ppq = (np.mean(ppq_values) * 100) if ppq_values else 0
        
        # Jitter: DDP (Average absolute difference of differences)
        if len(period_diffs) > 1:
            ddp_values = np.abs(np.diff(period_diffs))
            jitter_ddp = (np.mean(ddp_values) / mean_period) * 100
        else:
            jitter_ddp = 0
        
        return {
            'jitter_percent': jitter_percent,
            'jitter_abs': jitter_abs,
            'jitter_rap': jitter_rap,
            'jitter_ppq': jitter_ppq,
            'jitter_ddp': jitter_ddp
        }
    
    def calculate_shimmer_features(self, voiced_frames, sr):
        """Calculate all shimmer-based features (MDVP standards)"""
        if len(voiced_frames) < 3:
            return {
                'shimmer_percent': 0, 'shimmer_db': 0, 'shimmer_apq3': 0,
                'shimmer_apq5': 0, 'shimmer_apq': 0, 'shimmer_dda': 0
            }
        
        # Calculate RMS amplitude for each frame
        amplitudes = []
        for frame in voiced_frames:
            rms = np.sqrt(np.mean(np.square(frame)))
            if rms > 0:
                amplitudes.append(rms)
        
        if len(amplitudes) < 3:
            return {
                'shimmer_percent': 0, 'shimmer_db': 0, 'shimmer_apq3': 0,
                'shimmer_apq5': 0, 'shimmer_apq': 0, 'shimmer_dda': 0
            }
        
        amplitudes = np.array(amplitudes)
        mean_amplitude = np.mean(amplitudes)
        
        # MDVP: Shimmer (%)
        amp_diffs = np.abs(np.diff(amplitudes))
        shimmer_percent = (np.mean(amp_diffs) / mean_amplitude) * 100
        
        # MDVP: Shimmer (dB)
        shimmer_db = 20 * np.log10(1 + shimmer_percent/100) if shimmer_percent > 0 else 0
        
        # Shimmer: APQ3 (3-point Amplitude Perturbation Quotient)
        apq3_values = []
        for i in range(1, len(amplitudes)-1):
            local_mean = np.mean(amplitudes[i-1:i+2])
            if local_mean > 0:
                apq3_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq3 = (np.mean(apq3_values) * 100) if apq3_values else 0
        
        # Shimmer: APQ5 (5-point Amplitude Perturbation Quotient)
        apq5_values = []
        for i in range(2, len(amplitudes)-2):
            local_mean = np.mean(amplitudes[i-2:i+3])
            if local_mean > 0:
                apq5_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq5 = (np.mean(apq5_values) * 100) if apq5_values else 0
        
        # MDVP: APQ (General Amplitude Perturbation Quotient)
        shimmer_apq = shimmer_apq5  # Often same as APQ5
        
        # Shimmer: DDA (Average absolute difference of differences)
        if len(amp_diffs) > 1:
            dda_values = np.abs(np.diff(amp_diffs))
            shimmer_dda = (np.mean(dda_values) / mean_amplitude) * 100
        else:
            shimmer_dda = 0
        
        return {
            'shimmer_percent': shimmer_percent,
            'shimmer_db': shimmer_db,
            'shimmer_apq3': shimmer_apq3,
            'shimmer_apq5': shimmer_apq5,
            'shimmer_apq': shimmer_apq,
            'shimmer_dda': shimmer_dda
        }
    
    def calculate_noise_features(self, voiced_frames, f0_values, sr):
        """Calculate HNR and NHR features"""
        if len(voiced_frames) == 0 or len(f0_values) == 0:
            return {'hnr': 0, 'nhr': 0}
        
        hnr_values = []
        nhr_values = []
        
        for i, frame in enumerate(voiced_frames[:len(f0_values)]):
            if i >= len(f0_values):
                break
                
            f0 = f0_values[i]
            if f0 <= 0:
                continue
            
            # FFT for harmonic analysis
            fft = np.fft.fft(frame, n=len(frame))
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(frame), 1/sr)
            
            # Find harmonic and noise components
            harmonic_power = 0
            noise_power = 0
            
            # Simple harmonic detection (peaks near F0 harmonics)
            for h in range(1, 6):  # First 5 harmonics
                target_freq = f0 * h
                if target_freq < sr/2:
                    # Find closest frequency bin
                    freq_idx = np.argmin(np.abs(freqs[:len(freqs)//2] - target_freq))
                    # Sum power in small window around harmonic
                    window = 3  # ±3 bins
                    start_idx = max(0, freq_idx - window)
                    end_idx = min(len(magnitude)//2, freq_idx + window + 1)
                    harmonic_power += np.sum(magnitude[start_idx:end_idx]**2)
            
            # Noise power (total - harmonic estimate)
            total_power = np.sum(magnitude[:len(magnitude)//2]**2)
            noise_power = max(0.001, total_power - harmonic_power)
            harmonic_power = max(0.001, harmonic_power)
            
            # Calculate HNR and NHR
            hnr = 10 * np.log10(harmonic_power / noise_power)
            nhr = noise_power / (harmonic_power + noise_power)
            
            hnr_values.append(hnr)
            nhr_values.append(nhr)
        
        return {
            'hnr': np.mean(hnr_values) if hnr_values else 0,
            'nhr': np.mean(nhr_values) if nhr_values else 0
        }
    
    def calculate_prosodic_features(self, f0_values):
        """Calculate prosodic features (F0 statistics)"""
        if len(f0_values) == 0:
            return {
                'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0,
                'f0_range': 0, 'f0_cv': 0
            }
        
        f0_array = np.array(f0_values)
        
        return {
            'f0_mean': np.mean(f0_array),
            'f0_std': np.std(f0_array),
            'f0_min': np.min(f0_array),
            'f0_max': np.max(f0_array),
            'f0_range': np.max(f0_array) - np.min(f0_array),
            'f0_cv': (np.std(f0_array) / np.mean(f0_array)) * 100 if np.mean(f0_array) > 0 else 0
        }
    
    def calculate_nonlinear_features(self, f0_values, signal):
        """Calculate nonlinear complexity features"""
        if len(f0_values) < 10:
            return {
                'rpde': 0, 'd2': 0, 'dfa': 0, 'spread1': 0, 'spread2': 0, 'ppe': 0
            }
        
        f0_array = np.array(f0_values)
        
        # RPDE (simplified approximation)
        # Calculate period density entropy
        if len(f0_array) > 0:
            periods = 1.0 / f0_array
            period_diff = np.diff(periods)
            if len(period_diff) > 0:
                rpde = -np.sum((period_diff**2) * np.log(period_diff**2 + 1e-10)) / len(period_diff)
            else:
                rpde = 0
        else:
            rpde = 0
        
        # D2 (Correlation Dimension - simplified)
        # Approximate using F0 variability
        d2 = np.std(f0_array) / np.mean(f0_array) if np.mean(f0_array) > 0 else 0
        
        # DFA (Detrended Fluctuation Analysis - simplified)
        # Using F0 sequence
        if len(f0_array) > 4:
            # Simple scaling exponent approximation
            dfa = np.log(np.std(f0_array)) / np.log(len(f0_array))
        else:
            dfa = 0
        
        # Spread measures (F0 variability measures)
        spread1 = np.std(f0_array) if len(f0_array) > 1 else 0
        spread2 = np.var(f0_array) if len(f0_array) > 1 else 0
        
        # PPE (Pitch Period Entropy)
        if len(f0_array) > 1:
            periods = 1.0 / f0_array
            period_hist, _ = np.histogram(periods, bins=min(10, len(periods)//2))
            period_hist = period_hist + 1e-10  # Avoid log(0)
            period_prob = period_hist / np.sum(period_hist)
            ppe = -np.sum(period_prob * np.log2(period_prob))
        else:
            ppe = 0
        
        return {
            'rpde': rpde,
            'd2': d2,
            'dfa': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'ppe': ppe
        }
    
    def calculate_additional_features(self, signal, voiced_frames, sr):
        """Calculate additional signal processing features"""
        # Short-Term Energy
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.01 * sr)     # 10ms
        
        energy_values = []
        zcr_values = []
        
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            
            # Short-term energy
            ste = np.sum(np.square(frame))
            energy_values.append(ste)
            
            # Zero crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zcr_values.append(zcr)
        
        # Voice activity ratio
        voiced_ratio = len(voiced_frames) / max(1, len(energy_values))
        
        return {
            'ste_mean': np.mean(energy_values) if energy_values else 0,
            'ste_std': np.std(energy_values) if len(energy_values) > 1 else 0,
            'zcr_mean': np.mean(zcr_values) if zcr_values else 0,
            'zcr_std': np.std(zcr_values) if len(zcr_values) > 1 else 0,
            'voiced_ratio': voiced_ratio
        }
    
    def extract_all_features(self, filepath):
        """Extract all comprehensive features from audio file"""
        print(f"Processing: {os.path.basename(filepath)}")
        
        # Load audio
        audio_data = self.loader.load_wav_file(filepath)
        if audio_data is None:
            print("  Failed to load audio")
            return None
        
        signal = np.array(audio_data['signal'])
        sr = audio_data['sample_rate']
        
        print(f"  Duration: {audio_data['duration']:.2f}s, Sample Rate: {sr}Hz")
        
        try:
            # Extract voiced segments and pitch
            voiced_frames, periods, f0_values = self.get_voiced_segments(signal, sr)
            print(f"  Found {len(voiced_frames)} voiced frames, {len(f0_values)} F0 values")
            
            # Extract all feature categories
            features = {}
            
            # 1. Jitter features
            jitter_features = self.calculate_jitter_features(periods)
            features.update(jitter_features)
            
            # 2. Shimmer features
            shimmer_features = self.calculate_shimmer_features(voiced_frames, sr)
            features.update(shimmer_features)
            
            # 3. Noise features
            noise_features = self.calculate_noise_features(voiced_frames, f0_values, sr)
            features.update(noise_features)
            
            # 4. Prosodic features
            prosodic_features = self.calculate_prosodic_features(f0_values)
            features.update(prosodic_features)
            
            # 5. Nonlinear features
            nonlinear_features = self.calculate_nonlinear_features(f0_values, signal)
            features.update(nonlinear_features)
            
            # 6. Additional features
            additional_features = self.calculate_additional_features(signal, voiced_frames, sr)
            features.update(additional_features)
            
            # Add metadata
            features.update({
                'filename': os.path.basename(filepath),
                'duration': audio_data['duration'],
                'sample_rate': sr,
                'num_voiced_frames': len(voiced_frames),
                'num_f0_values': len(f0_values)
            })
            
            print(f"  Extracted {len(features)} comprehensive features")
            return features
            
        except Exception as e:
            print(f"  Feature extraction error: {e}")
            return None

def main():
    """Main comprehensive feature extraction pipeline"""
    
    print("COMPREHENSIVE PD VOICE FEATURE EXTRACTION")
    print("=" * 50)
    print("Implementing all key features from research literature")
    print("Categories: Jitter, Shimmer, Noise, Prosodic, Nonlinear")
    print()
    
    analyzer = VoiceAnalyzer()
    all_results = []
    
    # Process HC and PD files
    for group, dirname in [('HC', 'preprocessed_data_percentile_1_99/HC'), 
                          ('PD', 'preprocessed_data_percentile_1_99/PD')]:
        
        print(f"Processing {group} files")
        print("-" * 30)
        
        if not os.path.exists(dirname):
            print(f"Directory not found: {dirname}")
            continue
        
        files = [f for f in os.listdir(dirname) if f.endswith('.wav')]
        print(f"Found {len(files)} files")
        
        # Process first 10 files of each group for testing
        for i, filename in enumerate(files[:10], 1):
            filepath = os.path.join(dirname, filename)
            print(f"[{i}/10] {filename}")
            
            features = analyzer.extract_all_features(filepath)
            if features:
                result = {'group': group, **features}
                all_results.append(result)
            print()
    
    # Save results
    print("Saving comprehensive results...")
    
    os.makedirs("comprehensive_features", exist_ok=True)
    
    if all_results:
        # Save CSV with all features
        csv_file = "comprehensive_features/pd_features_comprehensive.csv"
        with open(csv_file, "w", newline='') as f:
            if all_results:
                fieldnames = list(all_results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
        
        print(f"✅ Comprehensive features saved to: {csv_file}")
        
        # Print detailed summary
        hc_results = [r for r in all_results if r['group'] == 'HC']
        pd_results = [r for r in all_results if r['group'] == 'PD']
        
        print(f"\nCOMPREHENSIVE FEATURE SUMMARY:")
        print(f"Processed: {len(hc_results)} HC, {len(pd_results)} PD files")
        print(f"Features per file: {len(all_results[0])-1} (excluding group)")
        
        if hc_results and pd_results:
            print(f"\nKEY FEATURE COMPARISONS:")
            
            # Jitter features
            hc_jitter = [r['jitter_percent'] for r in hc_results]
            pd_jitter = [r['jitter_percent'] for r in pd_results]
            print(f"Jitter (%):     HC={statistics.mean(hc_jitter):.3f} ± {statistics.stdev(hc_jitter):.3f}, PD={statistics.mean(pd_jitter):.3f} ± {statistics.stdev(pd_jitter):.3f}")
            
            # Shimmer features
            hc_shimmer = [r['shimmer_percent'] for r in hc_results]
            pd_shimmer = [r['shimmer_percent'] for r in pd_results]
            print(f"Shimmer (%):    HC={statistics.mean(hc_shimmer):.3f} ± {statistics.stdev(hc_shimmer):.3f}, PD={statistics.mean(pd_shimmer):.3f} ± {statistics.stdev(pd_shimmer):.3f}")
            
            # HNR/NHR
            hc_hnr = [r['hnr'] for r in hc_results]
            pd_hnr = [r['hnr'] for r in pd_results]
            print(f"HNR (dB):       HC={statistics.mean(hc_hnr):.3f} ± {statistics.stdev(hc_hnr):.3f}, PD={statistics.mean(pd_hnr):.3f} ± {statistics.stdev(pd_hnr):.3f}")
            
            # F0 features
            hc_f0 = [r['f0_mean'] for r in hc_results]
            pd_f0 = [r['f0_mean'] for r in pd_results]
            print(f"F0 Mean (Hz):   HC={statistics.mean(hc_f0):.1f} ± {statistics.stdev(hc_f0):.1f}, PD={statistics.mean(pd_f0):.1f} ± {statistics.stdev(pd_f0):.1f}")
            
            # Nonlinear features
            hc_ppe = [r['ppe'] for r in hc_results]
            pd_ppe = [r['ppe'] for r in pd_results]
            print(f"PPE:            HC={statistics.mean(hc_ppe):.3f} ± {statistics.stdev(hc_ppe):.3f}, PD={statistics.mean(pd_ppe):.3f} ± {statistics.stdev(pd_ppe):.3f}")
    
    print("\nComprehensive PD feature extraction complete!")
    print("Features extracted:")
    print("• Jitter: jitter_percent, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp")
    print("• Shimmer: shimmer_percent, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_dda")
    print("• Noise: hnr, nhr")
    print("• Prosodic: f0_mean, f0_std, f0_min, f0_max, f0_range, f0_cv")
    print("• Nonlinear: rpde, d2, dfa, spread1, spread2, ppe")
    print("• Additional: ste_mean, ste_std, zcr_mean, zcr_std, voiced_ratio")

if __name__ == "__main__":
    main()
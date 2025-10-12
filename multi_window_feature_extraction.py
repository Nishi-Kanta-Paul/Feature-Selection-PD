#!/usr/bin/env python3
"""
MULTI-WINDOW SIZE PD VOICE FEATURE EXTRACTION
=============================================
Feature extraction with multiple window sizes: 5ms, 10ms, 20ms
Implements ALL requested features with different temporal resolutions
"""

import os
import wave
import struct
import numpy as np
import csv
import statistics
import math
from datetime import datetime
from scipy import signal
from scipy.fft import fft, dct, fftfreq
from scipy.signal import welch, spectrogram, butter, filtfilt, stft
import warnings
warnings.filterwarnings('ignore')


class WindowedAudioLoader:
    """Audio loader for windowed analysis"""

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


class MultiWindowVoiceAnalyzer:
    """Voice analyzer with multiple window sizes"""

    def __init__(self, window_size_ms=20, hop_ratio=0.5):
        """
        Initialize with windowing parameters
        window_size_ms: Window size in milliseconds (5ms, 10ms, or 20ms)
        hop_ratio: Hop size as ratio of window size (default 0.5 = 50% overlap)
        """
        self.loader = WindowedAudioLoader()
        self.window_size_ms = window_size_ms
        self.hop_size_ms = int(window_size_ms * hop_ratio)

    def resample_if_needed(self, signal, original_sr, target_sr=16000):
        """Resample signal if needed"""
        if original_sr == target_sr:
            return signal, original_sr

        resample_factor = target_sr / original_sr

        if resample_factor < 1:
            # Downsample
            step = int(1 / resample_factor)
            resampled = signal[::step]
        else:
            # Upsample (simple linear interpolation)
            new_length = int(len(signal) * resample_factor)
            old_indices = np.linspace(0, len(signal) - 1, len(signal))
            new_indices = np.linspace(0, len(signal) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, signal)

        return resampled, target_sr

    def apply_digital_filtering(self, signal, sr):
        """Apply digital filtering (preemphasis and bandpass)"""
        # Preemphasis filter
        preemphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

        # Bandpass filter for voice frequencies (80-8000 Hz)
        try:
            nyquist = sr / 2
            low = 80 / nyquist
            high = min(8000 / nyquist, 0.99)
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, preemphasized)
            return filtered
        except:
            return preemphasized

    def create_windows(self, signal, sr):
        """Create overlapping windows from signal"""
        window_size = int(self.window_size_ms * sr / 1000)
        hop_size = int(self.hop_size_ms * sr / 1000)

        # Ensure minimum window size
        if window_size < 50:  # At least 50 samples
            window_size = 50
        if hop_size < 25:
            hop_size = 25

        windows = []
        window_times = []

        for i in range(0, len(signal) - window_size + 1, hop_size):
            window = signal[i:i + window_size]
            # Apply Hamming window
            hamming_window = np.hamming(len(window))
            windowed = window * hamming_window
            windows.append(windowed)
            window_times.append(i / sr)  # Time stamp

        return windows, window_times

    def detect_voiced_windows(self, windows, sr):
        """Detect voiced windows using energy and ZCR thresholds"""
        voiced_windows = []
        voiced_indices = []

        for i, window in enumerate(windows):
            # Calculate energy
            energy = np.sum(window**2)

            # Calculate zero crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))

            # Voice activity detection (adjusted thresholds for smaller windows)
            energy_threshold = 0.005 if self.window_size_ms < 20 else 0.01
            zcr_threshold = 0.3

            if energy > energy_threshold and zcr < zcr_threshold:
                voiced_windows.append(window)
                voiced_indices.append(i)

        return voiced_windows, voiced_indices

    def estimate_f0_window(self, window, sr):
        """Estimate F0 for a single window using autocorrelation"""
        if len(window) < 50:  # Minimum samples needed
            return 0, 0

        # Autocorrelation
        autocorr = np.correlate(window, window, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Search for pitch
        min_f0, max_f0 = 50, 500
        min_lag = max(1, int(sr / max_f0))
        max_lag = min(len(autocorr) - 1, int(sr / min_f0))

        if max_lag > min_lag:
            search_range = autocorr[min_lag:max_lag]
            if len(search_range) > 0 and np.max(search_range) > 0.3:
                peak_idx = np.argmax(search_range) + min_lag
                f0 = sr / peak_idx
                period = peak_idx / sr
                return f0, period

        return 0, 0

    def extract_f0_and_periods(self, voiced_windows, sr):
        """Extract F0 values and periods from voiced windows"""
        f0_values = []
        periods = []
        amplitudes = []

        for window in voiced_windows:
            f0, period = self.estimate_f0_window(window, sr)
            if f0 > 0 and 50 <= f0 <= 500:
                f0_values.append(f0)
                periods.append(period)
                amplitudes.append(np.sqrt(np.sum(window**2)))

        return f0_values, periods, amplitudes

    def calculate_windowed_jitter_features(self, periods):
        """Calculate MDVP Jitter features from periods"""
        if len(periods) < 3:
            return {
                'mdvp_jitter_percent': 0, 'mdvp_jitter_abs': 0, 'mdvp_rap': 0,
                'mdvp_ppq': 0, 'jitter_ddp': 0
            }

        periods = np.array(periods)
        mean_period = np.mean(periods)

        if mean_period == 0:
            return {
                'mdvp_jitter_percent': 0, 'mdvp_jitter_abs': 0, 'mdvp_rap': 0,
                'mdvp_ppq': 0, 'jitter_ddp': 0
            }

        # MDVP: Jitter (%)
        period_diffs = np.abs(np.diff(periods))
        mdvp_jitter_percent = (np.mean(period_diffs) / mean_period) * 100

        # MDVP: Jitter (Abs) - in microseconds
        mdvp_jitter_abs = np.mean(period_diffs) * 1000000

        # MDVP: RAP (Relative Average Perturbation)
        rap_values = []
        for i in range(1, len(periods)-1):
            local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
            if local_mean > 0:
                rap_values.append(abs(periods[i] - local_mean) / local_mean)
        mdvp_rap = (np.mean(rap_values) * 100) if rap_values else 0

        # MDVP: PPQ (Five-point Period Perturbation Quotient)
        ppq_values = []
        for i in range(2, len(periods)-2):
            local_mean = np.mean(periods[i-2:i+3])
            if local_mean > 0:
                ppq_values.append(abs(periods[i] - local_mean) / local_mean)
        mdvp_ppq = (np.mean(ppq_values) * 100) if ppq_values else 0

        # Jitter: DDP
        if len(period_diffs) > 1:
            ddp_values = np.abs(np.diff(period_diffs))
            jitter_ddp = (np.mean(ddp_values) / mean_period) * 100
        else:
            jitter_ddp = 0

        return {
            'mdvp_jitter_percent': mdvp_jitter_percent,
            'mdvp_jitter_abs': mdvp_jitter_abs,
            'mdvp_rap': mdvp_rap,
            'mdvp_ppq': mdvp_ppq,
            'jitter_ddp': jitter_ddp
        }

    def calculate_windowed_shimmer_features(self, amplitudes):
        """Calculate MDVP Shimmer features from amplitudes"""
        if len(amplitudes) < 3:
            return {
                'mdvp_shimmer_percent': 0, 'mdvp_shimmer_db': 0, 'shimmer_apq3': 0,
                'shimmer_apq5': 0, 'mdvp_apq': 0, 'shimmer_dda': 0
            }

        amplitudes = np.array(amplitudes)
        mean_amplitude = np.mean(amplitudes)

        if mean_amplitude == 0:
            return {
                'mdvp_shimmer_percent': 0, 'mdvp_shimmer_db': 0, 'shimmer_apq3': 0,
                'shimmer_apq5': 0, 'mdvp_apq': 0, 'shimmer_dda': 0
            }

        # MDVP: Shimmer (%)
        amp_diffs = np.abs(np.diff(amplitudes))
        mdvp_shimmer_percent = (np.mean(amp_diffs) / mean_amplitude) * 100

        # MDVP: Shimmer (dB)
        mdvp_shimmer_db = 20 * \
            np.log10(1 + mdvp_shimmer_percent /
                     100) if mdvp_shimmer_percent > 0 else 0

        # Shimmer: APQ3 (3-point Amplitude Perturbation Quotient)
        apq3_values = []
        for i in range(1, len(amplitudes)-1):
            local_mean = np.mean(amplitudes[i-1:i+2])
            if local_mean > 0:
                apq3_values.append(
                    abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq3 = (np.mean(apq3_values) * 100) if apq3_values else 0

        # Shimmer: APQ5 (5-point Amplitude Perturbation Quotient)
        apq5_values = []
        for i in range(2, len(amplitudes)-2):
            local_mean = np.mean(amplitudes[i-2:i+3])
            if local_mean > 0:
                apq5_values.append(
                    abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq5 = (np.mean(apq5_values) * 100) if apq5_values else 0

        # MDVP: APQ (General Amplitude Perturbation Quotient)
        mdvp_apq = shimmer_apq5

        # Shimmer: DDA (Average absolute difference of consecutive amplitude differences)
        if len(amp_diffs) > 1:
            dda_values = np.abs(np.diff(amp_diffs))
            shimmer_dda = (np.mean(dda_values) / mean_amplitude) * 100
        else:
            shimmer_dda = 0

        return {
            'mdvp_shimmer_percent': mdvp_shimmer_percent,
            'mdvp_shimmer_db': mdvp_shimmer_db,
            'shimmer_apq3': shimmer_apq3,
            'shimmer_apq5': shimmer_apq5,
            'mdvp_apq': mdvp_apq,
            'shimmer_dda': shimmer_dda
        }

    def calculate_windowed_noise_features(self, voiced_windows, f0_values, sr):
        """Calculate NHR and HNR features from windowed analysis"""
        if len(voiced_windows) == 0 or len(f0_values) == 0:
            return {'nhr': 0, 'hnr': 0}

        hnr_values = []
        nhr_values = []

        for i, window in enumerate(voiced_windows):
            if i >= len(f0_values):
                break

            f0 = f0_values[i]
            if f0 <= 0:
                continue

            # FFT analysis
            n_fft = max(256, len(window))  # Reduced for smaller windows
            fft_window = fft(window, n=n_fft)
            magnitude = np.abs(fft_window[:n_fft//2])
            freqs = fftfreq(n_fft, 1/sr)[:n_fft//2]

            # Harmonic detection
            harmonic_power = 0
            total_power = np.sum(magnitude**2)

            # Detect first 7 harmonics
            for h in range(1, 8):
                target_freq = f0 * h
                if target_freq < sr/2:
                    freq_idx = np.argmin(np.abs(freqs - target_freq))
                    window_bins = max(1, int(0.1 * f0 * n_fft / sr))
                    start_idx = max(0, freq_idx - window_bins)
                    end_idx = min(len(magnitude), freq_idx + window_bins + 1)
                    harmonic_power += np.sum(magnitude[start_idx:end_idx]**2)

            # Calculate noise power
            noise_power = max(0.001, total_power - harmonic_power)
            harmonic_power = max(0.001, harmonic_power)

            # HNR and NHR
            hnr = 10 * np.log10(harmonic_power / noise_power)
            nhr = noise_power / (harmonic_power + noise_power)

            hnr_values.append(hnr)
            nhr_values.append(nhr)

        return {
            'nhr': np.mean(nhr_values) if nhr_values else 0,
            'hnr': np.mean(hnr_values) if hnr_values else 0
        }

    def calculate_windowed_prosodic_features(self, f0_values):
        """Calculate frequency-based prosodic features"""
        if len(f0_values) == 0:
            return {
                'mdvp_fo': 0, 'mdvp_fhi': 0, 'mdvp_flo': 0,
                'f0_std': 0, 'f0_range': 0, 'f0_cv': 0
            }

        f0_array = np.array(f0_values)

        return {
            'mdvp_fo': np.mean(f0_array),      # Mean fundamental frequency
            'mdvp_fhi': np.max(f0_array),      # Maximum fundamental frequency
            'mdvp_flo': np.min(f0_array),      # Minimum fundamental frequency
            'f0_std': np.std(f0_array),
            'f0_range': np.max(f0_array) - np.min(f0_array),
            'f0_cv': (np.std(f0_array) / np.mean(f0_array)) * 100 if np.mean(f0_array) > 0 else 0
        }

    def calculate_windowed_nonlinear_features(self, f0_values):
        """Calculate nonlinear dynamical complexity metrics"""
        if len(f0_values) < 5:
            return {
                'rpde': 0, 'd2': 0, 'dfa': 0, 'spread1': 0, 'spread2': 0, 'ppe': 0
            }

        f0_array = np.array(f0_values)

        # RPDE (Recurrence Period Density Entropy) - Simplified
        periods = 1.0 / f0_array
        if len(periods) > 5:
            period_diffs = np.diff(periods)
            if len(period_diffs) > 0 and np.var(period_diffs) > 0:
                rpde = np.var(period_diffs) / (np.mean(periods)**2)
            else:
                rpde = 0
        else:
            rpde = 0

        # D2 (Correlation dimension) - Approximation
        d2 = np.std(f0_array) / \
            np.mean(f0_array) if np.mean(f0_array) > 0 else 0

        # DFA (Detrended Fluctuation Analysis) - Simplified
        if len(f0_array) > 10:
            y = np.cumsum(f0_array - np.mean(f0_array))
            scales = [4, 8, 16]
            fluctuations = []

            for scale in scales:
                if scale < len(y) // 2:
                    n_segments = len(y) // scale
                    local_fluct = []

                    for i in range(n_segments):
                        start = i * scale
                        end = start + scale
                        segment = y[start:end]

                        if len(segment) > 1:
                            coeffs = np.polyfit(
                                range(len(segment)), segment, 1)
                            trend = np.polyval(coeffs, range(len(segment)))
                            detrended = segment - trend
                            local_fluct.append(np.std(detrended))

                    if local_fluct:
                        fluctuations.append(np.mean(local_fluct))

            if len(fluctuations) > 1:
                try:
                    log_scales = np.log(scales[:len(fluctuations)])
                    log_flucts = np.log(np.array(fluctuations) + 1e-10)
                    dfa = np.polyfit(log_scales, log_flucts, 1)[0]
                except:
                    dfa = 0
            else:
                dfa = 0
        else:
            dfa = 0

        # Spread1 and Spread2
        spread1 = np.std(f0_array)
        spread2 = np.var(f0_array)

        # PPE (Pitch Period Entropy)
        if len(f0_array) > 3:
            periods = 1.0 / f0_array
            n_bins = min(10, len(periods) // 2)
            if n_bins > 1:
                hist, _ = np.histogram(periods, bins=n_bins)
                hist = hist + 1e-10
                prob = hist / np.sum(hist)
                prob = prob[prob > 1e-10]
                ppe = -np.sum(prob * np.log2(prob))
            else:
                ppe = 0
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

    def calculate_windowed_signal_processing_features(self, windows, sr):
        """Calculate additional signal processing features"""
        if len(windows) == 0:
            return {
                'ste_mean': 0, 'ste_std': 0, 'zcr_mean': 0, 'zcr_std': 0,
                'spectral_entropy_mean': 0, 'spectral_entropy_std': 0
            }

        # Short-Term Energy (STE)
        ste_values = []
        for window in windows:
            ste = np.sum(window**2)
            ste_values.append(ste)

        # Zero-Crossing Rate (ZCR)
        zcr_values = []
        for window in windows:
            zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))
            zcr_values.append(zcr)

        # Spectral Entropy
        spectral_entropies = []
        for window in windows:
            # FFT
            n_fft = min(512, max(256, len(window) * 2))
            fft_window = fft(window, n=n_fft)
            magnitude = np.abs(fft_window[:n_fft//2])

            if np.sum(magnitude) > 0:
                # Normalize
                magnitude_norm = magnitude / np.sum(magnitude)
                magnitude_safe = magnitude_norm + 1e-10
                spectral_entropy = - \
                    np.sum(magnitude_safe * np.log2(magnitude_safe))
                spectral_entropies.append(spectral_entropy)

        return {
            'ste_mean': np.mean(ste_values),
            'ste_std': np.std(ste_values) if len(ste_values) > 1 else 0,
            'zcr_mean': np.mean(zcr_values),
            'zcr_std': np.std(zcr_values) if len(zcr_values) > 1 else 0,
            'spectral_entropy_mean': np.mean(spectral_entropies) if spectral_entropies else 0,
            'spectral_entropy_std': np.std(spectral_entropies) if len(spectral_entropies) > 1 else 0
        }

    def calculate_windowed_spectral_transformations(self, windows, sr):
        """Calculate spectral transformation features (FFT, DCT, STFT)"""
        if len(windows) == 0:
            return {
                'spectral_centroid_mean': 0, 'spectral_spread_mean': 0,
                'spectral_rolloff_mean': 0, 'spectral_flux_mean': 0,
                'dct_energy_mean': 0, 'stft_energy_mean': 0
            }

        spectral_centroids = []
        spectral_spreads = []
        spectral_rolloffs = []
        spectral_fluxes = []
        dct_energies = []
        stft_energies = []

        prev_magnitude = None

        for window in windows:
            # FFT analysis
            n_fft = min(512, max(256, len(window) * 2))
            fft_window = fft(window, n=n_fft)
            magnitude = np.abs(fft_window[:n_fft//2])
            freqs = np.linspace(0, sr/2, len(magnitude))

            if np.sum(magnitude) > 0:
                # Normalize
                magnitude_norm = magnitude / np.sum(magnitude)

                # Spectral Centroid
                centroid = np.sum(freqs * magnitude_norm)
                spectral_centroids.append(centroid)

                # Spectral Spread
                spread = np.sqrt(
                    np.sum(((freqs - centroid)**2) * magnitude_norm))
                spectral_spreads.append(spread)

                # Spectral Rolloff (85% energy)
                cumsum_mag = np.cumsum(magnitude_norm)
                rolloff_idx = np.where(cumsum_mag >= 0.85 * cumsum_mag[-1])[0]
                if len(rolloff_idx) > 0:
                    rolloff = freqs[rolloff_idx[0]]
                    spectral_rolloffs.append(rolloff)

                # Spectral Flux
                if prev_magnitude is not None and len(prev_magnitude) == len(magnitude):
                    flux = np.sum((magnitude - prev_magnitude)**2)
                    spectral_fluxes.append(flux)
                prev_magnitude = magnitude

            # DCT (Discrete Cosine Transform)
            dct_coeffs = dct(window, type=2, norm='ortho')
            dct_energy = np.sum(dct_coeffs**2)
            dct_energies.append(dct_energy)

            # STFT energy (simplified - using window energy)
            stft_energy = np.sum(window**2)
            stft_energies.append(stft_energy)

        return {
            'spectral_centroid_mean': np.mean(spectral_centroids) if spectral_centroids else 0,
            'spectral_spread_mean': np.mean(spectral_spreads) if spectral_spreads else 0,
            'spectral_rolloff_mean': np.mean(spectral_rolloffs) if spectral_rolloffs else 0,
            'spectral_flux_mean': np.mean(spectral_fluxes) if spectral_fluxes else 0,
            'dct_energy_mean': np.mean(dct_energies),
            'stft_energy_mean': np.mean(stft_energies)
        }

    def calculate_windowed_mfcc_features(self, windows, sr, n_mfcc=13):
        """Calculate MFCC features from windowed analysis"""
        if len(windows) == 0:
            mfcc_features = {}
            for i in range(n_mfcc):
                mfcc_features[f'mfcc_{i+1}_mean'] = 0
                mfcc_features[f'mfcc_{i+1}_std'] = 0
            return mfcc_features

        all_mfccs = []

        for window in windows:
            # FFT
            n_fft = min(512, max(256, len(window) * 2))
            fft_window = fft(window, n=n_fft)
            magnitude = np.abs(fft_window[:n_fft//2])

            # MFCC calculation
            mfccs = self.compute_mfcc_window(magnitude, sr, n_mfcc)
            if mfccs is not None:
                all_mfccs.append(mfccs)

        if not all_mfccs:
            mfcc_features = {}
            for i in range(n_mfcc):
                mfcc_features[f'mfcc_{i+1}_mean'] = 0
                mfcc_features[f'mfcc_{i+1}_std'] = 0
            return mfcc_features

        all_mfccs = np.array(all_mfccs)

        # Calculate statistics
        mfcc_features = {}
        for i in range(n_mfcc):
            if i < all_mfccs.shape[1]:
                mfcc_features[f'mfcc_{i+1}_mean'] = np.mean(all_mfccs[:, i])
                mfcc_features[f'mfcc_{i+1}_std'] = np.std(all_mfccs[:, i])
            else:
                mfcc_features[f'mfcc_{i+1}_mean'] = 0
                mfcc_features[f'mfcc_{i+1}_std'] = 0

        return mfcc_features

    def compute_mfcc_window(self, magnitude, sr, n_mfcc=13, n_mels=26):
        """Compute MFCC for a single window"""
        try:
            # Create mel filter bank
            mel_filters = self.create_mel_filterbank(
                len(magnitude), sr, n_mels)

            # Apply filters
            mel_spectrum = np.dot(mel_filters, magnitude**2)

            # Log spectrum
            log_mel = np.log(mel_spectrum + 1e-10)

            # DCT
            mfccs = dct(log_mel, type=2, norm='ortho')[:n_mfcc]

            return mfccs
        except:
            return None

    def create_mel_filterbank(self, nfft, sr, n_mels):
        """Create mel filter bank"""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)

        # Mel points
        low_freq_mel = 0
        high_freq_mel = hz_to_mel(sr // 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin points
        bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)

        # Filter bank
        filterbank = np.zeros((n_mels, nfft))

        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filterbank

    def extract_windowed_features(self, filepath):
        """Main function to extract all windowed features"""
        print(
            f"Processing: {os.path.basename(filepath)} [Window: {self.window_size_ms}ms]")

        # Load audio
        audio_data = self.loader.load_wav_file(filepath)
        if audio_data is None:
            print("  Failed to load audio")
            return None

        signal = np.array(audio_data['signal'])
        original_sr = audio_data['sample_rate']

        # Resample if needed
        if original_sr > 22050:
            signal, sr = self.resample_if_needed(signal, original_sr, 16000)
        else:
            sr = original_sr

        try:
            # Apply digital filtering
            filtered_signal = self.apply_digital_filtering(signal, sr)

            # Create overlapping windows
            windows, window_times = self.create_windows(filtered_signal, sr)

            # Detect voiced windows
            voiced_windows, voiced_indices = self.detect_voiced_windows(
                windows, sr)

            # Extract F0 and periods from voiced windows
            f0_values, periods, amplitudes = self.extract_f0_and_periods(
                voiced_windows, sr)

            # Initialize features dictionary
            features = {}

            # Calculate all feature categories
            jitter_features = self.calculate_windowed_jitter_features(periods)
            features.update(jitter_features)

            shimmer_features = self.calculate_windowed_shimmer_features(
                amplitudes)
            features.update(shimmer_features)

            noise_features = self.calculate_windowed_noise_features(
                voiced_windows, f0_values, sr)
            features.update(noise_features)

            prosodic_features = self.calculate_windowed_prosodic_features(
                f0_values)
            features.update(prosodic_features)

            nonlinear_features = self.calculate_windowed_nonlinear_features(
                f0_values)
            features.update(nonlinear_features)

            signal_features = self.calculate_windowed_signal_processing_features(
                windows, sr)
            features.update(signal_features)

            spectral_features = self.calculate_windowed_spectral_transformations(
                windows, sr)
            features.update(spectral_features)

            mfcc_features = self.calculate_windowed_mfcc_features(
                windows, sr, n_mfcc=13)
            features.update(mfcc_features)

            # Add metadata
            features.update({
                'filename': os.path.basename(filepath),
                'window_size_ms': self.window_size_ms,
                'hop_size_ms': self.hop_size_ms,
                'total_windows': len(windows),
                'voiced_windows': len(voiced_windows),
                'num_f0_values': len(f0_values)
            })

            print(
                f"  ✅ Windows: {len(windows)}, Voiced: {len(voiced_windows)}, F0: {len(f0_values)}")
            return features

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None


def process_with_multiple_window_sizes(filepath, group, window_sizes=[5, 10, 20]):
    """Process a file with multiple window sizes"""
    results = []

    for window_size in window_sizes:
        analyzer = MultiWindowVoiceAnalyzer(
            window_size_ms=window_size, hop_ratio=0.5)
        features = analyzer.extract_windowed_features(filepath)

        if features:
            result = {'group': group, **features}
            results.append(result)

    return results


def main():
    """Main multi-window feature extraction pipeline"""

    print("MULTI-WINDOW SIZE PD VOICE FEATURE EXTRACTION")
    print("=" * 70)
    print("🎯 WINDOW SIZES: 5ms, 10ms, 20ms")
    print("   Processing each file with all three window sizes")
    print()

    # Window sizes to test
    window_sizes = [5, 10, 20]  # milliseconds
    all_results = {ws: [] for ws in window_sizes}

    # Data directories to process
    data_paths = [
        ('HC', 'Processed_data_sample_raw_voice/rnnoise_out/0'),
        ('PD', 'Processed_data_sample_raw_voice/rnnoise_out/1'),
        ('SAMPLE', 'sample_audio_files')
    ]

    for group, dirname in data_paths:
        print(f"🔍 Processing {group} files from: {dirname}")
        print("-" * 50)

        if not os.path.exists(dirname):
            print(f"   Directory not found: {dirname}")
            continue

        if dirname == 'sample_audio_files':
            # Handle sample files directly
            files = [f for f in os.listdir(dirname) if f.endswith('.wav')]

            for filename in files[:2]:  # Process first 2 sample files
                filepath = os.path.join(dirname, filename)
                print(f"\n📄 File: {filename}")

                # Process with each window size
                for window_size in window_sizes:
                    analyzer = MultiWindowVoiceAnalyzer(
                        window_size_ms=window_size, hop_ratio=0.5)
                    features = analyzer.extract_windowed_features(filepath)

                    if features:
                        result = {'group': group, **features}
                        all_results[window_size].append(result)
        else:
            # Handle nested directory structure
            subdirs = [d for d in os.listdir(
                dirname) if os.path.isdir(os.path.join(dirname, d))]

            file_count = 0
            for subdir in subdirs[:2]:  # Process first 2 patient directories
                subdir_path = os.path.join(dirname, subdir)
                try:
                    files = [f for f in os.listdir(
                        subdir_path) if f.endswith('.wav')]

                    for filename in files[:1]:  # Process 1 file per patient
                        filepath = os.path.join(subdir_path, filename)
                        print(f"\n📄 File: {subdir}/{filename}")

                        # Process with each window size
                        for window_size in window_sizes:
                            analyzer = MultiWindowVoiceAnalyzer(
                                window_size_ms=window_size, hop_ratio=0.5)
                            features = analyzer.extract_windowed_features(
                                filepath)

                            if features:
                                result = {'group': group, **features}
                                all_results[window_size].append(result)

                        file_count += 1
                        if file_count >= 2:
                            break
                except Exception as e:
                    print(f"   Error accessing {subdir_path}: {e}")
                    continue

        print()

    # Save results for each window size
    print("\n" + "="*70)
    print("💾 SAVING RESULTS FOR EACH WINDOW SIZE...")

    os.makedirs("comprehensive_features", exist_ok=True)

    for window_size in window_sizes:
        if all_results[window_size]:
            csv_file = f"comprehensive_features/windowed_pd_features_{window_size}ms.csv"

            with open(csv_file, "w", newline='') as f:
                fieldnames = list(all_results[window_size][0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results[window_size])

            print(
                f"✅ Window {window_size}ms: {csv_file} ({len(all_results[window_size])} files)")

            # Quick stats
            groups = {}
            for result in all_results[window_size]:
                g = result['group']
                if g not in groups:
                    groups[g] = []
                groups[g].append(result)

            print(
                f"   Groups: {', '.join([f'{g}={len(r)}' for g, r in groups.items()])}")

    print("\n" + "="*70)
    print("🎉 MULTI-WINDOW FEATURE EXTRACTION COMPLETE!")
    print(
        f"✅ Created {len(window_sizes)} datasets with different temporal resolutions")
    print("🔬 Ready for comparative analysis and machine learning!")
    print("="*70)


if __name__ == "__main__":
    main()

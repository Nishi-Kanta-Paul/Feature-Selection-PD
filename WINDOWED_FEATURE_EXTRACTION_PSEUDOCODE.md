# MULTI-WINDOW SIZE PD FEATURE EXTRACTION PSEUDOCODE

================================================================

## COMPLETE ALGORITHMIC FLOW FOR MULTI-WINDOW FEATURE EXTRACTION

---

## 📋 OVERVIEW

```
Purpose: Extract comprehensive PD voice features using MULTIPLE window sizes
Input: WAV audio files (voice recordings)
Output: 63 core features + 6 metadata features per audio file per window size
Window Sizes: 5ms, 10ms, 20ms (three different temporal resolutions)
Hop Ratio: 50% overlap for all window sizes
```

---

## 🏗️ CLASS STRUCTURE

```pseudocode
CLASS WindowedAudioLoader:
    FUNCTION load_wav_file(filepath):
        """
        Bengali: WAV file load kore signal + metadata return kore
        Purpose: Audio file theke signal extract kora
        """
        OPEN wav_file AT filepath
        READ sample_rate, frames, raw_data FROM wav_file

        IF sample_width == 16-bit THEN
            CONVERT raw_data TO normalized_signal (-1.0 to 1.0)
        ELSE
            RETURN None
        END IF

        RETURN {
            'signal': normalized_signal,
            'sample_rate': sample_rate,
            'duration': frames / sample_rate
        }
    END FUNCTION
END CLASS


CLASS MultiWindowVoiceAnalyzer:
    """
    Bengali: Multiple window sizes diye comprehensive feature extraction
    Purpose: 5ms, 10ms, 20ms - three temporal resolutions e features extract kora
    """

    INITIALIZE:
        window_size_ms = 20        // 5ms, 10ms, or 20ms (configurable)
        hop_ratio = 0.5            // 50% overlap for all window sizes
        hop_size_ms = window_size_ms * hop_ratio
        loader = WindowedAudioLoader()


    // ============================================================
    // STEP 1: AUDIO PREPROCESSING
    // ============================================================

    FUNCTION resample_if_needed(signal, original_sr, target_sr=16000):
        """
        Bengali: Sample rate adjust kora (jodi dorkar hoy)
        Purpose: Consistent sample rate ensure kora
        """
        IF original_sr == target_sr THEN
            RETURN signal, original_sr
        END IF

        CALCULATE resample_factor = target_sr / original_sr

        IF resample_factor < 1 THEN
            // Downsample
            step = INTEGER(1 / resample_factor)
            resampled = signal[every step-th sample]
        ELSE
            // Upsample using linear interpolation
            new_length = INTEGER(length(signal) * resample_factor)
            resampled = INTERPOLATE(signal, new_length)
        END IF

        RETURN resampled, target_sr
    END FUNCTION


    FUNCTION apply_digital_filtering(signal, sample_rate):
        """
        Bengali: Digital filter apply kora (preemphasis + bandpass)
        Purpose: Voice frequency enhance kora, noise reduce kora
        """
        // Step 1: Preemphasis filter (high frequency boost)
        preemphasized[0] = signal[0]
        FOR i = 1 TO length(signal) DO
            preemphasized[i] = signal[i] - 0.97 * signal[i-1]
        END FOR

        // Step 2: Bandpass filter (80-8000 Hz for voice)
        nyquist = sample_rate / 2
        low_cutoff = 80 / nyquist
        high_cutoff = MIN(8000 / nyquist, 0.99)

        CREATE butterworth_filter(order=4, band=[low_cutoff, high_cutoff])
        filtered_signal = APPLY_FILTER(butterworth_filter, preemphasized)

        RETURN filtered_signal
    END FUNCTION


    // ============================================================
    // STEP 2: WINDOWING AND SEGMENTATION
    // ============================================================

    FUNCTION create_windows(signal, sample_rate):
        """
        Bengali: Signal ke overlapping windows e divide kora
        Purpose: Temporal analysis er jonno signal segmentation
        Note: Adjusted for smaller window sizes (5ms, 10ms, 20ms)
        """
        // Calculate window and hop sizes in samples
        window_size = INTEGER(window_size_ms * sample_rate / 1000)
        hop_size = INTEGER(hop_size_ms * sample_rate / 1000)

        // Ensure minimum window size (at least 50 samples)
        IF window_size < 50 THEN
            window_size = 50
        END IF
        IF hop_size < 25 THEN
            hop_size = 25
        END IF

        INITIALIZE windows = []
        INITIALIZE window_times = []

        // Create overlapping windows
        FOR i = 0 TO (length(signal) - window_size) STEP hop_size DO
            // Extract window
            window = signal[i : i + window_size]

            // Apply Hamming window function
            hamming_window = CREATE_HAMMING_WINDOW(length(window))
            windowed_signal = window * hamming_window

            // Store window and timestamp
            APPEND windowed_signal TO windows
            APPEND (i / sample_rate) TO window_times
        END FOR

        RETURN windows, window_times
    END FUNCTION


    // ============================================================
    // STEP 3: VOICE ACTIVITY DETECTION
    // ============================================================

    FUNCTION detect_voiced_windows(windows, sample_rate):
        """
        Bengali: Kon windows e voice ache ta detect kora
        Purpose: Voiced/unvoiced discrimination for accurate F0 extraction
        Note: Adjusted thresholds for smaller window sizes
        """
        INITIALIZE voiced_windows = []
        INITIALIZE voiced_indices = []

        FOR i = 0 TO length(windows) DO
            window = windows[i]

            // Calculate Short-Term Energy (STE)
            energy = SUM(window^2)

            // Calculate Zero-Crossing Rate (ZCR)
            zero_crossings = COUNT(SIGN_CHANGES(window))
            zcr = zero_crossings / (2 * length(window))

            // Voice Activity Detection thresholds (adjusted for smaller windows)
            IF window_size_ms < 20 THEN
                energy_threshold = 0.005  // Lower threshold for smaller windows
            ELSE
                energy_threshold = 0.01   // Standard threshold
            END IF
            zcr_threshold = 0.3

            // Voiced segment detection
            IF energy > energy_threshold AND zcr < zcr_threshold THEN
                APPEND window TO voiced_windows
                APPEND i TO voiced_indices
            END IF
        END FOR

        RETURN voiced_windows, voiced_indices
    END FUNCTION


    // ============================================================
    // STEP 4: PITCH (F0) DETECTION
    // ============================================================

    FUNCTION estimate_f0_window(window, sample_rate):
        """
        Bengali: Ekta window er jonno F0 (fundamental frequency) ber kora
        Purpose: Pitch detection using autocorrelation method
        Note: Minimum window size check added for smaller windows
        """
        IF length(window) < 50 THEN  // Minimum 50 samples needed for F0 detection
            RETURN 0, 0
        END IF

        // Autocorrelation calculation
        autocorr = CORRELATE(window, window, mode='full')
        autocorr = autocorr[middle_to_end]

        // Normalize autocorrelation
        IF autocorr[0] > 0 THEN
            autocorr = autocorr / autocorr[0]
        END IF

        // Define pitch search range
        min_f0 = 50   // Minimum expected F0 (Hz)
        max_f0 = 500  // Maximum expected F0 (Hz)
        min_lag = MAX(1, INTEGER(sample_rate / max_f0))
        max_lag = MIN(length(autocorr) - 1, INTEGER(sample_rate / min_f0))

        // Find peak in autocorrelation
        IF max_lag > min_lag THEN
            search_range = autocorr[min_lag : max_lag]

            IF MAX(search_range) > 0.3 THEN  // Threshold for valid peak
                peak_index = INDEX_OF_MAX(search_range) + min_lag
                f0 = sample_rate / peak_index
                period = peak_index / sample_rate
                RETURN f0, period
            END IF
        END IF

        RETURN 0, 0  // No valid pitch found
    END FUNCTION


    FUNCTION extract_f0_and_periods(voiced_windows, sample_rate):
        """
        Bengali: Sob voiced windows theke F0 values ber kora
        Purpose: Collect all F0 values, periods, and amplitudes
        """
        INITIALIZE f0_values = []
        INITIALIZE periods = []
        INITIALIZE amplitudes = []

        FOR EACH window IN voiced_windows DO
            // Estimate F0 for this window
            f0, period = estimate_f0_window(window, sample_rate)

            // Validate F0 range (human voice: 50-500 Hz)
            IF f0 > 0 AND 50 <= f0 <= 500 THEN
                APPEND f0 TO f0_values
                APPEND period TO periods

                // Calculate RMS amplitude
                amplitude = SQRT(SUM(window^2))
                APPEND amplitude TO amplitudes
            END IF
        END FOR

        RETURN f0_values, periods, amplitudes
    END FUNCTION


    // ============================================================
    // STEP 5: JITTER FEATURES (F0 VARIATION)
    // ============================================================

    FUNCTION calculate_windowed_jitter_features(periods):
        """
        Bengali: Period variation theke jitter features calculate kora
        Purpose: MDVP Jitter, Jitter Abs, RAP, PPQ, DDP extract kora
        """
        IF length(periods) < 3 THEN
            RETURN ALL_ZERO_JITTER_FEATURES
        END IF

        mean_period = MEAN(periods)

        IF mean_period == 0 THEN
            RETURN ALL_ZERO_JITTER_FEATURES
        END IF

        // 1. MDVP: Jitter (%)
        period_differences = ABSOLUTE_DIFFERENCE(consecutive periods)
        mdvp_jitter_percent = (MEAN(period_differences) / mean_period) * 100

        // 2. MDVP: Jitter (Abs) - in microseconds
        mdvp_jitter_abs = MEAN(period_differences) * 1000000

        // 3. MDVP: RAP (Relative Average Perturbation)
        INITIALIZE rap_values = []
        FOR i = 1 TO length(periods) - 2 DO
            local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
            IF local_mean > 0 THEN
                rap = ABSOLUTE(periods[i] - local_mean) / local_mean
                APPEND rap TO rap_values
            END IF
        END FOR
        mdvp_rap = MEAN(rap_values) * 100

        // 4. MDVP: PPQ (Five-point Period Perturbation Quotient)
        INITIALIZE ppq_values = []
        FOR i = 2 TO length(periods) - 3 DO
            local_mean = MEAN(periods[i-2 : i+3])  // 5-point average
            IF local_mean > 0 THEN
                ppq = ABSOLUTE(periods[i] - local_mean) / local_mean
                APPEND ppq TO ppq_values
            END IF
        END FOR
        mdvp_ppq = MEAN(ppq_values) * 100

        // 5. Jitter: DDP (Difference of Differences of Periods)
        IF length(period_differences) > 1 THEN
            ddp_values = ABSOLUTE_DIFFERENCE(consecutive period_differences)
            jitter_ddp = (MEAN(ddp_values) / mean_period) * 100
        ELSE
            jitter_ddp = 0
        END IF

        RETURN {
            'mdvp_jitter_percent': mdvp_jitter_percent,
            'mdvp_jitter_abs': mdvp_jitter_abs,
            'mdvp_rap': mdvp_rap,
            'mdvp_ppq': mdvp_ppq,
            'jitter_ddp': jitter_ddp
        }
    END FUNCTION


    // ============================================================
    // STEP 6: SHIMMER FEATURES (AMPLITUDE VARIATION)
    // ============================================================

    FUNCTION calculate_windowed_shimmer_features(amplitudes):
        """
        Bengali: Amplitude variation theke shimmer features calculate kora
        Purpose: MDVP Shimmer, Shimmer dB, APQ3, APQ5, APQ, DDA extract kora
        """
        IF length(amplitudes) < 3 THEN
            RETURN ALL_ZERO_SHIMMER_FEATURES
        END IF

        mean_amplitude = MEAN(amplitudes)

        IF mean_amplitude == 0 THEN
            RETURN ALL_ZERO_SHIMMER_FEATURES
        END IF

        // 1. MDVP: Shimmer (%)
        amplitude_differences = ABSOLUTE_DIFFERENCE(consecutive amplitudes)
        mdvp_shimmer_percent = (MEAN(amplitude_differences) / mean_amplitude) * 100

        // 2. MDVP: Shimmer (dB)
        mdvp_shimmer_db = 20 * LOG10(1 + mdvp_shimmer_percent/100)

        // 3. Shimmer: APQ3 (3-point Amplitude Perturbation Quotient)
        INITIALIZE apq3_values = []
        FOR i = 1 TO length(amplitudes) - 2 DO
            local_mean = MEAN(amplitudes[i-1 : i+2])  // 3-point average
            IF local_mean > 0 THEN
                apq3 = ABSOLUTE(amplitudes[i] - local_mean) / local_mean
                APPEND apq3 TO apq3_values
            END IF
        END FOR
        shimmer_apq3 = MEAN(apq3_values) * 100

        // 4. Shimmer: APQ5 (5-point Amplitude Perturbation Quotient)
        INITIALIZE apq5_values = []
        FOR i = 2 TO length(amplitudes) - 3 DO
            local_mean = MEAN(amplitudes[i-2 : i+3])  // 5-point average
            IF local_mean > 0 THEN
                apq5 = ABSOLUTE(amplitudes[i] - local_mean) / local_mean
                APPEND apq5 TO apq5_values
            END IF
        END FOR
        shimmer_apq5 = MEAN(apq5_values) * 100

        // 5. MDVP: APQ (General Amplitude Perturbation Quotient)
        mdvp_apq = shimmer_apq5

        // 6. Shimmer: DDA (Average absolute difference of amplitude differences)
        IF length(amplitude_differences) > 1 THEN
            dda_values = ABSOLUTE_DIFFERENCE(consecutive amplitude_differences)
            shimmer_dda = (MEAN(dda_values) / mean_amplitude) * 100
        ELSE
            shimmer_dda = 0
        END IF

        RETURN {
            'mdvp_shimmer_percent': mdvp_shimmer_percent,
            'mdvp_shimmer_db': mdvp_shimmer_db,
            'shimmer_apq3': shimmer_apq3,
            'shimmer_apq5': shimmer_apq5,
            'mdvp_apq': mdvp_apq,
            'shimmer_dda': shimmer_dda
        }
    END FUNCTION


    // ============================================================
    // STEP 7: VOICE QUALITY FEATURES (NHR, HNR)
    // ============================================================

    FUNCTION calculate_windowed_noise_features(voiced_windows, f0_values, sample_rate):
        """
        Bengali: Harmonic vs noise analysis kore NHR and HNR calculate kora
        Purpose: Voice quality assessment through noise measurement
        Note: FFT size adjusted for smaller windows
        """
        IF length(voiced_windows) == 0 OR length(f0_values) == 0 THEN
            RETURN {'nhr': 0, 'hnr': 0}
        END IF

        INITIALIZE hnr_values = []
        INITIALIZE nhr_values = []

        FOR i = 0 TO length(voiced_windows) DO
            window = voiced_windows[i]
            f0 = f0_values[i]

            IF f0 <= 0 THEN
                CONTINUE
            END IF

            // FFT analysis (adjusted for smaller windows)
            n_fft = MAX(256, length(window))  // Reduced from 512 for smaller windows
            fft_spectrum = FFT(window, n=n_fft)
            magnitude = ABSOLUTE(fft_spectrum[0 : n_fft/2])
            frequencies = FREQUENCY_BINS(n_fft, sample_rate)[0 : n_fft/2]

            // Detect harmonics (first 7 harmonics)
            harmonic_power = 0
            total_power = SUM(magnitude^2)

            FOR harmonic_number = 1 TO 7 DO
                target_frequency = f0 * harmonic_number

                IF target_frequency < sample_rate/2 THEN
                    // Find frequency bin closest to harmonic
                    freq_index = ARGMIN(ABSOLUTE(frequencies - target_frequency))

                    // Define window around harmonic
                    window_bins = MAX(1, INTEGER(0.1 * f0 * n_fft / sample_rate))
                    start_index = MAX(0, freq_index - window_bins)
                    end_index = MIN(length(magnitude), freq_index + window_bins + 1)

                    // Sum harmonic energy
                    harmonic_power += SUM(magnitude[start_index : end_index]^2)
                END IF
            END FOR

            // Calculate noise power
            noise_power = MAX(0.001, total_power - harmonic_power)
            harmonic_power = MAX(0.001, harmonic_power)

            // HNR (Harmonics-to-Noise Ratio) in dB
            hnr = 10 * LOG10(harmonic_power / noise_power)
            APPEND hnr TO hnr_values

            // NHR (Noise-to-Harmonic Ratio) - normalized
            nhr = noise_power / (harmonic_power + noise_power)
            APPEND nhr TO nhr_values
        END FOR

        RETURN {
            'nhr': MEAN(nhr_values),
            'hnr': MEAN(hnr_values)
        }
    END FUNCTION


    // ============================================================
    // STEP 8: PROSODIC FEATURES (F0 STATISTICS)
    // ============================================================

    FUNCTION calculate_windowed_prosodic_features(f0_values):
        """
        Bengali: F0 statistics theke prosodic features ber kora
        Purpose: MDVP Fo, Fhi, Flo and other F0 variability measures
        """
        IF length(f0_values) == 0 THEN
            RETURN ALL_ZERO_PROSODIC_FEATURES
        END IF

        // 1. MDVP: Fo (Mean fundamental frequency)
        mdvp_fo = MEAN(f0_values)

        // 2. MDVP: Fhi (Maximum fundamental frequency)
        mdvp_fhi = MAX(f0_values)

        // 3. MDVP: Flo (Minimum fundamental frequency)
        mdvp_flo = MIN(f0_values)

        // 4. F0 Standard Deviation
        f0_std = STANDARD_DEVIATION(f0_values)

        // 5. F0 Range
        f0_range = mdvp_fhi - mdvp_flo

        // 6. F0 Coefficient of Variation (%)
        f0_cv = (f0_std / mdvp_fo) * 100 IF mdvp_fo > 0 ELSE 0

        RETURN {
            'mdvp_fo': mdvp_fo,
            'mdvp_fhi': mdvp_fhi,
            'mdvp_flo': mdvp_flo,
            'f0_std': f0_std,
            'f0_range': f0_range,
            'f0_cv': f0_cv
        }
    END FUNCTION


    // ============================================================
    // STEP 9: NONLINEAR COMPLEXITY FEATURES
    // ============================================================

    FUNCTION calculate_windowed_nonlinear_features(f0_values):
        """
        Bengali: Chaos theory and nonlinear dynamics theke features calculate kora
        Purpose: RPDE, D2, DFA, Spread1, Spread2, PPE extract kora
        """
        IF length(f0_values) < 5 THEN
            RETURN ALL_ZERO_NONLINEAR_FEATURES
        END IF

        // 1. RPDE (Recurrence Period Density Entropy) - Simplified
        periods = 1.0 / f0_values
        IF length(periods) > 5 THEN
            period_differences = DIFFERENCE(consecutive periods)
            IF VARIANCE(period_differences) > 0 THEN
                rpde = VARIANCE(period_differences) / (MEAN(periods)^2)
            ELSE
                rpde = 0
            END IF
        ELSE
            rpde = 0
        END IF

        // 2. D2 (Correlation dimension) - Approximation
        d2 = STANDARD_DEVIATION(f0_values) / MEAN(f0_values)

        // 3. DFA (Detrended Fluctuation Analysis)
        IF length(f0_values) > 10 THEN
            // Cumulative sum (integration)
            y = CUMULATIVE_SUM(f0_values - MEAN(f0_values))

            // Define scales for fluctuation analysis
            scales = [4, 8, 16]
            INITIALIZE fluctuations = []

            FOR EACH scale IN scales DO
                IF scale < length(y) / 2 THEN
                    n_segments = length(y) / scale
                    INITIALIZE local_fluctuations = []

                    FOR segment_index = 0 TO n_segments DO
                        start = segment_index * scale
                        end = start + scale
                        segment = y[start : end]

                        IF length(segment) > 1 THEN
                            // Fit linear trend
                            trend = LINEAR_FIT(segment)

                            // Detrend
                            detrended = segment - trend

                            // Calculate fluctuation
                            fluctuation = STANDARD_DEVIATION(detrended)
                            APPEND fluctuation TO local_fluctuations
                        END IF
                    END FOR

                    IF length(local_fluctuations) > 0 THEN
                        APPEND MEAN(local_fluctuations) TO fluctuations
                    END IF
                END IF
            END FOR

            // Calculate DFA exponent (slope in log-log plot)
            IF length(fluctuations) > 1 THEN
                log_scales = LOG(scales[0 : length(fluctuations)])
                log_fluctuations = LOG(fluctuations + 1e-10)
                dfa = SLOPE(LINEAR_FIT(log_scales, log_fluctuations))
            ELSE
                dfa = 0
            END IF
        ELSE
            dfa = 0
        END IF

        // 4. Spread1 (F0 standard deviation)
        spread1 = STANDARD_DEVIATION(f0_values)

        // 5. Spread2 (F0 variance)
        spread2 = VARIANCE(f0_values)

        // 6. PPE (Pitch Period Entropy)
        IF length(f0_values) > 3 THEN
            periods = 1.0 / f0_values
            n_bins = MIN(10, length(periods) / 2)

            IF n_bins > 1 THEN
                // Create histogram
                histogram = HISTOGRAM(periods, bins=n_bins)

                // Add small value to avoid log(0)
                histogram = histogram + 1e-10

                // Normalize to probability
                probability = histogram / SUM(histogram)

                // Calculate entropy
                probability = FILTER(probability > 1e-10)
                ppe = -SUM(probability * LOG2(probability))
            ELSE
                ppe = 0
            END IF
        ELSE
            ppe = 0
        END IF

        RETURN {
            'rpde': rpde,
            'd2': d2,
            'dfa': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'ppe': ppe
        }
    END FUNCTION


    // ============================================================
    // STEP 10: SIGNAL PROCESSING FEATURES
    // ============================================================

    FUNCTION calculate_windowed_signal_processing_features(windows, sample_rate):
        """
        Bengali: STE, ZCR, Spectral Entropy calculate kora
        Purpose: Basic signal processing features for ML
        Note: FFT size adjusted for smaller windows
        """
        IF length(windows) == 0 THEN
            RETURN ALL_ZERO_SIGNAL_FEATURES
        END IF

        // 1. Short-Term Energy (STE) calculation
        INITIALIZE ste_values = []
        FOR EACH window IN windows DO
            ste = SUM(window^2)
            APPEND ste TO ste_values
        END FOR

        // 2. Zero-Crossing Rate (ZCR) calculation
        INITIALIZE zcr_values = []
        FOR EACH window IN windows DO
            sign_changes = COUNT(SIGN(window[i]) != SIGN(window[i+1]))
            zcr = sign_changes / (2 * length(window))
            APPEND zcr TO zcr_values
        END FOR

        // 3. Spectral Entropy calculation
        INITIALIZE spectral_entropies = []
        FOR EACH window IN windows DO
            // FFT (adjusted size for smaller windows)
            n_fft = MIN(512, MAX(256, length(window) * 2))
            fft_spectrum = FFT(window, n=n_fft)
            magnitude = ABSOLUTE(fft_spectrum[0 : n_fft/2])

            IF SUM(magnitude) > 0 THEN
                // Normalize to probability distribution
                magnitude_normalized = magnitude / SUM(magnitude)
                magnitude_safe = magnitude_normalized + 1e-10

                // Calculate spectral entropy
                spectral_entropy = -SUM(magnitude_safe * LOG2(magnitude_safe))
                APPEND spectral_entropy TO spectral_entropies
            END IF
        END FOR

        RETURN {
            'ste_mean': MEAN(ste_values),
            'ste_std': STANDARD_DEVIATION(ste_values),
            'zcr_mean': MEAN(zcr_values),
            'zcr_std': STANDARD_DEVIATION(zcr_values),
            'spectral_entropy_mean': MEAN(spectral_entropies),
            'spectral_entropy_std': STANDARD_DEVIATION(spectral_entropies)
        }
    END FUNCTION


    // ============================================================
    // STEP 11: SPECTRAL TRANSFORMATIONS (FFT, DCT, STFT)
    // ============================================================

    FUNCTION calculate_windowed_spectral_transformations(windows, sample_rate):
        """
        Bengali: FFT, DCT, STFT analysis kore spectral features ber kora
        Purpose: Frequency domain characterization
        Note: Adaptive FFT size for different window sizes
        """
        IF length(windows) == 0 THEN
            RETURN ALL_ZERO_SPECTRAL_FEATURES
        END IF

        INITIALIZE spectral_centroids = []
        INITIALIZE spectral_spreads = []
        INITIALIZE spectral_rolloffs = []
        INITIALIZE spectral_fluxes = []
        INITIALIZE dct_energies = []
        INITIALIZE stft_energies = []

        previous_magnitude = None

        FOR EACH window IN windows DO
            // FFT analysis (adaptive size)
            n_fft = MIN(512, MAX(256, length(window) * 2))
            fft_spectrum = FFT(window, n=n_fft)
            magnitude = ABSOLUTE(fft_spectrum[0 : n_fft/2])
            frequencies = LINSPACE(0, sample_rate/2, length(magnitude))

            IF SUM(magnitude) > 0 THEN
                // Normalize magnitude
                magnitude_normalized = magnitude / SUM(magnitude)

                // 1. Spectral Centroid (center of mass of spectrum)
                centroid = SUM(frequencies * magnitude_normalized)
                APPEND centroid TO spectral_centroids

                // 2. Spectral Spread (variance around centroid)
                spread = SQRT(SUM(((frequencies - centroid)^2) * magnitude_normalized))
                APPEND spread TO spectral_spreads

                // 3. Spectral Rolloff (85% energy point)
                cumulative_sum = CUMULATIVE_SUM(magnitude_normalized)
                rolloff_indices = WHERE(cumulative_sum >= 0.85 * cumulative_sum[end])
                IF length(rolloff_indices) > 0 THEN
                    rolloff = frequencies[rolloff_indices[0]]
                    APPEND rolloff TO spectral_rolloffs
                END IF

                // 4. Spectral Flux (spectral change)
                IF previous_magnitude IS NOT None THEN
                    flux = SUM((magnitude - previous_magnitude)^2)
                    APPEND flux TO spectral_fluxes
                END IF
                previous_magnitude = magnitude
            END IF

            // DCT (Discrete Cosine Transform)
            dct_coefficients = DCT(window, type=2, normalized=True)
            dct_energy = SUM(dct_coefficients^2)
            APPEND dct_energy TO dct_energies

            // STFT energy (simplified - window energy)
            stft_energy = SUM(window^2)
            APPEND stft_energy TO stft_energies
        END FOR

        RETURN {
            'spectral_centroid_mean': MEAN(spectral_centroids),
            'spectral_spread_mean': MEAN(spectral_spreads),
            'spectral_rolloff_mean': MEAN(spectral_rolloffs),
            'spectral_flux_mean': MEAN(spectral_fluxes),
            'dct_energy_mean': MEAN(dct_energies),
            'stft_energy_mean': MEAN(stft_energies)
        }
    END FUNCTION


    // ============================================================
    // STEP 12: MFCC FEATURES
    // ============================================================

    FUNCTION calculate_windowed_mfcc_features(windows, sample_rate, n_mfcc=13):
        """
        Bengali: Mel-Frequency Cepstral Coefficients calculate kora
        Purpose: Spectral envelope characterization for ML
        Note: Adaptive FFT size for different window sizes
        """
        IF length(windows) == 0 THEN
            RETURN ALL_ZERO_MFCC_FEATURES(n_mfcc)
        END IF

        INITIALIZE all_mfccs = []

        FOR EACH window IN windows DO
            // FFT (adaptive size)
            n_fft = MIN(512, MAX(256, length(window) * 2))
            fft_spectrum = FFT(window, n=n_fft)
            magnitude = ABSOLUTE(fft_spectrum[0 : n_fft/2])

            // Compute MFCC for this window
            mfccs = compute_mfcc_window(magnitude, sample_rate, n_mfcc)

            IF mfccs IS NOT None THEN
                APPEND mfccs TO all_mfccs
            END IF
        END FOR

        IF length(all_mfccs) == 0 THEN
            RETURN ALL_ZERO_MFCC_FEATURES(n_mfcc)
        END IF

        // Calculate statistics for each MFCC coefficient
        INITIALIZE mfcc_features = {}
        FOR i = 1 TO n_mfcc DO
            mfcc_features['mfcc_' + i + '_mean'] = MEAN(all_mfccs[:, i-1])
            mfcc_features['mfcc_' + i + '_std'] = STANDARD_DEVIATION(all_mfccs[:, i-1])
        END FOR

        RETURN mfcc_features
    END FUNCTION


    FUNCTION compute_mfcc_window(magnitude, sample_rate, n_mfcc=13, n_mels=26):
        """
        Bengali: Ekta window er jonno MFCC compute kora
        Purpose: Single window MFCC calculation
        """
        TRY:
            // 1. Create mel filter bank
            mel_filters = create_mel_filterbank(length(magnitude), sample_rate, n_mels)

            // 2. Apply mel filters to power spectrum
            mel_spectrum = MATRIX_MULTIPLY(mel_filters, magnitude^2)

            // 3. Log mel spectrum
            log_mel_spectrum = LOG(mel_spectrum + 1e-10)

            // 4. DCT to get MFCCs
            mfccs = DCT(log_mel_spectrum, type=2, normalized=True)[0 : n_mfcc]

            RETURN mfccs
        CATCH:
            RETURN None
        END TRY
    END FUNCTION


    FUNCTION create_mel_filterbank(nfft, sample_rate, n_mels):
        """
        Bengali: Mel filter bank create kora
        Purpose: Mel-scale frequency mapping for MFCC
        """
        // Helper functions for Hz-Mel conversion
        FUNCTION hz_to_mel(hz):
            RETURN 2595 * LOG10(1 + hz / 700)
        END FUNCTION

        FUNCTION mel_to_hz(mel):
            RETURN 700 * (10^(mel / 2595) - 1)
        END FUNCTION

        // Define mel scale points
        low_freq_mel = 0
        high_freq_mel = hz_to_mel(sample_rate / 2)
        mel_points = LINSPACE(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        // Convert to FFT bin numbers
        bin_points = FLOOR((nfft + 1) * hz_points / sample_rate)

        // Create triangular filters
        INITIALIZE filterbank[n_mels][nfft] = 0

        FOR m = 1 TO n_mels DO
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            // Rising slope
            FOR k = f_m_minus TO f_m DO
                IF f_m > f_m_minus THEN
                    filterbank[m - 1][k] = (k - f_m_minus) / (f_m - f_m_minus)
                END IF
            END FOR

            // Falling slope
            FOR k = f_m TO f_m_plus DO
                IF f_m_plus > f_m THEN
                    filterbank[m - 1][k] = (f_m_plus - k) / (f_m_plus - f_m)
                END IF
            END FOR
        END FOR

        RETURN filterbank
    END FUNCTION


    // ============================================================
    // STEP 13: MAIN FEATURE EXTRACTION PIPELINE
    // ============================================================

    FUNCTION extract_windowed_features(filepath):
        """
        Bengali: Ekta audio file theke sob windowed features extract kora
        Purpose: Complete feature extraction pipeline for one window size
        """
        PRINT "Processing: " + FILENAME(filepath) + " [Window: " + window_size_ms + "ms]"

        // STEP 1: Load audio
        audio_data = loader.load_wav_file(filepath)
        IF audio_data IS None THEN
            PRINT "Failed to load audio"
            RETURN None
        END IF

        signal = audio_data['signal']
        original_sample_rate = audio_data['sample_rate']

        // STEP 2: Resample if needed
        IF original_sample_rate > 22050 THEN
            signal, sample_rate = resample_if_needed(signal, original_sample_rate, 16000)
        ELSE
            sample_rate = original_sample_rate
        END IF

        TRY:
            // STEP 3: Apply digital filtering
            filtered_signal = apply_digital_filtering(signal, sample_rate)

            // STEP 4: Create overlapping windows
            windows, window_times = create_windows(filtered_signal, sample_rate)

            // STEP 5: Detect voiced windows
            voiced_windows, voiced_indices = detect_voiced_windows(windows, sample_rate)

            // STEP 6: Extract F0 and periods
            f0_values, periods, amplitudes = extract_f0_and_periods(voiced_windows, sample_rate)

            // STEP 7: Initialize features dictionary
            INITIALIZE features = {}

            // STEP 8: Calculate all feature categories

            // 8a. MDVP Jitter Features (5 features)
            jitter_features = calculate_windowed_jitter_features(periods)
            ADD jitter_features TO features

            // 8b. MDVP Shimmer Features (6 features)
            shimmer_features = calculate_windowed_shimmer_features(amplitudes)
            ADD shimmer_features TO features

            // 8c. Voice Quality & Noise Features (2 features)
            noise_features = calculate_windowed_noise_features(voiced_windows, f0_values, sample_rate)
            ADD noise_features TO features

            // 8d. Frequency-based Prosodic Features (6 features)
            prosodic_features = calculate_windowed_prosodic_features(f0_values)
            ADD prosodic_features TO features

            // 8e. Nonlinear Dynamical Complexity Metrics (6 features)
            nonlinear_features = calculate_windowed_nonlinear_features(f0_values)
            ADD nonlinear_features TO features

            // 8f. Signal Processing Features (6 features)
            signal_features = calculate_windowed_signal_processing_features(windows, sample_rate)
            ADD signal_features TO features

            // 8g. Spectral Transformations (6 features)
            spectral_features = calculate_windowed_spectral_transformations(windows, sample_rate)
            ADD spectral_features TO features

            // 8h. MFCC Features (26 features: 13 mean + 13 std)
            mfcc_features = calculate_windowed_mfcc_features(windows, sample_rate, n_mfcc=13)
            ADD mfcc_features TO features

            // STEP 9: Add metadata
            ADD TO features {
                'filename': BASENAME(filepath),
                'window_size_ms': window_size_ms,
                'hop_size_ms': hop_size_ms,
                'total_windows': length(windows),
                'voiced_windows': length(voiced_windows),
                'num_f0_values': length(f0_values)
            }

            PRINT "✅ Windows: " + length(windows) + ", Voiced: " + length(voiced_windows) +
                  ", F0: " + length(f0_values)
            RETURN features

        CATCH Exception e:
            PRINT "❌ Feature extraction error: " + e
            RETURN None
        END TRY
    END FUNCTION

END CLASS
```

---

## 🎯 MAIN EXECUTION FLOW

```pseudocode
FUNCTION main():
    """
    Bengali: Main pipeline - multiple window sizes diye sob audio files process kora
    Purpose: Multi-window feature extraction system (5ms, 10ms, 20ms)
    """
    PRINT "MULTI-WINDOW SIZE PD VOICE FEATURE EXTRACTION"
    PRINT "Window Sizes: 5ms, 10ms, 20ms"
    PRINT "Processing each file with all three window sizes"

    // Define window sizes to test
    window_sizes = [5, 10, 20]  // milliseconds

    // Initialize results storage for each window size
    INITIALIZE all_results = {}
    FOR EACH ws IN window_sizes DO
        all_results[ws] = []
    END FOR

    // Define data directories to process
    data_paths = [
        ('HC', 'Processed_data_sample_raw_voice/rnnoise_out/0'),
        ('PD', 'Processed_data_sample_raw_voice/rnnoise_out/1'),
        ('SAMPLE', 'sample_audio_files')
    ]

    // Process each directory
    FOR EACH (group, directory) IN data_paths DO
        PRINT "Processing " + group + " files from: " + directory

        IF NOT DIRECTORY_EXISTS(directory) THEN
            PRINT "Directory not found"
            CONTINUE
        END IF

        // Get audio files from directory
        IF directory == 'sample_audio_files' THEN
            files = LIST_FILES(directory, pattern='*.wav')

            FOR EACH filename IN files[0:2] DO  // Process first 2
                filepath = JOIN(directory, filename)
                PRINT "File: " + filename

                // Process with each window size
                FOR EACH window_size IN window_sizes DO
                    // Create analyzer for this window size
                    analyzer = MultiWindowVoiceAnalyzer(window_size_ms=window_size, hop_ratio=0.5)

                    // Extract features
                    features = analyzer.extract_windowed_features(filepath)

                    IF features IS NOT None THEN
                        result = {'group': group, ...features}
                        APPEND result TO all_results[window_size]
                    END IF
                END FOR
            END FOR
        ELSE
            // Handle nested directory structure
            subdirectories = LIST_DIRECTORIES(directory)

            file_count = 0
            FOR EACH subdirectory IN subdirectories[0:2] DO  // First 2 patients
                subdir_path = JOIN(directory, subdirectory)
                files = LIST_FILES(subdir_path, pattern='*.wav')

                FOR EACH filename IN files[0:1] DO  // First 1 file per patient
                    filepath = JOIN(subdir_path, filename)
                    PRINT "File: " + subdirectory + "/" + filename

                    // Process with each window size
                    FOR EACH window_size IN window_sizes DO
                        // Create analyzer for this window size
                        analyzer = MultiWindowVoiceAnalyzer(window_size_ms=window_size, hop_ratio=0.5)

                        // Extract features
                        features = analyzer.extract_windowed_features(filepath)

                        IF features IS NOT None THEN
                            result = {'group': group, ...features}
                            APPEND result TO all_results[window_size]
                        END IF
                    END FOR

                    file_count++
                    IF file_count >= 2 THEN
                        BREAK
                    END IF
                END FOR
            END FOR
        END IF
    END FOR

    // Save results for each window size
    PRINT "Saving results for each window size..."

    CREATE_DIRECTORY("comprehensive_features")

    FOR EACH window_size IN window_sizes DO
        IF length(all_results[window_size]) > 0 THEN
            csv_file = "comprehensive_features/windowed_pd_features_" + window_size + "ms.csv"

            // Write CSV file
            OPEN csv_file FOR WRITING
            fieldnames = KEYS(all_results[window_size][0])
            WRITE_CSV_HEADER(fieldnames)

            FOR EACH result IN all_results[window_size] DO
                WRITE_CSV_ROW(result)
            END FOR

            CLOSE csv_file

            PRINT "✅ Window " + window_size + "ms: " + csv_file +
                  " (" + length(all_results[window_size]) + " files)"

            // Quick stats
            INITIALIZE groups = {}
            FOR EACH result IN all_results[window_size] DO
                g = result['group']
                IF g NOT IN groups THEN
                    groups[g] = []
                END IF
                APPEND result TO groups[g]
            END FOR

            PRINT "   Groups: " + JOIN([g + "=" + length(r) FOR g, r IN groups])
        END IF
    END FOR

    PRINT "🎉 MULTI-WINDOW FEATURE EXTRACTION COMPLETE!"
    PRINT "✅ Created " + length(window_sizes) + " datasets with different temporal resolutions"
    PRINT "🔬 Ready for comparative analysis and machine learning!"
END FUNCTION

// Execute main pipeline
CALL main()
```

---

## 📊 FEATURE SUMMARY

### Total Features Extracted Per Window Size: 63 core features + 6 metadata = 69 features

### Window Sizes: 5ms, 10ms, 20ms (three temporal resolutions)

1. **Jitter Features (5)**: mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp
2. **Shimmer Features (6)**: mdvp_shimmer_percent, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda
3. **Noise Features (2)**: nhr, hnr
4. **Prosodic Features (6)**: mdvp_fo, mdvp_fhi, mdvp_flo, f0_std, f0_range, f0_cv
5. **Nonlinear Features (6)**: rpde, d2, dfa, spread1, spread2, ppe
6. **Signal Processing (6)**: ste_mean, ste_std, zcr_mean, zcr_std, spectral_entropy_mean, spectral_entropy_std
7. **Spectral Transform (6)**: spectral_centroid_mean, spectral_spread_mean, spectral_rolloff_mean, spectral_flux_mean, dct_energy_mean, stft_energy_mean
8. **MFCC (26)**: mfcc_1_mean through mfcc_13_mean, mfcc_1_std through mfcc_13_std
9. **Metadata (6)**: filename, window_size_ms, hop_size_ms, total_windows, voiced_windows, num_f0_values

---

## 🔬 KEY ALGORITHMIC CONCEPTS

### Multi-Window Strategy

- **5ms windows**: Finest temporal resolution (~6,900 windows per 10s file)
- **10ms windows**: Balanced resolution (~2,750 windows per 10s file)
- **20ms windows**: Standard resolution (~1,380 windows per 10s file)
- **Hop Ratio**: 50% overlap for all window sizes

### Window Size Selection Trade-offs

- **Smaller windows (5ms)**: Best temporal resolution, harder F0 detection
- **Medium windows (10ms)**: Balanced approach, good F0 detection
- **Larger windows (20ms)**: Excellent F0 detection, adequate resolution

### Voice Activity Detection

- **Energy Threshold**: Adjusted for window size (0.005 for <20ms, 0.01 for ≥20ms)
- **ZCR Threshold**: Consistent 0.3 across all window sizes

### Pitch Detection

- **Autocorrelation Method**: Robust F0 estimation
- **F0 Range**: 50-500 Hz (human voice range)
- **Minimum Window Size**: 50 samples for F0 detection

### Feature Aggregation

- **Windowed Calculation**: Features computed per window
- **Statistical Aggregation**: Mean/std across all windows

### Adaptive FFT Sizing

- **Small windows (5ms)**: FFT size = MIN(512, MAX(256, window_length \* 2))
- **Medium windows (10ms)**: FFT size = MIN(512, MAX(256, window_length \* 2))
- **Large windows (20ms)**: FFT size = MIN(512, MAX(256, window_length \* 2))

---

## 📁 OUTPUT STRUCTURE

### Generated Files (one per window size):

```
comprehensive_features/
├── windowed_pd_features_5ms.csv   (finest temporal resolution)
├── windowed_pd_features_10ms.csv  (balanced resolution)
└── windowed_pd_features_20ms.csv  (standard resolution)
```

### Multi-Scale Analysis Benefits:

1. Capture micro-variations (5ms) and macro-patterns (20ms)
2. Compare feature stability across temporal scales
3. Enable multi-resolution ensemble learning
4. Identify optimal window size for specific features

---

_End of Multi-Window PD Feature Extraction Pseudocode_

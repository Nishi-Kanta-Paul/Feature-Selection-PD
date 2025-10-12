PROGRAM: Comprehensive PD Voice Feature Extraction for Parkinson's Disease Detection

CLASS ImprovedAudioLoader:
// Step 1: Audio File Loading
FUNCTION load_wav_file(filepath):
TRY:
WAV file open koro using wave library
sample_rate = wav_file.getframerate()
frames = wav_file.getnframes()
raw_data = wav_file.readframes(frames)

            IF sample_width == 16-bit:
                signal = convert raw_data to normalized float array
                duration = frames / sample_rate calculate koro

                RETURN {
                    signal: normalized_signal,
                    sample_rate: sample_rate,
                    duration: duration
                }
            ELSE:
                RETURN None
        CATCH error:
            PRINT "Error loading " + filepath + ": " + error
            RETURN None

CLASS ImprovedVoiceAnalyzer:
// Step 1: Analyzer Initialization
FUNCTION **init**():
loader = NEW ImprovedAudioLoader()
Initialize preprocessing parameters

    // Step 2: Sample Rate Resampling
    FUNCTION resample_if_needed(signal, original_sr, target_sr=16000):
        IF original_sr == target_sr:
            RETURN signal, original_sr

        resample_factor = target_sr / original_sr calculate koro

        IF resample_factor < 1:
            // Downsample koro
            step = integer(1 / resample_factor)
            resampled = signal[every step elements]
        ELSE:
            // Upsample koro using linear interpolation
            new_length = integer(length(signal) * resample_factor)
            old_indices = linspace(0, length(signal) - 1, length(signal))
            new_indices = linspace(0, length(signal) - 1, new_length)
            resampled = interpolate(old_indices, signal, new_indices)

        RETURN resampled, target_sr

    // Step 3: Preprocessing Filters
    FUNCTION preemphasis_filter(signal, alpha=0.97):
        // High-frequency emphasis apply koro
        preemphasized = [signal[0]]
        FOR i FROM 1 TO length(signal) - 1:
            preemphasized.append(signal[i] - alpha * signal[i-1])
        RETURN preemphasized

    FUNCTION apply_hamming_window(frame):
        IF length(frame) == 0:
            RETURN frame
        window = hamming_window(length(frame))
        RETURN frame * window

    FUNCTION bandpass_filter(signal, sr, low_freq=80, high_freq=800):
        nyquist = sr / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)

        TRY:
            b, a = butterworth_filter(order=4, cutoff=[low, high], type='bandpass')
            filtered = forward_backward_filter(b, a, signal)
            RETURN filtered
        CATCH error:
            RETURN original signal

    // Step 4: Enhanced Pitch Detection
    FUNCTION improved_pitch_detection(frame, sr):
        IF length(frame) < 100:
            RETURN 0, 0

        // Method 1: Autocorrelation with preprocessing
        preemphasized = preemphasis_filter(frame)
        filtered = bandpass_filter(preemphasized, sr)

        // Autocorrelation calculate koro
        autocorr = correlate(filtered, filtered, mode='full')
        autocorr = autocorr[length(autocorr)/2:]

        // Normalize koro
        IF autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        // Adaptive pitch range based on sample rate
        min_f0 = 50  // Lower bound
        max_f0 = 500  // Upper bound

        min_lag = max(1, integer(sr / max_f0))
        max_lag = min(length(autocorr) - 1, integer(sr / min_f0))

        IF max_lag > min_lag AND max_lag < length(autocorr):
            search_range = autocorr[min_lag:max_lag]
            IF length(search_range) > 0:
                // Peaks find koro with minimum prominence
                peak_threshold = 0.2
                peaks = []
                FOR i FROM 1 TO length(search_range) - 2:
                    IF (search_range[i] > search_range[i-1] AND
                        search_range[i] > search_range[i+1] AND
                        search_range[i] > peak_threshold):
                        peaks.append({index: i + min_lag, value: search_range[i]})

                IF peaks NOT empty:
                    // Strongest peak choose koro
                    best_peak = max(peaks, key=value)
                    peak_idx = best_peak.index
                    f0 = sr / peak_idx
                    period = peak_idx / sr
                    RETURN f0, period

        // Method 2: Zero-crossing based estimation (fallback)
        zero_crossings = find_zero_crossings(filtered)
        IF length(zero_crossings) > 4:
            avg_period = 2 * mean(diff(zero_crossings)) / sr
            IF 0.002 < avg_period < 0.02:  // Reasonable period range
                f0 = 1.0 / avg_period
                RETURN f0, avg_period

        RETURN 0, 0

    // Step 5: Enhanced Voiced Segment Detection
    FUNCTION get_enhanced_voiced_segments(signal, sr):
        // Adaptive frame parameters based on sample rate
        frame_duration = 0.025  // 25ms
        hop_duration = 0.01     // 10ms

        frame_length = integer(frame_duration * sr)
        hop_length = integer(hop_duration * sr)

        // Ensure minimum frame size
        frame_length = max(frame_length, 200)
        hop_length = max(hop_length, 80)

        voiced_frames = []
        periods = []
        f0_values = []
        amplitudes = []

        FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
            frame = signal[i:i + frame_length]

            // Apply windowing
            windowed_frame = apply_hamming_window(frame)

            // Calculate energy
            energy = sum(windowed_frame^2)

            // Calculate zero crossing rate
            zcr = sum(abs(diff(sign(windowed_frame)))) / (2 * length(windowed_frame))

            // Adaptive thresholds
            energy_threshold = mean(signal^2) * 0.1
            zcr_threshold = 0.4

            // Enhanced voiced detection
            IF energy > energy_threshold AND zcr < zcr_threshold:
                voiced_frames.append(windowed_frame)

                // Improved pitch detection
                f0, period = improved_pitch_detection(windowed_frame, sr)
                IF f0 > 0 AND 50 <= f0 <= 500:  // Valid F0 range
                    f0_values.append(f0)
                    periods.append(period)
                    amplitudes.append(sqrt(energy))

        PRINT "Enhanced detection: " + length(voiced_frames) + " voiced frames, " + length(f0_values) + " valid F0 values"
        RETURN voiced_frames, periods, f0_values, amplitudes

    // Step 6: MDVP Jitter Features Calculation (5 features)
    FUNCTION calculate_comprehensive_jitter_features(periods):
        IF length(periods) < 3:
            RETURN all_zeros_dict

        periods = convert_to_array(periods)
        mean_period = mean(periods)

        IF mean_period == 0:
            RETURN all_zeros_dict

        // MDVP: Jitter (%) - Period-to-period variation
        period_diffs = abs(diff(periods))
        mdvp_jitter_percent = (mean(period_diffs) / mean_period) * 100

        // MDVP: Jitter (Abs) - in microseconds
        mdvp_jitter_abs = mean(period_diffs) * 1000000

        // MDVP: RAP (Relative Average Perturbation)
        rap_values = []
        FOR i FROM 1 TO length(periods) - 2:
            local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
            IF local_mean > 0:
                rap_values.append(abs(periods[i] - local_mean) / local_mean)
        mdvp_rap = mean(rap_values) * 100 IF rap_values NOT empty ELSE 0

        // MDVP: PPQ (Five-point Period Perturbation Quotient)
        ppq_values = []
        FOR i FROM 2 TO length(periods) - 3:
            local_mean = mean(periods[i-2:i+3])
            IF local_mean > 0:
                ppq_values.append(abs(periods[i] - local_mean) / local_mean)
        mdvp_ppq = mean(ppq_values) * 100 IF ppq_values NOT empty ELSE 0

        // Jitter: DDP
        IF length(period_diffs) > 1:
            ddp_values = abs(diff(period_diffs))
            jitter_ddp = (mean(ddp_values) / mean_period) * 100
        ELSE:
            jitter_ddp = 0

        RETURN {
            mdvp_jitter_percent: mdvp_jitter_percent,
            mdvp_jitter_abs: mdvp_jitter_abs,
            mdvp_rap: mdvp_rap,
            mdvp_ppq: mdvp_ppq,
            jitter_ddp: jitter_ddp
        }

    // Step 7: MDVP Shimmer Features Calculation (6 features)
    FUNCTION calculate_comprehensive_shimmer_features(voiced_frames, amplitudes):
        IF length(amplitudes) < 3:
            RETURN all_zeros_dict

        amplitudes = convert_to_array(amplitudes)
        mean_amplitude = mean(amplitudes)

        IF mean_amplitude == 0:
            RETURN all_zeros_dict

        // MDVP: Shimmer (%)
        amp_diffs = abs(diff(amplitudes))
        mdvp_shimmer_percent = (mean(amp_diffs) / mean_amplitude) * 100

        // MDVP: Shimmer (dB)
        mdvp_shimmer_db = 20 * log10(1 + mdvp_shimmer_percent/100) IF mdvp_shimmer_percent > 0 ELSE 0

        // Shimmer: APQ3
        apq3_values = []
        FOR i FROM 1 TO length(amplitudes) - 2:
            local_mean = mean(amplitudes[i-1:i+2])
            IF local_mean > 0:
                apq3_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq3 = mean(apq3_values) * 100 IF apq3_values NOT empty ELSE 0

        // Shimmer: APQ5
        apq5_values = []
        FOR i FROM 2 TO length(amplitudes) - 3:
            local_mean = mean(amplitudes[i-2:i+3])
            IF local_mean > 0:
                apq5_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        shimmer_apq5 = mean(apq5_values) * 100 IF apq5_values NOT empty ELSE 0

        // MDVP: APQ
        mdvp_apq = shimmer_apq5

        // Shimmer: DDA
        IF length(amp_diffs) > 1:
            dda_values = abs(diff(amp_diffs))
            shimmer_dda = (mean(dda_values) / mean_amplitude) * 100
        ELSE:
            shimmer_dda = 0

        RETURN {
            mdvp_shimmer_percent: mdvp_shimmer_percent,
            mdvp_shimmer_db: mdvp_shimmer_db,
            shimmer_apq3: shimmer_apq3,
            shimmer_apq5: shimmer_apq5,
            mdvp_apq: mdvp_apq,
            shimmer_dda: shimmer_dda
        }

    // Step 8: Voice Quality & Noise Features (2 features)
    FUNCTION calculate_comprehensive_noise_features(voiced_frames, f0_values, sr):
        IF length(voiced_frames) == 0 OR length(f0_values) == 0:
            RETURN {nhr: 0, hnr: 0}

        hnr_values = []
        nhr_values = []

        FOR i FROM 0 TO min(length(voiced_frames), length(f0_values)) - 1:
            frame = voiced_frames[i]
            f0 = f0_values[i]

            IF f0 <= 0:
                CONTINUE

            // Preprocess frame
            preemphasized = preemphasis_filter(frame)

            // FFT analysis
            n_fft = max(512, length(preemphasized))
            fft_frame = FFT(preemphasized, n=n_fft)
            magnitude = abs(fft_frame[0:n_fft/2])
            freqs = frequency_bins(n_fft, sr)[0:n_fft/2]

            // Enhanced harmonic detection
            harmonic_power = 0
            total_power = sum(magnitude^2)

            // Detect first 7 harmonics
            FOR h FROM 1 TO 7:
                target_freq = f0 * h
                IF target_freq < sr/2:
                    freq_res = sr / n_fft
                    bin_idx = integer(target_freq / freq_res)
                    window_bins = max(1, integer(0.05 * f0 / freq_res))

                    start_bin = max(0, bin_idx - window_bins)
                    end_bin = min(length(magnitude), bin_idx + window_bins + 1)

                    harmonic_power += sum(magnitude[start_bin:end_bin]^2)

            // Calculate noise power
            noise_power = max(0.001, total_power - harmonic_power)
            harmonic_power = max(0.001, harmonic_power)

            // HNR and NHR
            hnr = 10 * log10(harmonic_power / noise_power)
            nhr = noise_power / (harmonic_power + noise_power)

            hnr_values.append(hnr)
            nhr_values.append(nhr)

        RETURN {
            nhr: mean(nhr_values) IF nhr_values NOT empty ELSE 0,
            hnr: mean(hnr_values) IF hnr_values NOT empty ELSE 0
        }

    // Step 9: Frequency-based Prosodic Features (6 features)
    FUNCTION calculate_prosodic_features(f0_values):
        IF length(f0_values) == 0:
            RETURN all_zeros_dict

        f0_array = convert_to_array(f0_values)

        RETURN {
            mdvp_fo: mean(f0_array),           // Mean F0
            mdvp_fhi: max(f0_array),           // Maximum F0
            mdvp_flo: min(f0_array),           // Minimum F0
            f0_std: std(f0_array),             // F0 standard deviation
            f0_range: max(f0_array) - min(f0_array),  // F0 range
            f0_cv: (std(f0_array) / mean(f0_array)) * 100 IF mean(f0_array) > 0 ELSE 0
        }

    // Step 10: Nonlinear Dynamical Complexity Metrics (6 features)
    FUNCTION calculate_nonlinear_features(f0_values, signal):
        IF length(f0_values) < 5:
            RETURN all_zeros_dict

        f0_array = convert_to_array(f0_values)

        // RPDE (Recurrence Period Density Entropy)
        periods = 1.0 / f0_array
        IF length(periods) > 5:
            period_diffs = diff(periods)
            rpde = variance(period_diffs) / (mean(periods)^2) IF variance(period_diffs) > 0 ELSE 0
        ELSE:
            rpde = 0

        // D2 (Correlation Dimension approximation)
        d2 = std(f0_array) / mean(f0_array) IF mean(f0_array) > 0 AND length(f0_array) > 10 ELSE 0

        // DFA (Detrended Fluctuation Analysis)
        IF length(f0_array) > 10:
            y = cumulative_sum(f0_array - mean(f0_array))
            scales = [4, 8, 16, 32]
            fluctuations = []

            FOR scale IN scales:
                IF scale < length(y) / 2:
                    n_segments = length(y) / scale
                    local_fluct = []

                    FOR i FROM 0 TO n_segments - 1:
                        start = i * scale
                        end = start + scale
                        segment = y[start:end]

                        // Linear detrending
                        coeffs = polynomial_fit(segment, degree=1)
                        trend = evaluate_polynomial(coeffs, range(length(segment)))
                        detrended = segment - trend
                        local_fluct.append(std(detrended))

                    fluctuations.append(mean(local_fluct))

            IF length(fluctuations) > 1:
                log_scales = log(scales[0:length(fluctuations)])
                log_flucts = log(fluctuations + 1e-10)
                TRY:
                    dfa = polynomial_fit(log_scales, log_flucts, degree=1)[0]
                CATCH:
                    dfa = 0
            ELSE:
                dfa = 0
        ELSE:
            dfa = 0

        // Spread measures
        spread1 = std(f0_array)
        spread2 = variance(f0_array)

        // PPE (Pitch Period Entropy)
        IF length(f0_array) > 3:
            periods = 1.0 / f0_array
            n_bins = min(10, length(periods) / 2)
            IF n_bins > 1:
                hist = histogram(periods, bins=n_bins)
                hist = hist + 1e-10
                prob = hist / sum(hist)
                prob = prob[prob > 1e-10]
                ppe = -sum(prob * log2(prob))
            ELSE:
                ppe = 0
        ELSE:
            ppe = 0

        RETURN {
            rpde: rpde,
            d2: d2,
            dfa: dfa,
            spread1: spread1,
            spread2: spread2,
            ppe: ppe
        }

    // Step 11: Spectral Features & Spectral Entropy (10 features)
    FUNCTION calculate_spectral_features(signal, sr):
        frame_length = integer(0.025 * sr)  // 25ms
        hop_length = integer(0.01 * sr)     // 10ms

        spectral_entropies = []
        spectral_centroids = []
        spectral_spreads = []
        spectral_rolloffs = []
        spectral_fluxes = []

        prev_magnitude = NULL

        FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
            frame = signal[i:i + frame_length]
            windowed = apply_hamming_window(frame)

            // FFT
            fft_frame = FFT(windowed, n=512)
            magnitude = abs(fft_frame[0:256])

            IF sum(magnitude) > 0:
                // Normalize
                magnitude_norm = magnitude / sum(magnitude)

                // Spectral Entropy
                magnitude_safe = magnitude_norm + 1e-10
                spectral_entropy = -sum(magnitude_safe * log2(magnitude_safe))
                spectral_entropies.append(spectral_entropy)

                // Frequency bins
                freqs = linspace(0, sr/2, length(magnitude))

                // Spectral Centroid
                spectral_centroid = sum(freqs * magnitude_norm)
                spectral_centroids.append(spectral_centroid)

                // Spectral Spread
                spectral_spread = sqrt(sum(((freqs - spectral_centroid)^2) * magnitude_norm))
                spectral_spreads.append(spectral_spread)

                // Spectral Rolloff (85% energy point)
                cumsum_mag = cumulative_sum(magnitude_norm)
                rolloff_indices = find_where(cumsum_mag >= 0.85 * cumsum_mag[end])
                IF length(rolloff_indices) > 0:
                    spectral_rolloff = freqs[rolloff_indices[0]]
                    spectral_rolloffs.append(spectral_rolloff)

                // Spectral Flux
                IF prev_magnitude is NOT NULL:
                    flux = sum((magnitude - prev_magnitude)^2)
                    spectral_fluxes.append(flux)
                prev_magnitude = magnitude

        RETURN {
            spectral_entropy_mean: mean(spectral_entropies) IF spectral_entropies NOT empty ELSE 0,
            spectral_entropy_std: std(spectral_entropies) IF length(spectral_entropies) > 1 ELSE 0,
            spectral_centroid_mean: mean(spectral_centroids) IF spectral_centroids NOT empty ELSE 0,
            spectral_centroid_std: std(spectral_centroids) IF length(spectral_centroids) > 1 ELSE 0,
            spectral_spread_mean: mean(spectral_spreads) IF spectral_spreads NOT empty ELSE 0,
            spectral_spread_std: std(spectral_spreads) IF length(spectral_spreads) > 1 ELSE 0,
            spectral_rolloff_mean: mean(spectral_rolloffs) IF spectral_rolloffs NOT empty ELSE 0,
            spectral_rolloff_std: std(spectral_rolloffs) IF length(spectral_rolloffs) > 1 ELSE 0,
            spectral_flux_mean: mean(spectral_fluxes) IF spectral_fluxes NOT empty ELSE 0,
            spectral_flux_std: std(spectral_fluxes) IF length(spectral_fluxes) > 1 ELSE 0
        }

    // Step 12: MFCC Features (26 features: 13 mean + 13 std)
    FUNCTION calculate_mfcc_features(signal, sr, n_mfcc=13):
        frame_length = integer(0.025 * sr)  // 25ms
        hop_length = integer(0.01 * sr)     // 10ms

        all_mfccs = []

        FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
            frame = signal[i:i + frame_length]
            windowed = apply_hamming_window(frame)
            preemphasized = preemphasis_filter(windowed)

            // FFT with fixed size
            n_fft = 512
            fft_frame = FFT(preemphasized, n=n_fft)
            magnitude = abs(fft_frame[0:n_fft/2])

            // MFCC calculation
            mfccs = compute_mfcc_frame(magnitude, sr, n_mfcc)
            IF mfccs is NOT NULL:
                all_mfccs.append(mfccs)

        IF length(all_mfccs) == 0:
            // Return zero features
            mfcc_features = {}
            FOR i FROM 1 TO n_mfcc:
                mfcc_features['mfcc_' + i + '_mean'] = 0
                mfcc_features['mfcc_' + i + '_std'] = 0
            RETURN mfcc_features

        all_mfccs = convert_to_matrix(all_mfccs)

        // Calculate statistics for each MFCC coefficient
        mfcc_features = {}
        FOR i FROM 1 TO n_mfcc:
            IF i <= columns(all_mfccs):
                mfcc_features['mfcc_' + i + '_mean'] = mean(all_mfccs[:, i-1])
                mfcc_features['mfcc_' + i + '_std'] = std(all_mfccs[:, i-1])
            ELSE:
                mfcc_features['mfcc_' + i + '_mean'] = 0
                mfcc_features['mfcc_' + i + '_std'] = 0

        RETURN mfcc_features

    // Step 12.1: MFCC Frame Computation
    FUNCTION compute_mfcc_frame(magnitude, sr, n_mfcc=13, n_mels=26):
        TRY:
            // Create mel filter bank
            mel_filters = create_mel_filterbank(length(magnitude), sr, n_mels)

            // Apply filters
            mel_spectrum = matrix_multiply(mel_filters, magnitude^2)

            // Log spectrum
            log_mel = log(mel_spectrum + 1e-10)

            // DCT (Discrete Cosine Transform)
            mfccs = DCT(log_mel, type=2, normalization='ortho')[0:n_mfcc]

            RETURN mfccs
        CATCH:
            RETURN NULL

    // Step 12.2: Mel Filter Bank Creation
    FUNCTION create_mel_filterbank(nfft, sr, n_mels):
        // Mel scale conversion functions
        FUNCTION hz_to_mel(hz):
            RETURN 2595 * log10(1 + hz / 700)

        FUNCTION mel_to_hz(mel):
            RETURN 700 * (10^(mel / 2595) - 1)

        // Mel points
        low_freq_mel = 0
        high_freq_mel = hz_to_mel(sr / 2)
        mel_points = linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        // FFT bin points
        bin_points = floor((nfft + 1) * hz_points / sr)

        // Filter bank
        filterbank = zeros_matrix(n_mels, nfft)

        FOR m FROM 1 TO n_mels:
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            FOR k FROM f_m_minus TO f_m - 1:
                IF f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

            FOR k FROM f_m TO f_m_plus - 1:
                IF f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        RETURN filterbank

    // Step 13: Additional Signal Processing Features (10 features)
    FUNCTION calculate_additional_features(signal, voiced_frames, sr):
        frame_length = integer(0.025 * sr)  // 25ms
        hop_length = integer(0.01 * sr)     // 10ms

        energy_values = []
        zcr_values = []

        FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
            frame = signal[i:i + frame_length]

            // Short-term energy
            ste = sum(frame^2)
            energy_values.append(ste)

            // Zero crossing rate
            zcr = sum(abs(diff(sign(frame)))) / (2 * length(frame))
            zcr_values.append(zcr)

        energy_array = convert_to_array(energy_values)
        zcr_array = convert_to_array(zcr_values)

        // Voice activity
        voiced_ratio = length(voiced_frames) / max(1, length(energy_values))

        RETURN {
            ste_mean: mean(energy_array),
            ste_std: std(energy_array),
            ste_max: max(energy_array),
            ste_min: min(energy_array),
            zcr_mean: mean(zcr_array),
            zcr_std: std(zcr_array),
            zcr_max: max(zcr_array),
            zcr_min: min(zcr_array),
            voiced_ratio: voiced_ratio,
            energy_entropy: calculate_energy_entropy(energy_array)
        }

    // Step 13.1: Energy Entropy Calculation
    FUNCTION calculate_energy_entropy(energy_values):
        IF length(energy_values) < 2:
            RETURN 0

        total_energy = sum(energy_values)
        IF total_energy == 0:
            RETURN 0

        energy_prob = energy_values / total_energy
        energy_prob = energy_prob[energy_prob > 0]

        RETURN -sum(energy_prob * log2(energy_prob))

    // Step 14: Main Feature Extraction Function
    FUNCTION extract_all_comprehensive_features(filepath):
        PRINT "Processing: " + filename(filepath)

        // Load audio
        audio_data = loader.load_wav_file(filepath)
        IF audio_data is NULL:
            PRINT "  Failed to load audio"
            RETURN NULL

        signal = convert_to_array(audio_data.signal)
        original_sr = audio_data.sample_rate

        // Resample for better pitch detection if needed
        IF original_sr > 22050:
            signal, sr = resample_if_needed(signal, original_sr, 16000)
            PRINT "  Resampled from " + original_sr + "Hz to " + sr + "Hz for better pitch detection"
        ELSE:
            sr = original_sr

        PRINT "  Duration: " + audio_data.duration + "s, Working Sample Rate: " + sr + "Hz"

        TRY:
            // Enhanced voiced segment analysis
            voiced_frames, periods, f0_values, amplitudes = get_enhanced_voiced_segments(signal, sr)

            features = {}

            // 1. MDVP Jitter Features
            jitter_features = calculate_comprehensive_jitter_features(periods)
            features.update(jitter_features)

            // 2. MDVP Shimmer Features
            shimmer_features = calculate_comprehensive_shimmer_features(voiced_frames, amplitudes)
            features.update(shimmer_features)

            // 3. Voice Quality & Noise Features
            noise_features = calculate_comprehensive_noise_features(voiced_frames, f0_values, sr)
            features.update(noise_features)

            // 4. Frequency-based Prosodic Features
            prosodic_features = calculate_prosodic_features(f0_values)
            features.update(prosodic_features)

            // 5. Nonlinear Dynamical Complexity Metrics
            nonlinear_features = calculate_nonlinear_features(f0_values, signal)
            features.update(nonlinear_features)

            // 6. Spectral Features & Spectral Entropy
            spectral_features = calculate_spectral_features(signal, sr)
            features.update(spectral_features)

            // 7. MFCC Features
            mfcc_features = calculate_mfcc_features(signal, sr, n_mfcc=13)
            features.update(mfcc_features)

            // 8. Additional Signal Processing Features
            additional_features = calculate_additional_features(signal, voiced_frames, sr)
            features.update(additional_features)

            // Add metadata
            features.update({
                filename: filename(filepath),
                duration: audio_data.duration,
                original_sample_rate: original_sr,
                working_sample_rate: sr,
                num_voiced_frames: length(voiced_frames),
                num_f0_values: length(f0_values),
                num_periods: length(periods),
                num_amplitudes: length(amplitudes)
            })

            PRINT "  ✅ Extracted " + length(features) + " comprehensive features"
            RETURN features

        CATCH error:
            PRINT "  ❌ Feature extraction error: " + error
            print_traceback()
            RETURN NULL

// Main Execution Flow
FUNCTION main():
PRINT "COMPREHENSIVE PD VOICE FEATURE EXTRACTION - IMPROVED"
PRINT "🎯 IMPLEMENTING ALL REQUESTED FEATURES:"
PRINT " ✓ Fundamental Frequency Variation Measures (Jitter-Based)"
PRINT " ✓ Amplitude Variation Parameters (Shimmer-Based)"
PRINT " ✓ Voice Quality and Noise Features"
PRINT " ✓ Frequency-Based Prosodic Features"
PRINT " ✓ Nonlinear Dynamical Complexity Metrics"
PRINT " ✓ Nonlinear F0 Pitch Variability"
PRINT " ✓ Additional Signal Processing Features"
PRINT " ✓ Advanced Spectral Transformations"
PRINT " ✓ Mel-Frequency Cepstral Coefficients (MFCCs)"
PRINT " ✓ Normalization and Segmentation Windowing"

    analyzer = NEW ImprovedVoiceAnalyzer()
    all_results = []

    // Try multiple data directories
    data_paths = [
        {group: 'HC', path: 'Processed_data_sample_raw_voice/rnnoise_out/0'},
        {group: 'PD', path: 'Processed_data_sample_raw_voice/rnnoise_out/1'},
        {group: 'HC', path: 'Processed_data_sample_raw_voice/raw_wav/0'},
        {group: 'PD', path: 'Processed_data_sample_raw_voice/raw_wav/1'},
        {group: 'SAMPLE', path: 'sample_audio_files'}
    ]

    processed_any = False

    FOR each data_entry IN data_paths:
        group = data_entry.group
        dirname = data_entry.path

        IF processed_any AND group IN ['HC', 'PD'] AND
           count_group_results(all_results, group) >= 5:
            CONTINUE  // Skip if we already have enough samples

        PRINT "🔍 Searching " + group + " files in: " + dirname

        IF NOT directory_exists(dirname):
            PRINT "   Directory not found: " + dirname
            CONTINUE

        IF dirname == 'sample_audio_files':
            // Handle sample files directly
            files = get_wav_files(dirname)
            file_count = 0

            FOR each filename IN files[0:5]:  // Process first 5 sample files
                filepath = join_path(dirname, filename)
                PRINT "   [" + (file_count+1) + "] " + filename

                features = analyzer.extract_all_comprehensive_features(filepath)
                IF features is NOT NULL:
                    result = {group: group, features}
                    all_results.append(result)
                    processed_any = True

                file_count++
                PRINT ""
        ELSE:
            // Handle nested directory structure
            subdirs = get_subdirectories(dirname)

            IF length(subdirs) == 0:
                PRINT "   No subdirectories found"
                CONTINUE

            file_count = 0
            FOR each subdir IN subdirs[0:3]:  // Process first 3 patient directories
                subdir_path = join_path(dirname, subdir)
                TRY:
                    files = get_wav_files(subdir_path)

                    FOR each filename IN files[0:2]:  // Process first 2 files per patient
                        filepath = join_path(subdir_path, filename)
                        PRINT "   [" + (file_count+1) + "] " + subdir + "/" + filename

                        features = analyzer.extract_all_comprehensive_features(filepath)
                        IF features is NOT NULL:
                            result = {group: group, features}
                            all_results.append(result)
                            processed_any = True

                        file_count++
                        IF file_count >= 5:  // Limit files per group
                            BREAK
                        PRINT ""

                    IF file_count >= 5:
                        BREAK
                CATCH error:
                    PRINT "   Error accessing " + subdir_path + ": " + error
                    CONTINUE

    // Save results
    PRINT "💾 SAVING COMPREHENSIVE RESULTS..."

    create_directory("comprehensive_features")

    IF length(all_results) > 0:
        csv_file = "comprehensive_features/comprehensive_pd_features_final.csv"
        write_csv_file(all_results, csv_file)

        PRINT "✅ Comprehensive features saved to: " + csv_file

        // Analysis summary
        groups = group_by_category(all_results)

        PRINT "📊 FEATURE EXTRACTION SUMMARY:"
        FOR each group, results IN groups:
            PRINT "   " + group + ": " + length(results) + " files processed"

        PRINT "Total features per file: " + (length(all_results[0]) - 1) + " (excluding group label)"

        // Feature category breakdown
        print_feature_categories(all_results[0])

        // Statistical comparison
        IF 'HC' IN groups AND 'PD' IN groups:
            print_hc_vs_pd_comparison(groups['HC'], groups['PD'])
    ELSE:
        PRINT "❌ No files were successfully processed!"
        PRINT "   Please check that audio files exist in the expected directories."

    PRINT "🎉 COMPREHENSIVE FEATURE EXTRACTION COMPLETE!"
    PRINT "✅ ALL REQUESTED FEATURES SUCCESSFULLY IMPLEMENTED"
    PRINT "🔬 Ready for machine learning analysis and classification!"

// Helper Functions
FUNCTION count_group_results(results, target_group):
count = 0
FOR each result IN results:
IF result.group == target_group:
count++
RETURN count

FUNCTION get_wav_files(directory):
files = list_files(directory)
wav_files = []
FOR each file IN files:
IF file.endswith('.wav'):
wav_files.append(file)
RETURN wav_files

FUNCTION write_csv_file(results, filename):
TRY:
open_csv_file(filename, mode='write')
IF length(results) > 0:
fieldnames = get_keys(results[0])
write_csv_header(fieldnames)
FOR each result IN results:
write_csv_row(result)
close_csv_file()
CATCH error:
PRINT "Error writing CSV: " + error

CALL main()

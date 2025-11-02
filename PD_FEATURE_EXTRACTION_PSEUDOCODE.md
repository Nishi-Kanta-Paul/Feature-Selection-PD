# COMPREHENSIVE PD VOICE FEATURE EXTRACTION - PSEUDOCODE FLOW

## ⚠️ **CORRECTED WORKFLOW**

**IMPORTANT:** Features extracted from **UNFILTERED** audio (basic preprocessed only)

```
Correct Pipeline:
Raw Audio → Basic Preprocessing (16kHz, NO filter) →
Feature Extraction (79 features) → Optional Feature Selection → Model Training
```

---

## 🔄 COMPLETE FEATURE EXTRACTION WORKFLOW

```
PROGRAM: Comprehensive PD Voice Feature Extraction for Parkinson's Disease Detection
INPUT: Basic preprocessed audio from preprocessed_data_basic/ (UNFILTERED 16kHz audio)

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
    FUNCTION __init__():
        loader = NEW ImprovedAudioLoader()
        Initialize preprocessing parameters

    // ========================================
    // PHASE 2: DATA DISCOVERY & PREPROCESSING
    // ========================================

    FOR each data_directory IN [HC_path, PD_path, Sample_path]:
        Print "🔍 Searching files in: " + data_directory

        IF directory_exists(data_directory):
            subdirectories = get_subdirectories(data_directory)

            FOR each subdir IN subdirectories[0:3]:  // Limit to 3 patients
                audio_files = get_wav_files(subdir)

                FOR each audio_file IN audio_files[0:2]:  // Limit to 2 files per patient
                    filepath = join_path(subdir, audio_file)

                    // Call main feature extraction
                    features = Extract_All_Comprehensive_Features(filepath)

                    IF features is not NULL:
                        features['group'] = determine_group(data_directory)
                        results.append(features)
                    END IF
                END FOR
            END FOR
        END IF
    END FOR

END MAIN_FLOW

// ========================================
// CORE FUNCTION: Feature Extraction Pipeline
// ========================================

FUNCTION Extract_All_Comprehensive_Features(filepath)
    Print "Processing: " + filename(filepath)

    // ========================================
    // STEP 1: AUDIO LOADING & PREPROCESSING
    // ========================================

    audio_data = Load_WAV_File(filepath)
    IF audio_data is NULL:
        Return NULL
    END IF

    signal = convert_to_array(audio_data.signal)
    original_sr = audio_data.sample_rate
    duration = audio_data.duration

    // Adaptive sample rate handling
    IF original_sr > 22050:
        signal, sr = Resample_Audio(signal, original_sr, target_sr=16000)
        Print "Resampled from " + original_sr + "Hz to " + sr + "Hz"
    ELSE:
        sr = original_sr
    END IF

    Print "Duration: " + duration + "s, Working Sample Rate: " + sr + "Hz"

    // ========================================
    // STEP 2: VOICE ACTIVITY DETECTION & PITCH ANALYSIS
    // ========================================

    voiced_frames, periods, f0_values, amplitudes = Get_Enhanced_Voiced_Segments(signal, sr)
    Print "Enhanced detection: " + count(voiced_frames) + " voiced frames, " + count(f0_values) + " valid F0 values"

    Initialize features = empty_dictionary

    // ========================================
    // STEP 3: FEATURE EXTRACTION (8 CATEGORIES)
    // ========================================

    // Category 1: MDVP Jitter Features (5 features)
    jitter_features = Calculate_Comprehensive_Jitter_Features(periods)
    features.update(jitter_features)

    // Category 2: MDVP Shimmer Features (6 features)
    shimmer_features = Calculate_Comprehensive_Shimmer_Features(voiced_frames, amplitudes)
    features.update(shimmer_features)

    // Category 3: Voice Quality & Noise Features (2 features)
    noise_features = Calculate_Comprehensive_Noise_Features(voiced_frames, f0_values, sr)
    features.update(noise_features)

    // Category 4: Frequency-based Prosodic Features (6 features)
    prosodic_features = Calculate_Prosodic_Features(f0_values)
    features.update(prosodic_features)

    // Category 5: Nonlinear Dynamical Complexity Metrics (6 features)
    nonlinear_features = Calculate_Nonlinear_Features(f0_values, signal)
    features.update(nonlinear_features)

    // Category 6: Spectral Features & Spectral Entropy (10 features)
    spectral_features = Calculate_Spectral_Features(signal, sr)
    features.update(spectral_features)

    // Category 7: MFCC Features (26 features: 13 mean + 13 std)
    mfcc_features = Calculate_MFCC_Features(signal, sr, n_mfcc=13)
    features.update(mfcc_features)

    // Category 8: Additional Signal Processing Features (10 features)
    additional_features = Calculate_Additional_Features(signal, voiced_frames, sr)
    features.update(additional_features)

    // Add metadata (8 features)
    features['filename'] = filename(filepath)
    features['duration'] = duration
    features['original_sample_rate'] = original_sr
    features['working_sample_rate'] = sr
    features['num_voiced_frames'] = count(voiced_frames)
    features['num_f0_values'] = count(f0_values)
    features['num_periods'] = count(periods)
    features['num_amplitudes'] = count(amplitudes)

    Print "✅ Extracted " + count(features) + " comprehensive features"
    Return features

END FUNCTION

// ========================================
// DETAILED SUBFUNCTIONS
// ========================================

FUNCTION Load_WAV_File(filepath)
    TRY:
        Open WAV file using wave library
        Extract sample_rate, frames, raw_data

        IF sample_width == 16-bit:
            signal = convert_to_normalized_float(raw_data)
            Return {signal, sample_rate, duration}
        ELSE:
            Return NULL
        END IF
    CATCH error:
        Print "Error loading " + filepath + ": " + error
        Return NULL
    END TRY
END FUNCTION

FUNCTION Resample_Audio(signal, original_sr, target_sr)
    resample_factor = target_sr / original_sr

    IF resample_factor < 1:
        // Downsample using decimation
        step = integer(1 / resample_factor)
        resampled = signal[every step elements]
    ELSE:
        // Upsample using linear interpolation
        new_length = integer(length(signal) * resample_factor)
        resampled = linear_interpolate(signal, new_length)
    END IF

    Return resampled, target_sr
END FUNCTION

FUNCTION Get_Enhanced_Voiced_Segments(signal, sr)
    // Frame parameters
    frame_duration = 0.025  // 25ms
    hop_duration = 0.01     // 10ms
    frame_length = integer(frame_duration * sr)
    hop_length = integer(hop_duration * sr)

    Initialize voiced_frames = empty_list
    Initialize periods = empty_list
    Initialize f0_values = empty_list
    Initialize amplitudes = empty_list

    FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
        frame = signal[i : i + frame_length]
        windowed_frame = Apply_Hamming_Window(frame)

        // Voice activity detection
        energy = sum(windowed_frame^2)
        zcr = count_zero_crossings(windowed_frame) / length(windowed_frame)

        energy_threshold = mean(signal^2) * 0.1
        zcr_threshold = 0.4

        IF energy > energy_threshold AND zcr < zcr_threshold:
            voiced_frames.append(windowed_frame)

            // Enhanced pitch detection
            f0, period = Improved_Pitch_Detection(windowed_frame, sr)

            IF f0 > 0 AND 50 <= f0 <= 500:
                f0_values.append(f0)
                periods.append(period)
                amplitudes.append(sqrt(energy))
            END IF
        END IF
    END FOR

    Return voiced_frames, periods, f0_values, amplitudes
END FUNCTION

FUNCTION Improved_Pitch_Detection(frame, sr)
    IF length(frame) < 100:
        Return 0, 0
    END IF

    // Method 1: Enhanced Autocorrelation
    preemphasized = Apply_Preemphasis_Filter(frame, alpha=0.97)
    filtered = Apply_Bandpass_Filter(preemphasized, sr, low=80, high=800)

    autocorr = autocorrelation(filtered)
    autocorr = normalize(autocorr)

    min_f0 = 50
    max_f0 = 500
    min_lag = integer(sr / max_f0)
    max_lag = integer(sr / min_f0)

    search_range = autocorr[min_lag : max_lag]
    peaks = find_peaks(search_range, threshold=0.2)

    IF peaks is not empty:
        best_peak = max_value_peak(peaks)
        peak_idx = best_peak.index + min_lag
        f0 = sr / peak_idx
        period = peak_idx / sr
        Return f0, period
    END IF

    // Method 2: Zero-crossing fallback
    zero_crossings = find_zero_crossings(filtered)
    IF count(zero_crossings) > 4:
        avg_period = 2 * mean(diff(zero_crossings)) / sr
        IF 0.002 < avg_period < 0.02:
            f0 = 1.0 / avg_period
            Return f0, avg_period
        END IF
    END IF

    Return 0, 0
END FUNCTION

FUNCTION Calculate_Comprehensive_Jitter_Features(periods)
    IF count(periods) < 3:
        Return all_zeros_dict
    END IF

    periods = convert_to_array(periods)
    mean_period = mean(periods)

    IF mean_period == 0:
        Return all_zeros_dict
    END IF

    // MDVP: Jitter (%)
    period_diffs = abs(diff(periods))
    mdvp_jitter_percent = (mean(period_diffs) / mean_period) * 100

    // MDVP: Jitter (Abs) in microseconds
    mdvp_jitter_abs = mean(period_diffs) * 1000000

    // MDVP: RAP (Relative Average Perturbation)
    rap_values = empty_list
    FOR i FROM 1 TO length(periods) - 2:
        local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
        IF local_mean > 0:
            rap_values.append(abs(periods[i] - local_mean) / local_mean)
        END IF
    END FOR
    mdvp_rap = mean(rap_values) * 100 IF rap_values not empty ELSE 0

    // MDVP: PPQ (Five-point Period Perturbation Quotient)
    ppq_values = empty_list
    FOR i FROM 2 TO length(periods) - 3:
        local_mean = mean(periods[i-2 : i+3])
        IF local_mean > 0:
            ppq_values.append(abs(periods[i] - local_mean) / local_mean)
        END IF
    END FOR
    mdvp_ppq = mean(ppq_values) * 100 IF ppq_values not empty ELSE 0

    // Jitter: DDP
    IF length(period_diffs) > 1:
        ddp_values = abs(diff(period_diffs))
        jitter_ddp = (mean(ddp_values) / mean_period) * 100
    ELSE:
        jitter_ddp = 0
    END IF

    Return {
        mdvp_jitter_percent,
        mdvp_jitter_abs,
        mdvp_rap,
        mdvp_ppq,
        jitter_ddp
    }
END FUNCTION

FUNCTION Calculate_Comprehensive_Shimmer_Features(voiced_frames, amplitudes)
    IF count(amplitudes) < 3:
        Return all_zeros_dict
    END IF

    amplitudes = convert_to_array(amplitudes)
    mean_amplitude = mean(amplitudes)

    IF mean_amplitude == 0:
        Return all_zeros_dict
    END IF

    // MDVP: Shimmer (%)
    amp_diffs = abs(diff(amplitudes))
    mdvp_shimmer_percent = (mean(amp_diffs) / mean_amplitude) * 100

    // MDVP: Shimmer (dB)
    mdvp_shimmer_db = 20 * log10(1 + mdvp_shimmer_percent/100)

    // Shimmer: APQ3 (3-point)
    apq3_values = empty_list
    FOR i FROM 1 TO length(amplitudes) - 2:
        local_mean = mean(amplitudes[i-1 : i+2])
        IF local_mean > 0:
            apq3_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        END IF
    END FOR
    shimmer_apq3 = mean(apq3_values) * 100 IF apq3_values not empty ELSE 0

    // Shimmer: APQ5 (5-point)
    apq5_values = empty_list
    FOR i FROM 2 TO length(amplitudes) - 3:
        local_mean = mean(amplitudes[i-2 : i+3])
        IF local_mean > 0:
            apq5_values.append(abs(amplitudes[i] - local_mean) / local_mean)
        END IF
    END FOR
    shimmer_apq5 = mean(apq5_values) * 100 IF apq5_values not empty ELSE 0

    // MDVP: APQ
    mdvp_apq = shimmer_apq5

    // Shimmer: DDA
    IF length(amp_diffs) > 1:
        dda_values = abs(diff(amp_diffs))
        shimmer_dda = (mean(dda_values) / mean_amplitude) * 100
    ELSE:
        shimmer_dda = 0
    END IF

    Return {
        mdvp_shimmer_percent,
        mdvp_shimmer_db,
        shimmer_apq3,
        shimmer_apq5,
        mdvp_apq,
        shimmer_dda
    }
END FUNCTION

FUNCTION Calculate_Comprehensive_Noise_Features(voiced_frames, f0_values, sr)
    IF count(voiced_frames) == 0 OR count(f0_values) == 0:
        Return {nhr: 0, hnr: 0}
    END IF

    hnr_values = empty_list
    nhr_values = empty_list

    FOR i FROM 0 TO min(length(voiced_frames), length(f0_values)) - 1:
        frame = voiced_frames[i]
        f0 = f0_values[i]

        IF f0 <= 0:
            Continue
        END IF

        // Preprocess frame
        preemphasized = Apply_Preemphasis_Filter(frame)

        // FFT analysis
        n_fft = max(512, length(preemphasized))
        fft_frame = FFT(preemphasized, n=n_fft)
        magnitude = abs(fft_frame[0 : n_fft/2])
        freqs = frequency_bins(n_fft, sr)[0 : n_fft/2]

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

                harmonic_power += sum(magnitude[start_bin : end_bin]^2)
            END IF
        END FOR

        // Calculate noise power
        noise_power = max(0.001, total_power - harmonic_power)
        harmonic_power = max(0.001, harmonic_power)

        // HNR and NHR
        hnr = 10 * log10(harmonic_power / noise_power)
        nhr = noise_power / (harmonic_power + noise_power)

        hnr_values.append(hnr)
        nhr_values.append(nhr)
    END FOR

    Return {
        nhr: mean(nhr_values) IF nhr_values not empty ELSE 0,
        hnr: mean(hnr_values) IF hnr_values not empty ELSE 0
    }
END FUNCTION

FUNCTION Calculate_Prosodic_Features(f0_values)
    IF count(f0_values) == 0:
        Return all_zeros_dict
    END IF

    f0_array = convert_to_array(f0_values)

    Return {
        mdvp_fo: mean(f0_array),           // Mean F0
        mdvp_fhi: max(f0_array),           // Maximum F0
        mdvp_flo: min(f0_array),           // Minimum F0
        f0_std: std(f0_array),             // F0 standard deviation
        f0_range: max(f0_array) - min(f0_array),  // F0 range
        f0_cv: (std(f0_array) / mean(f0_array)) * 100 IF mean(f0_array) > 0 ELSE 0
    }
END FUNCTION

FUNCTION Calculate_Nonlinear_Features(f0_values, signal)
    IF count(f0_values) < 5:
        Return all_zeros_dict
    END IF

    f0_array = convert_to_array(f0_values)

    // RPDE (Recurrence Period Density Entropy)
    periods = 1.0 / f0_array
    IF length(periods) > 5:
        period_diffs = diff(periods)
        rpde = variance(period_diffs) / (mean(periods)^2) IF variance(period_diffs) > 0 ELSE 0
    ELSE:
        rpde = 0
    END IF

    // D2 (Correlation Dimension approximation)
    d2 = std(f0_array) / mean(f0_array) IF mean(f0_array) > 0 AND length(f0_array) > 10 ELSE 0

    // DFA (Detrended Fluctuation Analysis)
    IF length(f0_array) > 10:
        y = cumulative_sum(f0_array - mean(f0_array))
        scales = [4, 8, 16, 32]
        fluctuations = empty_list

        FOR scale IN scales:
            IF scale < length(y) / 2:
                n_segments = length(y) / scale
                local_fluct = empty_list

                FOR i FROM 0 TO n_segments - 1:
                    start = i * scale
                    end = start + scale
                    segment = y[start : end]

                    // Linear detrending
                    coeffs = polynomial_fit(segment, degree=1)
                    trend = evaluate_polynomial(coeffs, range(length(segment)))
                    detrended = segment - trend
                    local_fluct.append(std(detrended))
                END FOR

                fluctuations.append(mean(local_fluct))
            END IF
        END FOR

        IF length(fluctuations) > 1:
            log_scales = log(scales[0 : length(fluctuations)])
            log_flucts = log(fluctuations + 1e-10)
            dfa = polynomial_fit(log_scales, log_flucts, degree=1)[0]
        ELSE:
            dfa = 0
        END IF
    ELSE:
        dfa = 0
    END IF

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
        END IF
    ELSE:
        ppe = 0
    END IF

    Return {
        rpde,
        d2,
        dfa,
        spread1,
        spread2,
        ppe
    }
END FUNCTION

FUNCTION Calculate_Spectral_Features(signal, sr)
    frame_length = integer(0.025 * sr)  // 25ms
    hop_length = integer(0.01 * sr)     // 10ms

    spectral_entropies = empty_list
    spectral_centroids = empty_list
    spectral_spreads = empty_list
    spectral_rolloffs = empty_list
    spectral_fluxes = empty_list

    prev_magnitude = NULL

    FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
        frame = signal[i : i + frame_length]
        windowed = Apply_Hamming_Window(frame)

        // FFT
        fft_frame = FFT(windowed, n=512)
        magnitude = abs(fft_frame[0 : 256])

        IF sum(magnitude) > 0:
            // Normalize
            magnitude_norm = magnitude / sum(magnitude)

            // Spectral Entropy
            magnitude_safe = magnitude_norm + 1e-10
            spectral_entropy = -sum(magnitude_safe * log2(magnitude_safe))
            spectral_entropies.append(spectral_entropy)

            // Frequency bins
            freqs = linear_space(0, sr/2, length(magnitude))

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
            END IF

            // Spectral Flux
            IF prev_magnitude is not NULL:
                flux = sum((magnitude - prev_magnitude)^2)
                spectral_fluxes.append(flux)
            END IF
            prev_magnitude = magnitude
        END IF
    END FOR

    Return {
        spectral_entropy_mean: mean(spectral_entropies),
        spectral_entropy_std: std(spectral_entropies),
        spectral_centroid_mean: mean(spectral_centroids),
        spectral_centroid_std: std(spectral_centroids),
        spectral_spread_mean: mean(spectral_spreads),
        spectral_spread_std: std(spectral_spreads),
        spectral_rolloff_mean: mean(spectral_rolloffs),
        spectral_rolloff_std: std(spectral_rolloffs),
        spectral_flux_mean: mean(spectral_fluxes),
        spectral_flux_std: std(spectral_fluxes)
    }
END FUNCTION

FUNCTION Calculate_MFCC_Features(signal, sr, n_mfcc=13)
    frame_length = integer(0.025 * sr)  // 25ms
    hop_length = integer(0.01 * sr)     // 10ms

    all_mfccs = empty_list

    FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
        frame = signal[i : i + frame_length]
        windowed = Apply_Hamming_Window(frame)
        preemphasized = Apply_Preemphasis_Filter(windowed)

        // FFT with fixed size
        n_fft = 512
        fft_frame = FFT(preemphasized, n=n_fft)
        magnitude = abs(fft_frame[0 : n_fft/2])

        // MFCC calculation
        mfccs = Compute_MFCC_Frame(magnitude, sr, n_mfcc)
        IF mfccs is not NULL:
            all_mfccs.append(mfccs)
        END IF
    END FOR

    IF count(all_mfccs) == 0:
        Return all_zeros_mfcc_dict
    END IF

    all_mfccs = convert_to_matrix(all_mfccs)

    // Calculate statistics for each MFCC coefficient
    mfcc_features = empty_dict
    FOR i FROM 1 TO n_mfcc:
        IF i <= columns(all_mfccs):
            mfcc_features['mfcc_' + i + '_mean'] = mean(all_mfccs[:, i-1])
            mfcc_features['mfcc_' + i + '_std'] = std(all_mfccs[:, i-1])
        ELSE:
            mfcc_features['mfcc_' + i + '_mean'] = 0
            mfcc_features['mfcc_' + i + '_std'] = 0
        END IF
    END FOR

    Return mfcc_features
END FUNCTION

FUNCTION Compute_MFCC_Frame(magnitude, sr, n_mfcc=13, n_mels=26)
    TRY:
        // Create mel filter bank
        mel_filters = Create_Mel_Filterbank(length(magnitude), sr, n_mels)

        // Apply filters
        mel_spectrum = matrix_multiply(mel_filters, magnitude^2)

        // Log spectrum
        log_mel = log(mel_spectrum + 1e-10)

        // DCT (Discrete Cosine Transform)
        mfccs = DCT(log_mel, type=2, normalization='ortho')[0 : n_mfcc]

        Return mfccs
    CATCH:
        Return NULL
    END TRY
END FUNCTION

FUNCTION Create_Mel_Filterbank(nfft, sr, n_mels)
    // Mel scale conversion functions
    FUNCTION hz_to_mel(hz):
        Return 2595 * log10(1 + hz / 700)
    END FUNCTION

    FUNCTION mel_to_hz(mel):
        Return 700 * (10^(mel / 2595) - 1)
    END FUNCTION

    // Mel points
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(sr / 2)
    mel_points = linear_space(low_freq_mel, high_freq_mel, n_mels + 2)
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
            END IF
        END FOR

        FOR k FROM f_m TO f_m_plus - 1:
            IF f_m_plus > f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
            END IF
        END FOR
    END FOR

    Return filterbank
END FUNCTION

FUNCTION Calculate_Additional_Features(signal, voiced_frames, sr)
    frame_length = integer(0.025 * sr)  // 25ms
    hop_length = integer(0.01 * sr)     // 10ms

    energy_values = empty_list
    zcr_values = empty_list

    FOR i FROM 0 TO length(signal) - frame_length BY hop_length:
        frame = signal[i : i + frame_length]

        // Short-term energy
        ste = sum(frame^2)
        energy_values.append(ste)

        // Zero crossing rate
        zcr = sum(abs(diff(sign(frame)))) / (2 * length(frame))
        zcr_values.append(zcr)
    END FOR

    energy_array = convert_to_array(energy_values)
    zcr_array = convert_to_array(zcr_values)

    // Voice activity
    voiced_ratio = length(voiced_frames) / max(1, length(energy_values))

    Return {
        ste_mean: mean(energy_array),
        ste_std: std(energy_array),
        ste_max: max(energy_array),
        ste_min: min(energy_array),
        zcr_mean: mean(zcr_array),
        zcr_std: std(zcr_array),
        zcr_max: max(zcr_array),
        zcr_min: min(zcr_array),
        voiced_ratio: voiced_ratio,
        energy_entropy: Calculate_Energy_Entropy(energy_array)
    }
END FUNCTION

FUNCTION Calculate_Energy_Entropy(energy_values)
    IF length(energy_values) < 2:
        Return 0
    END IF

    total_energy = sum(energy_values)
    IF total_energy == 0:
        Return 0
    END IF

    energy_prob = energy_values / total_energy
    energy_prob = energy_prob[energy_prob > 0]

    Return -sum(energy_prob * log2(energy_prob))
END FUNCTION

// ========================================
// FINAL PROCESSING & OUTPUT
// ========================================

FUNCTION Save_Results_And_Analysis(results)
    Create_Directory("comprehensive_features")

    csv_file = "comprehensive_features/comprehensive_pd_features_final.csv"

    Write_CSV(results, csv_file)
    Print "✅ Comprehensive features saved to: " + csv_file

    // Analysis
    groups = Group_By(results, 'group')

    Print "📊 FEATURE EXTRACTION SUMMARY:"
    FOR group, data IN groups:
        Print "   " + group + ": " + count(data) + " files processed"
    END FOR

    Print "Total features per file: " + (count(results[0]) - 1) + " (excluding group label)"

    // Feature category breakdown
    Print_Feature_Categories(results[0])

    // Statistical comparison
    IF 'HC' IN groups AND 'PD' IN groups:
        Print_HC_vs_PD_Comparison(groups['HC'], groups['PD'])
    END IF

    Print "🎉 COMPREHENSIVE FEATURE EXTRACTION COMPLETE!"
    Print "✅ ALL REQUESTED FEATURES SUCCESSFULLY IMPLEMENTED"
    Print "🔬 Ready for machine learning analysis and classification!"
END FUNCTION

```

## 🎯 FEATURE CATEGORIES SUMMARY

### Total Features Extracted: **79 Features**

1. **MDVP Jitter Features (5)**: Period-to-period F0 variations
2. **MDVP Shimmer Features (6)**: Amplitude variations
3. **Voice Quality Features (2)**: Noise ratios (NHR, HNR)
4. **Prosodic Features (6)**: F0 statistics (mean, max, min, etc.)
5. **Nonlinear Features (6)**: Complexity measures (RPDE, DFA, PPE, etc.)
6. **Spectral Features (10)**: Frequency domain analysis
7. **MFCC Features (26)**: Mel-frequency cepstral coefficients (13 mean + 13 std)
8. **Signal Processing Features (10)**: Energy, ZCR, entropy measures
9. **Metadata (8)**: File information and processing stats

## 🔄 PROCESSING FLOW SUMMARY

```
Audio Input → Preprocessing → Voice Detection → Pitch Analysis →
Feature Extraction (8 Categories) → Statistical Analysis → CSV Output
```

**Ei pseudocode-e pura feature extraction process er complete flow dekhano hoyeche!**

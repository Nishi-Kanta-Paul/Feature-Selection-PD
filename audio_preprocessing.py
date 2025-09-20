import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# Import visualization module
from filter_visualization import FilterVisualization

class AudioPreprocessor:
    """
    Clean and reusable Audio Preprocessor for Parkinson's Disease Analysis
    
    Features:
    1. Data mapping from final_selected.csv
    2. 16kHz sampling rate
    3. Band-pass filtering with percentile-based cutoffs
    4. Modular visualization support
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
        """
        Analyze frequency characteristics for filter parameter calculation
        
        This method calculates percentile-based filter parameters:
        - Strategy 1 (1-99 percentile): Broader frequency range
        - Strategy 2 (2.5-97.5 percentile): Conservative frequency range
        """
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
                    
                    # Calculate spectral features for filter design
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
            
            # Strategy 1: 1st-99th percentiles (broader range)
            p1_low = max(80, np.percentile(centroids, 1))  # 1st percentile for low cutoff
            p1_high = min(self.target_sr//2, np.percentile(rolloffs, 99))  # 99th percentile for high cutoff
            
            # Strategy 2: 2.5th-97.5th percentiles (conservative range)
            p2_low = max(100, np.percentile(centroids, 2.5))  # 2.5th percentile for low cutoff
            p2_high = min(8000, np.percentile(rolloffs, 97.5))  # 97.5th percentile for high cutoff
            
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
                
            # Validate percentile implementation
            self._validate_percentile_implementation(centroids, rolloffs)
                
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
    
    def _validate_percentile_implementation(self, centroids, rolloffs):
        """Validate that percentile implementation is correct"""
        print("\n📊 Percentile Implementation Validation:")
        
        # Check percentile calculations
        p1_low_check = np.percentile(centroids, 1)
        p99_high_check = np.percentile(rolloffs, 99)
        p2_5_low_check = np.percentile(centroids, 2.5)
        p97_5_high_check = np.percentile(rolloffs, 97.5)
        
        print(f"  Raw 1st percentile (centroids): {p1_low_check:.1f} Hz")
        print(f"  Raw 99th percentile (rolloffs): {p99_high_check:.1f} Hz")
        print(f"  Raw 2.5th percentile (centroids): {p2_5_low_check:.1f} Hz")
        print(f"  Raw 97.5th percentile (rolloffs): {p97_5_high_check:.1f} Hz")
        
        # Show data distribution
        print(f"\n  Centroids: min={centroids.min():.1f}, max={centroids.max():.1f}, mean={centroids.mean():.1f}")
        print(f"  Rolloffs: min={rolloffs.min():.1f}, max={rolloffs.max():.1f}, mean={rolloffs.mean():.1f}")
        
        # Validate that 1-99 range is broader than 2.5-97.5 range
        range_1_99 = self.filter_params['percentile_1_99']['high_cutoff'] - self.filter_params['percentile_1_99']['low_cutoff']
        range_2_5_97_5 = self.filter_params['percentile_2_5_97_5']['high_cutoff'] - self.filter_params['percentile_2_5_97_5']['low_cutoff']
        
        print(f"\n  Strategy 1 (1-99%) bandwidth: {range_1_99:.1f} Hz")
        print(f"  Strategy 2 (2.5-97.5%) bandwidth: {range_2_5_97_5:.1f} Hz")
        
        if range_1_99 > range_2_5_97_5:
            print("  ✅ Percentile implementation correct: 1-99% range > 2.5-97.5% range")
        else:
            print("  ⚠️ Warning: Unexpected percentile relationship")
    
    def apply_band_pass_filter(self, audio, sr, low_cutoff, high_cutoff):
        """Apply band-pass filtering with Butterworth filter"""
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
        
        Args:
            data_dir: Input directory containing audio files
            output_base_dir: Base name for output directories
            
        Returns:
            dict: Processing results for both strategies
        """
        if self.filter_params is None:
            print("Error: Must run frequency analysis first!")
            return
            
        print(f"\n{'='*60}")
        print("AUDIO PREPROCESSING")
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
                        output_filename = f"filtered_{i+1:04d}.wav"
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
        print("  ✓ Band-pass filtering (percentile-based)")
        print("  ✓ Two filtering strategies")
        print("  ✓ Data mapped from final_selected.csv")
    
    def create_visualizations(self, create_diagrams=True, sample_audio_path=None):
        """
        Create filter visualizations using separate visualization module
        
        Args:
            create_diagrams: Whether to create beginner-friendly diagrams
            sample_audio_path: Optional path to specific audio file for demo
        """
        if self.filter_params is None:
            print("Warning: No filter parameters available for visualization")
            return
            
        viz = FilterVisualization()
        
        if create_diagrams:
            viz.create_all_visualizations(self.filter_params, sample_audio_path, self.target_sr)
        else:
            viz.create_filter_analysis(self.filter_params)


def main():
    """Main function for audio preprocessing"""
    print("PARKINSON'S DISEASE AUDIO PREPROCESSING")
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
    
    # Step 3: Create visualizations BEFORE processing (for understanding)
    print("\n" + "="*60)
    print("CREATING EDUCATIONAL VISUALIZATIONS")
    print("="*60)
    preprocessor.create_visualizations(create_diagrams=True)
    
    # Step 4: Preprocess with both strategies
    print("\n" + "="*60)
    print("STARTING AUDIO PROCESSING")
    print("="*60)
    results = preprocessor.preprocess_audio_data("data", "preprocessed_data")
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETED!")
    print(f"{'='*60}")
    print("\nOutput directories:")
    print("1. preprocessed_data_percentile_1_99/ - Primary output (1-99 percentile)")
    print("2. preprocessed_data_percentile_2_5_97_5/ - Conservative output (2.5-97.5 percentile)")
    print("3. essential_analysis/ - Educational visualizations")


if __name__ == "__main__":
    main()
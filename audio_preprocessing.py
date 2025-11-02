import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """
    Clean and reusable Audio Preprocessor for Parkinson's Disease Analysis
    
    CORRECTED WORKFLOW:
    1. Data mapping from final_selected.csv
    2. Convert to 16kHz mono WAV (NO filtering at this stage)
    3. Save for feature extraction
    4. Percentile-based filtering happens AFTER feature extraction
    
    This preprocessor only handles:
    - Format standardization (16kHz, mono, WAV)
    - Basic quality checks
    - File organization
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.data_mapping = None
        
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
    
    def preprocess_audio_basic(self, data_dir="data", output_dir="preprocessed_data_basic"):
        """
        Basic preprocessing: Only format standardization
        NO FILTERING - Just convert to 16kHz mono WAV
        
        This is the CORRECT first step before feature extraction
        """
        print(f"\n{'='*60}")
        print("BASIC AUDIO PREPROCESSING (Format Standardization Only)")
        print(f"{'='*60}")
        print("Note: NO filtering applied at this stage")
        print("Percentile filtering will be done AFTER feature extraction")
        print(f"{'='*60}\n")
        
        # Create output directory
        for cohort in ["PD", "HC"]:
            os.makedirs(os.path.join(output_dir, cohort), exist_ok=True)
        
        processed_count = {"PD": 0, "HC": 0}
        
        for cohort in ["PD", "HC"]:
            input_dir = os.path.join(data_dir, cohort)
            if not os.path.exists(input_dir):
                print(f"Warning: {input_dir} not found")
                continue
                
            audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
            print(f"Processing {len(audio_files)} {cohort} files...")
            
            for i, filename in enumerate(audio_files):
                try:
                    input_path = os.path.join(input_dir, filename)
                    output_filename = f"preprocessed_{i+1:04d}.wav"
                    output_path = os.path.join(output_dir, cohort, output_filename)
                    
                    # Load audio and resample to 16kHz
                    audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
                    
                    # Save as 16kHz mono WAV (NO FILTERING)
                    sf.write(output_path, audio, sr)
                    processed_count[cohort] += 1
                    
                except Exception as e:
                    print(f"  Error with {filename}: {e}")
        
        print(f"\nBasic preprocessing completed!")
        print(f"  PD: {processed_count['PD']} files")
        print(f"  HC: {processed_count['HC']} files")
        print(f"  Output: {output_dir}/")
        print("\nNext step: Run feature extraction on these files")
        
        return processed_count
    
    def preprocess_audio_data(self, data_dir="data", output_base_dir="preprocessed_data"):
        """
        DEPRECATED: This function applies filtering at preprocessing stage
        
        CORRECT APPROACH: Use preprocess_audio_basic() instead
        Filtering should be done AFTER feature extraction, not before
        """
        print("⚠️ WARNING: This function is DEPRECATED")
        print("❌ Filtering should NOT be done at preprocessing stage")
        print("✅ Use preprocess_audio_basic() instead")
        print("\nCorrect workflow:")
        print("  1. preprocess_audio_basic() - Format standardization only")
        print("  2. Feature extraction")
        print("  3. Percentile-based feature filtering (on extracted features)")
        
        return None


def main():
    """Main function for BASIC audio preprocessing (NO FILTERING)"""
    print("PARKINSON'S DISEASE AUDIO PREPROCESSING")
    print("="*60)
    print("CORRECTED WORKFLOW - Basic Preprocessing Only")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(target_sr=16000)
    
    # Step 1: Load data mapping
    data_mapping = preprocessor.load_data_mapping()
    if data_mapping is None:
        print("Failed to load data mapping. Exiting.")
        return
    
    # Step 2: Basic preprocessing (16kHz conversion only, NO FILTERING)
    print("\n" + "="*60)
    print("BASIC PREPROCESSING - Format Standardization")
    print("="*60)
    results = preprocessor.preprocess_audio_basic("data", "preprocessed_data_basic")
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETED!")
    print(f"{'='*60}")
    print("\nOutput directory:")
    print("  preprocessed_data_basic/ - 16kHz mono WAV files (NO filtering)")
    print("\nKey features:")
    print("  ✓ 16kHz sampling rate")
    print("  ✓ Mono channel")
    print("  ✓ WAV format")
    print("  ✓ Data mapped from final_selected.csv")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. ✅ Basic preprocessing complete")
    print("2. → Run feature extraction: python comprehensive_pd_features.py")
    print("3. → Apply percentile filtering on extracted features")
    print("4. → Feature selection and model training")


if __name__ == "__main__":
    main()
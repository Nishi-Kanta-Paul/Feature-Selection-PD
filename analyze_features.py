import pandas as pd
import numpy as np

def analyze_extracted_features():
    """
    Analyze the extracted features and display summary statistics
    """
    
    # Load the extracted features
    try:
        df = pd.read_csv('extracted_features.csv')
        print("=== FEATURE EXTRACTION ANALYSIS ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Samples: {len(df)}")
        print(f"Features: {len(df.columns) - 4}")  # Excluding metadata columns
        
        # Cohort distribution
        print(f"\nCohort Distribution:")
        cohort_counts = df['cohort'].value_counts()
        for cohort, count in cohort_counts.items():
            print(f"  {cohort}: {count} samples")
        
        # Feature categories analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in ['cohort_numeric']]
        
        print(f"\nFeature Categories:")
        categories = {
            'Time Domain': [f for f in feature_cols if any(td in f for td in ['mean_amplitude', 'std_amplitude', 'rms_energy', 'zcr', 'energy', 'duration'])],
            'Frequency Domain': [f for f in feature_cols if any(fd in f for fd in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness'])],
            'MFCC': [f for f in feature_cols if 'mfcc' in f],
            'Spectral': [f for f in feature_cols if any(sp in f for sp in ['mel_', 'chroma_', 'contrast_', 'tonnetz_'])],
            'Prosodic': [f for f in feature_cols if any(pr in f for pr in ['f0_', 'jitter', 'voiced_ratio', 'hnr'])]
        }
        
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
            if len(features) <= 5:  # Show feature names for smaller categories
                for feature in features:
                    print(f"    - {feature}")
        
        # Statistical summary for key features
        print(f"\n=== KEY FEATURE STATISTICS ===")
        
        key_features = [
            'mean_amplitude', 'rms_energy', 'zcr_mean', 
            'spectral_centroid', 'spectral_bandwidth',
            'mfcc_1_mean', 'f0_mean', 'voiced_ratio'
        ]
        
        available_key_features = [f for f in key_features if f in df.columns]
        
        if available_key_features:
            for feature in available_key_features:
                print(f"\n{feature}:")
                overall_stats = df[feature].describe()
                print(f"  Overall: Mean={overall_stats['mean']:.4f}, Std={overall_stats['std']:.4f}")
                
                # By cohort
                for cohort in ['PD', 'HC']:
                    cohort_data = df[df['cohort'] == cohort][feature]
                    if len(cohort_data) > 0:
                        print(f"  {cohort}: Mean={cohort_data.mean():.4f}, Std={cohort_data.std():.4f}")
        
        # Missing values check
        print(f"\n=== DATA QUALITY CHECK ===")
        missing_data = df[feature_cols].isnull().sum()
        features_with_missing = missing_data[missing_data > 0]
        
        if len(features_with_missing) > 0:
            print(f"Features with missing values:")
            for feature, count in features_with_missing.items():
                print(f"  {feature}: {count} missing")
        else:
            print("No missing values found in features!")
        
        # Basic feature range analysis
        print(f"\n=== FEATURE RANGE ANALYSIS ===")
        feature_ranges = {}
        for feature in feature_cols[:10]:  # First 10 features
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            feature_range = feature_max - feature_min
            feature_ranges[feature] = {
                'min': feature_min,
                'max': feature_max,
                'range': feature_range
            }
        
        for feature, stats in feature_ranges.items():
            print(f"{feature}: [{stats['min']:.4f}, {stats['max']:.4f}] (range: {stats['range']:.4f})")
        
        print(f"\n=== FILES GENERATED ===")
        print(f"1. extracted_features.csv - Complete feature dataset")
        print(f"2. feature_analysis/ - Visualization directory with 5 plots")
        print(f"   - pipeline_diagram.png")
        print(f"   - feature_distributions.png")
        print(f"   - correlation_matrix.png")
        print(f"   - pd_vs_hc_comparison.png")
        print(f"   - feature_importance.png")
        
        return df
        
    except FileNotFoundError:
        print("extracted_features.csv not found. Please run feature_extraction.py first.")
        return None
    except Exception as e:
        print(f"Error analyzing features: {e}")
        return None

if __name__ == "__main__":
    analyze_extracted_features()

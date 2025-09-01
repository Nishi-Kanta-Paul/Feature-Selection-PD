"""
Enhanced Frequency Filtering Demo for Parkinson's Disease Audio Analysis

This script demonstrates the implementation of frequency filtering using dataset percentiles
to determine optimal cutoff frequencies for high-pass and band-pass filters.

Key Features:
1. Dataset-wide frequency analysis
2. Percentile-based filter parameter calculation (1st-99th, 2.5th-97.5th)
3. Multiple filtering strategies (broad, conservative, voice-optimized)
4. Comprehensive feature extraction with filtering
5. Comparison analysis between filtered and unfiltered approaches

Usage:
    python demo_enhanced_filtering.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_extraction import AudioFeatureExtractor


def demonstrate_frequency_filtering():
    """Demonstrate the enhanced frequency filtering capabilities"""
    
    print("="*80)
    print("ENHANCED FREQUENCY FILTERING DEMONSTRATION")
    print("Parkinson's Disease Audio Analysis")
    print("="*80)
    
    # Check if preprocessed data exists
    if not os.path.exists("preprocessed_data"):
        print("Error: preprocessed_data directory not found!")
        print("Please run audio_preprocessing.py first to create preprocessed audio files.")
        return
    
    # Step 1: Initialize feature extractor with enhanced filtering
    print("\nStep 1: Initializing AudioFeatureExtractor with enhanced filtering...")
    extractor = AudioFeatureExtractor(sr=16000, enable_enhanced_filtering=True)
    
    # Step 2: Analyze dataset frequencies
    print("\nStep 2: Analyzing dataset frequencies to determine optimal filter parameters...")
    filter_params = extractor.analyze_dataset_frequencies("preprocessed_data")
    
    if filter_params is None:
        print("Failed to analyze frequencies. Exiting.")
        return
    
    # Step 3: Display calculated filter parameters
    print("\nStep 3: Filter Parameters Based on Dataset Percentiles")
    print("-" * 60)
    
    for filter_type, params in filter_params.items():
        if filter_type != 'raw_data':
            print(f"{filter_type.replace('_', ' ').title()}:")
            print(f"  {params['description']}")
            print(f"  Low cutoff: {params['low_cutoff']:.1f} Hz")
            print(f"  High cutoff: {params['high_cutoff']:.1f} Hz")
            print()
    
    # Step 4: Extract features with different filtering approaches
    print("\nStep 4: Extracting features with different filtering approaches...")
    
    results = {}
    filter_types = ["conservative_filter", "voice_optimized_filter"]
    
    for filter_type in filter_types:
        print(f"\nProcessing with {filter_type.replace('_', ' ')}...")
        
        # Extract features
        features_df = extractor.process_dataset(
            "preprocessed_data", 
            filter_type=filter_type,
            enable_frequency_analysis=False  # Already done
        )
        
        if features_df is not None:
            # Save results
            output_file = f"demo_features_{filter_type}.csv"
            extractor.save_features(output_file)
            
            # Store results for comparison
            numerical_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col != 'cohort_numeric']
            
            results[filter_type] = {
                'dataframe': features_df,
                'total_samples': len(features_df),
                'pd_samples': len(features_df[features_df['cohort'] == 'PD']),
                'hc_samples': len(features_df[features_df['cohort'] == 'HC']),
                'total_features': len(feature_cols),
                'filter_params': filter_params[filter_type]
            }
            
            print(f"  Saved features to: {output_file}")
    
    # Step 5: Create comparison visualizations
    print("\nStep 5: Creating comparison visualizations...")
    create_filtering_comparison_plots(results, filter_params)
    
    # Step 6: Summary and recommendations
    print("\nStep 6: Summary and Recommendations")
    print("="*60)
    
    print("\nFilter Parameter Summary:")
    for filter_type, result in results.items():
        params = result['filter_params']
        print(f"\n{filter_type.replace('_', ' ').title()}:")
        print(f"  Frequency range: {params['low_cutoff']:.1f} - {params['high_cutoff']:.1f} Hz")
        print(f"  Samples processed: {result['total_samples']}")
        print(f"  Features extracted: {result['total_features']}")
    
    print("\nRecommendations:")
    print("- Conservative Filter: Best for general-purpose analysis")
    print("- Voice Optimized Filter: Best for speech-specific features")
    print("- Use percentile-based filtering to adapt to your specific dataset")
    
    return results


def create_filtering_comparison_plots(results, filter_params):
    """Create comparison plots between different filtering approaches"""
    
    if len(results) < 2:
        print("Need at least 2 filter results for comparison")
        return
    
    os.makedirs("filtering_comparison", exist_ok=True)
    
    # 1. Filter frequency response comparison
    create_filter_response_plot(filter_params)
    
    # 2. Feature comparison between filter types
    create_feature_comparison_plot(results)
    
    # 3. Statistical comparison
    create_statistical_comparison_plot(results)
    
    print("Comparison plots saved to: filtering_comparison/")


def create_filter_response_plot(filter_params):
    """Create filter frequency response comparison plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Frequency range for plotting
    freqs = np.logspace(1, np.log10(8000), 1000)  # 10 Hz to 8 kHz
    sr = 16000
    
    filter_types = ['conservative_filter', 'voice_optimized_filter', 'broad_filter']
    colors = ['blue', 'red', 'green']
    
    # Plot individual filter responses
    for i, (filter_type, color) in enumerate(zip(filter_types[:3], colors)):
        if filter_type not in filter_params:
            continue
            
        params = filter_params[filter_type]
        row, col = i // 2, i % 2
        
        if i < 3:
            ax = axes[row, col] if i < 2 else axes[1, 0]
            
            # Simulate filter response (simplified)
            low_cutoff = params['low_cutoff']
            high_cutoff = params['high_cutoff']
            
            # High-pass response
            hp_response = 1 / (1 + (low_cutoff / freqs)**6)  # 3rd order approximation
            
            # Band-pass response (if applicable)
            if high_cutoff < sr / 2:
                lp_response = 1 / (1 + (freqs / high_cutoff)**6)  # Low-pass component
                bp_response = hp_response * lp_response
            else:
                bp_response = hp_response
            
            ax.semilogx(freqs, 20 * np.log10(hp_response + 1e-10), 
                       '--', color=color, label=f'High-pass ({low_cutoff:.0f} Hz)', linewidth=2)
            ax.semilogx(freqs, 20 * np.log10(bp_response + 1e-10), 
                       '-', color=color, label=f'Band-pass ({low_cutoff:.0f}-{high_cutoff:.0f} Hz)', linewidth=2)
            
            ax.set_title(f'{filter_type.replace("_", " ").title()}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([-60, 5])
    
    # Combined comparison
    ax = axes[1, 1]
    for filter_type, color in zip(filter_types, colors):
        if filter_type not in filter_params:
            continue
            
        params = filter_params[filter_type]
        low_cutoff = params['low_cutoff']
        high_cutoff = params['high_cutoff']
        
        hp_response = 1 / (1 + (low_cutoff / freqs)**6)
        if high_cutoff < sr / 2:
            lp_response = 1 / (1 + (freqs / high_cutoff)**6)
            bp_response = hp_response * lp_response
        else:
            bp_response = hp_response
        
        ax.semilogx(freqs, 20 * np.log10(bp_response + 1e-10), 
                   color=color, linewidth=2, label=filter_type.replace('_', ' ').title())
    
    ax.set_title('Filter Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-60, 5])
    
    plt.suptitle('Percentile-Based Filter Frequency Responses', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('filtering_comparison/filter_responses.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_feature_comparison_plot(results):
    """Create feature comparison plot between different filters"""
    
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get common features across all filter types
    filter_types = list(results.keys())
    dfs = [results[ft]['dataframe'] for ft in filter_types]
    
    # Select a few key features for comparison
    key_features = ['spectral_centroid', 'mfcc_1_mean', 'f0_mean', 'rms_energy']
    
    for i, feature in enumerate(key_features):
        if i >= 4:
            break
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot distributions for each filter type
        for j, (filter_type, df) in enumerate(zip(filter_types, dfs)):
            if feature in df.columns:
                # Separate PD and HC
                pd_data = df[df['cohort'] == 'PD'][feature].dropna()
                hc_data = df[df['cohort'] == 'HC'][feature].dropna()
                
                # Plot histograms
                ax.hist(pd_data, bins=15, alpha=0.5, label=f'{filter_type} - PD', 
                       color='red' if j == 0 else 'darkred', density=True)
                ax.hist(hc_data, bins=15, alpha=0.5, label=f'{filter_type} - HC', 
                       color='blue' if j == 0 else 'darkblue', density=True)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Comparison: Different Filtering Approaches', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('filtering_comparison/feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_statistical_comparison_plot(results):
    """Create statistical comparison between filtering approaches"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    filter_types = list(results.keys())
    
    # 1. Sample distribution comparison
    ax1 = axes[0]
    x_labels = []
    pd_counts = []
    hc_counts = []
    
    for filter_type in filter_types:
        x_labels.append(filter_type.replace('_', '\n'))
        pd_counts.append(results[filter_type]['pd_samples'])
        hc_counts.append(results[filter_type]['hc_samples'])
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax1.bar(x - width/2, pd_counts, width, label='PD', color='red', alpha=0.7)
    ax1.bar(x + width/2, hc_counts, width, label='HC', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Filter Type')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Sample Distribution by Filter Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature count comparison
    ax2 = axes[1]
    feature_counts = [results[ft]['total_features'] for ft in filter_types]
    
    bars = ax2.bar(x_labels, feature_counts, color=['skyblue', 'lightcoral'][:len(filter_types)], alpha=0.7)
    ax2.set_xlabel('Filter Type')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Count by Filter Type')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Statistical Comparison: Filtering Approaches', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('filtering_comparison/statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main demo function"""
    try:
        results = demonstrate_frequency_filtering()
        
        if results:
            print(f"\n{'='*80}")
            print("DEMO COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print("\nGenerated files:")
            print("- demo_features_conservative_filter.csv")
            print("- demo_features_voice_optimized_filter.csv")
            print("- filtering_comparison/ (visualization directory)")
            print("\nNext steps:")
            print("1. Examine the generated CSV files to see extracted features")
            print("2. Review visualizations in filtering_comparison/ directory")
            print("3. Use the conservative_filter for your machine learning pipeline")
        else:
            print("Demo failed to complete successfully.")
            
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please ensure you have the required dependencies and preprocessed data.")


if __name__ == "__main__":
    main()

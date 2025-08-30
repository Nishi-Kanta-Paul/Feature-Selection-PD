import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_complete_pipeline():
    """
    Comprehensive analysis of the complete Parkinson's Disease detection pipeline
    """
    
    print("🔬 PARKINSON'S DISEASE DETECTION - COMPLETE PIPELINE ANALYSIS")
    print("=" * 80)
    
    # Load feature selection results
    try:
        results_df = pd.read_csv('feature_selection_results.csv')
        features_df = pd.read_csv('extracted_features.csv')
        
        print("✅ Data loaded successfully")
        print(f"   - Feature selection results: {len(results_df)} features ranked")
        print(f"   - Original dataset: {len(features_df)} samples, {len(features_df.columns)-4} features")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Pipeline summary
    print(f"\n📊 PIPELINE SUMMARY")
    print("=" * 50)
    
    # Dataset statistics
    cohort_counts = features_df['cohort'].value_counts()
    print(f"Dataset composition:")
    for cohort, count in cohort_counts.items():
        print(f"   - {cohort}: {count} samples")
    
    # Feature categories (original)
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col not in ['cohort_numeric']]
    
    categories = {
        'Time Domain': [f for f in feature_cols if any(td in f for td in ['mean_amplitude', 'std_amplitude', 'rms_energy', 'zcr', 'energy', 'duration'])],
        'Frequency Domain': [f for f in feature_cols if any(fd in f for fd in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness'])],
        'MFCC': [f for f in feature_cols if 'mfcc' in f],
        'Spectral': [f for f in feature_cols if any(sp in f for sp in ['mel_', 'chroma_', 'contrast_', 'tonnetz_'])],
        'Prosodic': [f for f in feature_cols if any(pr in f for pr in ['f0_', 'jitter', 'voiced_ratio', 'hnr'])]
    }
    
    print(f"\nOriginal feature categories:")
    total_features = 0
    for category, features in categories.items():
        print(f"   - {category}: {len(features)} features")
        total_features += len(features)
    print(f"   - Total: {total_features} features")
    
    # Top selected features analysis
    print(f"\n🏆 TOP SELECTED FEATURES")
    print("=" * 50)
    
    top_features = results_df.head(15)
    print(f"Top 15 features by combined ranking:")
    
    for i, row in top_features.iterrows():
        # Determine feature category
        feature_name = row['feature']
        category = 'Other'
        for cat, cat_features in categories.items():
            if feature_name in cat_features:
                category = cat
                break
        
        print(f"   {row['rank']:2d}. {feature_name:<30} | Score: {row['combined_score']:.4f} | {category}")
    
    # Feature category analysis of top features
    print(f"\n📈 FEATURE CATEGORY ANALYSIS (Top 50)")
    print("=" * 50)
    
    top_50 = results_df.head(50)
    category_counts = {cat: 0 for cat in categories.keys()}
    category_counts['Other'] = 0
    
    for _, row in top_50.iterrows():
        feature_name = row['feature']
        categorized = False
        for cat, cat_features in categories.items():
            if feature_name in cat_features:
                category_counts[cat] += 1
                categorized = True
                break
        if not categorized:
            category_counts['Other'] += 1
    
    print("Distribution of top 50 features by category:")
    for category, count in category_counts.items():
        percentage = (count / 50) * 100
        print(f"   - {category:<15}: {count:2d} features ({percentage:5.1f}%)")
    
    # Statistical significance analysis
    print(f"\n📊 STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # Analyze statistical scores
    f_classif_significant = len(results_df[results_df['f_classif_score'] > 0])
    mutual_info_scores = results_df['mutual_info_score']
    
    print(f"Statistical test results:")
    print(f"   - F-test: {f_classif_significant} features with scores > 0")
    print(f"   - Mutual Information: {len(mutual_info_scores[mutual_info_scores > 0])} features with MI > 0")
    print(f"   - Score range: {results_df['combined_score'].min():.4f} - {results_df['combined_score'].max():.4f}")
    
    # Performance insights
    print(f"\n🎯 KEY INSIGHTS")
    print("=" * 50)
    
    # Top feature insights
    top_feature = results_df.iloc[0]
    print(f"Best discriminative feature: {top_feature['feature']}")
    print(f"   - Combined score: {top_feature['combined_score']:.4f}")
    print(f"   - Mutual Information: {top_feature['mutual_info_score']:.4f}")
    
    # MFCC dominance
    mfcc_in_top_10 = len([f for f in top_features['feature'] if 'mfcc' in f])
    print(f"\nMFCC feature dominance:")
    print(f"   - {mfcc_in_top_10}/15 top features are MFCC-based ({(mfcc_in_top_10/15)*100:.1f}%)")
    
    # Spectral features importance
    spectral_features = ['mel_std', 'mel_mean', 'chroma_mean', 'chroma_std', 'contrast_mean', 'contrast_std']
    spectral_in_top = len([f for f in top_features['feature'] if f in spectral_features])
    print(f"   - {spectral_in_top}/15 top features are spectral-based ({(spectral_in_top/15)*100:.1f}%)")
    
    # Prosodic features
    prosodic_features = [f for f in top_features['feature'] if any(pr in f for pr in ['f0_', 'voiced_ratio', 'jitter', 'hnr'])]
    print(f"   - {len(prosodic_features)}/15 top features are prosodic-based ({(len(prosodic_features)/15)*100:.1f}%)")
    
    # Create summary visualization
    create_summary_visualization(results_df, category_counts)
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("For optimal Parkinson's Disease detection:")
    print("1. Focus on MFCC features (especially delta and delta-delta coefficients)")
    print("2. Include spectral statistics (mel-frequency standard deviation)")
    print("3. Consider voice activity and prosodic measures")
    print("4. Use combined filter approach for robust feature selection")
    print("5. Validate with larger datasets for better generalization")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("📁 Summary visualization saved as: pipeline_summary.png")

def create_summary_visualization(results_df, category_counts):
    """Create a comprehensive summary visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top 20 features bar plot
    top_20 = results_df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['combined_score'], color='steelblue', alpha=0.8)
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels([f[:25] for f in top_20['feature']], fontsize=8)
    axes[0, 0].set_xlabel('Combined Filter Score')
    axes[0, 0].set_title('Top 20 Features for PD Detection', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature category distribution
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    axes[0, 1].pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 1].set_title('Feature Category Distribution\n(Top 50 Features)', fontweight='bold')
    
    # 3. Score distribution
    axes[1, 0].hist(results_df['combined_score'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(results_df['combined_score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {results_df["combined_score"].mean():.4f}')
    axes[1, 0].set_xlabel('Combined Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Feature Score Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Method comparison (using available score columns)
    methods = ['f_classif_score', 'mutual_info_score']
    method_means = [results_df[method].mean() for method in methods]
    method_names = ['F-test', 'Mutual Info']
    
    bars = axes[1, 1].bar(method_names, method_means, color=['lightcoral', 'lightskyblue'], alpha=0.8)
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].set_title('Filter Method Performance', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, method_means):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.suptitle('Parkinson\'s Disease Detection - Feature Selection Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('pipeline_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_complete_pipeline()

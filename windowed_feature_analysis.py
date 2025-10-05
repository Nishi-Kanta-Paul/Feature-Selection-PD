#!/usr/bin/env python3
"""
WINDOWED FEATURE ANALYSIS AND SELECTION
=======================================
Analyzes window-based features for Parkinson's Disease detection
and applies feature selection to identify optimal temporal scales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WindowedFeatureAnalyzer:
    """
    Comprehensive analysis tool for windowed voice features
    """
    
    def __init__(self, csv_file):
        """Initialize with windowed features CSV"""
        self.df = pd.read_csv(csv_file)
        self.window_sizes = self._detect_window_sizes()
        self.feature_categories = self._categorize_features()
        
        print(f"🔍 Windowed Feature Analyzer Initialized")
        print(f"   Samples: {len(self.df)}")
        print(f"   Features: {len(self.df.columns)}")
        print(f"   Window sizes: {self.window_sizes}")
        
    def _detect_window_sizes(self):
        """Detect available window sizes from column names"""
        window_sizes = []
        for col in self.df.columns:
            if col.startswith('w') and 'ms_' in col:
                size_str = col.split('ms_')[0][1:]  # Remove 'w' and get size
                try:
                    size = int(size_str)
                    if size not in window_sizes:
                        window_sizes.append(size)
                except ValueError:
                    continue
        return sorted(window_sizes)
    
    def _categorize_features(self):
        """Categorize features by type and statistics"""
        categories = {}
        
        for window_size in self.window_sizes:
            categories[window_size] = {
                'feature_types': {},
                'stat_types': set()
            }
            
            prefix = f"w{window_size}ms_"
            window_cols = [col for col in self.df.columns if col.startswith(prefix)]
            
            for col in window_cols:
                # Parse feature name: w5ms_energy_mean -> feature=energy, stat=mean
                parts = col.replace(prefix, '').split('_')
                if len(parts) >= 2:
                    feature_name = '_'.join(parts[:-1])
                    stat_name = parts[-1]
                    
                    if feature_name not in categories[window_size]['feature_types']:
                        categories[window_size]['feature_types'][feature_name] = []
                    
                    categories[window_size]['feature_types'][feature_name].append(stat_name)
                    categories[window_size]['stat_types'].add(stat_name)
        
        return categories
    
    def analyze_feature_distribution(self):
        """Analyze feature distribution across window sizes"""
        print("\n📊 FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        for window_size in self.window_sizes:
            prefix = f"w{window_size}ms_"
            window_cols = [col for col in self.df.columns if col.startswith(prefix)]
            
            print(f"\n🪟 {window_size}ms Windows ({len(window_cols)} features)")
            
            # Feature types
            feature_types = self.feature_categories[window_size]['feature_types']
            print(f"   Feature types: {len(feature_types)}")
            
            # Most common feature types
            for feat_type, stats in list(feature_types.items())[:5]:
                print(f"     {feat_type}: {len(stats)} statistics")
            
            # Statistics types
            stat_types = self.feature_categories[window_size]['stat_types']
            print(f"   Statistic types: {sorted(stat_types)}")
            
            # Data quality
            numeric_cols = [col for col in window_cols if self.df[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                non_zero = (self.df[numeric_cols] != 0).sum().sum()
                total = len(self.df) * len(numeric_cols)
                print(f"   Non-zero values: {non_zero}/{total} ({100*non_zero/total:.1f}%)")
                
                # Value ranges
                means = self.df[numeric_cols].mean().mean()
                stds = self.df[numeric_cols].std().mean()
                print(f"   Average feature mean: {means:.4f}")
                print(f"   Average feature std: {stds:.4f}")
    
    def compare_window_discriminative_power(self):
        """Compare discriminative power across window sizes using Random Forest"""
        print("\n🎯 DISCRIMINATIVE POWER COMPARISON")
        print("=" * 50)
        
        if 'group' not in self.df.columns:
            print("❌ No 'group' column found for classification")
            return
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(self.df['group'])
        
        results = {}
        
        for window_size in self.window_sizes:
            prefix = f"w{window_size}ms_"
            window_cols = [col for col in self.df.columns if col.startswith(prefix)]
            
            # Get numeric features only
            numeric_cols = []
            for col in window_cols:
                if self.df[col].dtype in ['int64', 'float64']:
                    numeric_cols.append(col)
            
            if not numeric_cols:
                print(f"⚠️  No numeric features found for {window_size}ms")
                continue
            
            X = self.df[numeric_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=3,
                class_weight='balanced'
            )
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(rf, X_scaled, y, cv=3, scoring='accuracy')
                
                # Feature importance
                rf.fit(X_scaled, y)
                feature_importance = rf.feature_importances_
                
                results[window_size] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'num_features': len(numeric_cols),
                    'feature_importance': feature_importance,
                    'feature_names': numeric_cols,
                    'top_features': self._get_top_features(numeric_cols, feature_importance, top_k=10)
                }
                
                print(f"\n🪟 {window_size}ms Windows:")
                print(f"   Features used: {len(numeric_cols)}")
                print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   Top 5 features:")
                
                for i, (feat_name, importance) in enumerate(results[window_size]['top_features'][:5]):
                    print(f"     {i+1}. {feat_name}: {importance:.4f}")
                
            except Exception as e:
                print(f"❌ Error analyzing {window_size}ms: {e}")
                continue
        
        return results
    
    def _get_top_features(self, feature_names, importance_scores, top_k=10):
        """Get top features sorted by importance"""
        feature_importance_pairs = list(zip(feature_names, importance_scores))
        return sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    def find_optimal_window_features(self):
        """Find optimal features across all window sizes"""
        print("\n🏆 OPTIMAL WINDOW FEATURE SELECTION")
        print("=" * 50)
        
        # Collect all window features
        all_window_features = []
        feature_to_window = {}
        
        for window_size in self.window_sizes:
            prefix = f"w{window_size}ms_"
            window_cols = [col for col in self.df.columns if col.startswith(prefix)]
            
            for col in window_cols:
                if self.df[col].dtype in ['int64', 'float64']:
                    all_window_features.append(col)
                    feature_to_window[col] = window_size
        
        if not all_window_features or 'group' not in self.df.columns:
            print("❌ Insufficient data for analysis")
            return
        
        # Prepare data
        X = self.df[all_window_features].copy()
        le = LabelEncoder()
        y = le.fit_transform(self.df['group'])
        
        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model on all features
        rf = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=4,
            class_weight='balanced'
        )
        
        rf.fit(X_scaled, y)
        
        # Get feature importance
        feature_importance = rf.feature_importances_
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': all_window_features,
            'importance': feature_importance,
            'window_size': [feature_to_window[feat] for feat in all_window_features]
        }).sort_values('importance', ascending=False)
        
        print(f"🔍 Analyzed {len(all_window_features)} features across {len(self.window_sizes)} window sizes")
        
        # Top features overall
        print(f"\n🥇 TOP 15 FEATURES ACROSS ALL WINDOWS:")
        print("Rank | Feature                           | Window | Importance")
        print("-" * 70)
        
        for i, row in results_df.head(15).iterrows():
            print(f"{row.name+1:4d} | {row['feature']:32s} | {row['window_size']:4d}ms | {row['importance']:.6f}")
        
        # Window size analysis
        print(f"\n📊 FEATURE DISTRIBUTION BY WINDOW SIZE:")
        window_analysis = results_df.groupby('window_size').agg({
            'importance': ['count', 'mean', 'std', 'max']
        }).round(6)
        
        for window_size in self.window_sizes:
            window_data = results_df[results_df['window_size'] == window_size]
            top_features_count = len(window_data.head(50))  # Top 50 features
            avg_importance = window_data['importance'].mean()
            max_importance = window_data['importance'].max()
            
            print(f"  {window_size}ms: avg={avg_importance:.6f}, max={max_importance:.6f}, in_top50={top_features_count}")
        
        # Save results
        output_file = "comprehensive_features/windowed_feature_importance.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return results_df
    
    def create_feature_comparison_visualization(self, results_df):
        """Create visualizations comparing window-based features"""
        print("\n📈 CREATING FEATURE VISUALIZATIONS")
        print("=" * 40)
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Feature importance by window size
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Top features by window size
            for i, window_size in enumerate(self.window_sizes):
                ax = axes[i//2, i%2]
                
                window_data = results_df[results_df['window_size'] == window_size].head(10)
                
                bars = ax.barh(range(len(window_data)), window_data['importance'])
                ax.set_yticks(range(len(window_data)))
                ax.set_yticklabels([feat.replace(f'w{window_size}ms_', '') for feat in window_data['feature']], fontsize=8)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'Top 10 Features - {window_size}ms Windows')
                ax.invert_yaxis()
                
                # Color bars
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(j / len(bars)))
            
            # Overall comparison
            if len(self.window_sizes) < 4:
                ax = axes[1, 1]
                window_stats = results_df.groupby('window_size')['importance'].agg(['mean', 'max', 'std'])
                
                x = np.arange(len(self.window_sizes))
                width = 0.25
                
                ax.bar(x - width, window_stats['mean'], width, label='Mean Importance', alpha=0.8)
                ax.bar(x, window_stats['max'], width, label='Max Importance', alpha=0.8)
                ax.bar(x + width, window_stats['std'], width, label='Std Importance', alpha=0.8)
                
                ax.set_xlabel('Window Size (ms)')
                ax.set_ylabel('Feature Importance')
                ax.set_title('Window Size Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{ws}ms' for ws in self.window_sizes])
                ax.legend()
            
            plt.tight_layout()
            plt.savefig('comprehensive_features/windowed_feature_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Feature analysis visualization saved")
            
            # 2. Feature category distribution
            plt.figure(figsize=(12, 8))
            
            # Extract feature categories from names
            feature_categories = {}
            for _, row in results_df.head(30).iterrows():  # Top 30 features
                feature_name = row['feature']
                window_size = row['window_size']
                
                # Extract base feature name
                base_name = feature_name.replace(f'w{window_size}ms_', '').split('_')[0]
                
                if base_name not in feature_categories:
                    feature_categories[base_name] = 0
                feature_categories[base_name] += row['importance']
            
            # Plot category importance
            if feature_categories:
                categories = list(feature_categories.keys())
                importances = list(feature_categories.values())
                
                plt.barh(categories, importances)
                plt.xlabel('Cumulative Feature Importance')
                plt.title('Feature Category Importance (Top 30 Features)')
                plt.tight_layout()
                plt.savefig('comprehensive_features/feature_categories.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✅ Feature category visualization saved")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")

def main():
    """Main analysis pipeline for windowed features"""
    
    # Check if windowed features exist
    csv_file = "comprehensive_features/windowed_pd_features.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Windowed features file not found: {csv_file}")
        print("Please run windowed_feature_extraction.py first!")
        return
    
    # Initialize analyzer
    analyzer = WindowedFeatureAnalyzer(csv_file)
    
    # Perform comprehensive analysis
    print("\n🚀 STARTING WINDOWED FEATURE ANALYSIS")
    print("=" * 60)
    
    # 1. Feature distribution analysis
    analyzer.analyze_feature_distribution()
    
    # 2. Discriminative power comparison
    discriminative_results = analyzer.compare_window_discriminative_power()
    
    # 3. Find optimal features
    results_df = analyzer.find_optimal_window_features()
    
    # 4. Create visualizations
    if results_df is not None:
        analyzer.create_feature_comparison_visualization(results_df)
    
    print("\n🎉 Windowed feature analysis completed!")
    print("📁 Check 'comprehensive_features/' folder for:")
    print("   - windowed_feature_importance.csv")
    print("   - windowed_feature_analysis.png")
    print("   - feature_categories.png")

if __name__ == "__main__":
    import os
    main()
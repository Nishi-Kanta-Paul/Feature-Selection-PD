#!/usr/bin/env python3
"""
INTEGRATED WINDOWED + TRADITIONAL FEATURE SELECTION
===================================================
Combines the best windowed features with traditional full-signal features
for enhanced Parkinson's Disease detection performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class IntegratedFeatureSelector:
    """
    Combines windowed and traditional features for optimal PD detection
    """
    
    def __init__(self, traditional_csv, windowed_csv, windowed_importance_csv):
        """
        Initialize with traditional and windowed feature datasets
        
        Args:
            traditional_csv: Path to traditional comprehensive features CSV
            windowed_csv: Path to windowed features CSV
            windowed_importance_csv: Path to windowed feature importance CSV
        """
        self.traditional_df = pd.read_csv(traditional_csv)
        self.windowed_df = pd.read_csv(windowed_csv)
        self.windowed_importance = pd.read_csv(windowed_importance_csv)
        
        print(f"🔗 Integrated Feature Selector Initialized")
        print(f"   Traditional features: {len(self.traditional_df.columns)}")
        print(f"   Windowed features: {len(self.windowed_df.columns)}")
        print(f"   Samples (traditional): {len(self.traditional_df)}")
        print(f"   Samples (windowed): {len(self.windowed_df)}")
    
    def align_datasets(self):
        """Align traditional and windowed datasets by filename"""
        print("\n🔄 ALIGNING DATASETS")
        print("=" * 30)
        
        # Get common filenames
        traditional_files = set(self.traditional_df['filename'].str.lower())
        windowed_files = set(self.windowed_df['filename'].str.lower())
        
        common_files = traditional_files & windowed_files
        
        print(f"Traditional files: {len(traditional_files)}")
        print(f"Windowed files: {len(windowed_files)}")
        print(f"Common files: {len(common_files)}")
        
        if len(common_files) == 0:
            print("❌ No common files found!")
            return None, None
        
        # Filter datasets to common files
        traditional_aligned = self.traditional_df[
            self.traditional_df['filename'].str.lower().isin(common_files)
        ].copy()
        
        windowed_aligned = self.windowed_df[
            self.windowed_df['filename'].str.lower().isin(common_files)
        ].copy()
        
        # Sort by filename for proper alignment
        traditional_aligned = traditional_aligned.sort_values('filename').reset_index(drop=True)
        windowed_aligned = windowed_aligned.sort_values('filename').reset_index(drop=True)
        
        print(f"✅ Aligned datasets: {len(traditional_aligned)} samples")
        
        return traditional_aligned, windowed_aligned
    
    def select_best_windowed_features(self, top_k=50):
        """Select top windowed features based on importance scores"""
        print(f"\n🎯 SELECTING TOP {top_k} WINDOWED FEATURES")
        print("=" * 40)
        
        # Get top features
        top_windowed_features = self.windowed_importance.head(top_k)
        
        print("Selected windowed features by window size:")
        window_distribution = top_windowed_features['window_size'].value_counts().sort_index()
        for window_size, count in window_distribution.items():
            print(f"  {window_size}ms: {count} features")
        
        # Display top 10 features
        print(f"\nTop 10 windowed features:")
        for i, row in top_windowed_features.head(10).iterrows():
            feature_short = row['feature'].replace(f"w{row['window_size']}ms_", "")
            print(f"  {i+1:2d}. {feature_short} ({row['window_size']}ms): {row['importance']:.6f}")
        
        return top_windowed_features['feature'].tolist()
    
    def select_best_traditional_features(self, traditional_aligned, top_k=30):
        """Select best traditional features using Random Forest"""
        print(f"\n🏛️ SELECTING TOP {top_k} TRADITIONAL FEATURES")
        print("=" * 40)
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['group', 'filename', 'duration', 'sample_rate', 'num_voiced_frames', 'num_f0_values']
        feature_cols = [col for col in traditional_aligned.columns if col not in exclude_cols]
        
        print(f"Available traditional features: {len(feature_cols)}")
        
        # Prepare data
        X = traditional_aligned[feature_cols].copy()
        le = LabelEncoder()
        y = le.fit_transform(traditional_aligned['group'])
        
        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=4,
            class_weight='balanced'
        )
        
        rf.fit(X, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_traditional_features = feature_importance.head(top_k)
        
        print(f"Top 10 traditional features:")
        for i, row in top_traditional_features.head(10).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.6f}")
        
        return top_traditional_features['feature'].tolist()
    
    def create_integrated_dataset(self, top_traditional_features, top_windowed_features):
        """Create integrated dataset with selected features"""
        print(f"\n🔗 CREATING INTEGRATED DATASET")
        print("=" * 35)
        
        # Align datasets
        traditional_aligned, windowed_aligned = self.align_datasets()
        
        if traditional_aligned is None or windowed_aligned is None:
            return None
        
        # Verify alignment
        if not traditional_aligned['filename'].equals(windowed_aligned['filename']):
            print("⚠️  Filename mismatch, attempting to align manually...")
            # Sort both by filename
            traditional_aligned = traditional_aligned.sort_values('filename').reset_index(drop=True)
            windowed_aligned = windowed_aligned.sort_values('filename').reset_index(drop=True)
        
        # Select features from traditional dataset
        traditional_subset = traditional_aligned[['filename', 'group'] + top_traditional_features].copy()
        
        # Select features from windowed dataset
        windowed_subset = windowed_aligned[['filename'] + top_windowed_features].copy()
        
        # Merge datasets
        integrated_df = pd.merge(traditional_subset, windowed_subset, on='filename', how='inner')
        
        print(f"✅ Integrated dataset created:")
        print(f"   Samples: {len(integrated_df)}")
        print(f"   Traditional features: {len(top_traditional_features)}")
        print(f"   Windowed features: {len(top_windowed_features)}")
        print(f"   Total features: {len(top_traditional_features) + len(top_windowed_features)}")
        print(f"   Class distribution: {dict(integrated_df['group'].value_counts())}")
        
        return integrated_df
    
    def compare_performance(self, integrated_df, top_traditional_features, top_windowed_features):
        """Compare performance of different feature sets"""
        print(f"\n⚔️ PERFORMANCE COMPARISON")
        print("=" * 35)
        
        # Prepare labels
        le = LabelEncoder()
        y = le.fit_transform(integrated_df['group'])
        
        results = {}
        
        # 1. Traditional features only
        print("\n1️⃣ Traditional Features Only")
        X_traditional = integrated_df[top_traditional_features].copy()
        X_traditional = X_traditional.fillna(X_traditional.median())
        X_traditional = X_traditional.replace([np.inf, -np.inf], np.nan).fillna(X_traditional.median())
        
        scaler = StandardScaler()
        X_traditional_scaled = scaler.fit_transform(X_traditional)
        
        rf_traditional = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv_scores_traditional = cross_val_score(rf_traditional, X_traditional_scaled, y, cv=3, scoring='accuracy')
        
        results['traditional'] = {
            'cv_mean': cv_scores_traditional.mean(),
            'cv_std': cv_scores_traditional.std(),
            'num_features': len(top_traditional_features)
        }
        
        print(f"   Features: {len(top_traditional_features)}")
        print(f"   CV Accuracy: {cv_scores_traditional.mean():.3f} ± {cv_scores_traditional.std():.3f}")
        
        # 2. Windowed features only
        print("\n2️⃣ Windowed Features Only")
        X_windowed = integrated_df[top_windowed_features].copy()
        X_windowed = X_windowed.fillna(X_windowed.median())
        X_windowed = X_windowed.replace([np.inf, -np.inf], np.nan).fillna(X_windowed.median())
        
        scaler = StandardScaler()
        X_windowed_scaled = scaler.fit_transform(X_windowed)
        
        rf_windowed = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv_scores_windowed = cross_val_score(rf_windowed, X_windowed_scaled, y, cv=3, scoring='accuracy')
        
        results['windowed'] = {
            'cv_mean': cv_scores_windowed.mean(),
            'cv_std': cv_scores_windowed.std(),
            'num_features': len(top_windowed_features)
        }
        
        print(f"   Features: {len(top_windowed_features)}")
        print(f"   CV Accuracy: {cv_scores_windowed.mean():.3f} ± {cv_scores_windowed.std():.3f}")
        
        # 3. Integrated features
        print("\n3️⃣ Integrated Features (Traditional + Windowed)")
        all_features = top_traditional_features + top_windowed_features
        X_integrated = integrated_df[all_features].copy()
        X_integrated = X_integrated.fillna(X_integrated.median())
        X_integrated = X_integrated.replace([np.inf, -np.inf], np.nan).fillna(X_integrated.median())
        
        scaler = StandardScaler()
        X_integrated_scaled = scaler.fit_transform(X_integrated)
        
        rf_integrated = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv_scores_integrated = cross_val_score(rf_integrated, X_integrated_scaled, y, cv=3, scoring='accuracy')
        
        results['integrated'] = {
            'cv_mean': cv_scores_integrated.mean(),
            'cv_std': cv_scores_integrated.std(),
            'num_features': len(all_features)
        }
        
        print(f"   Features: {len(all_features)}")
        print(f"   CV Accuracy: {cv_scores_integrated.mean():.3f} ± {cv_scores_integrated.std():.3f}")
        
        # Summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 50)
        print("Method              | Features | CV Accuracy | Std")
        print("-" * 50)
        
        for method, stats in results.items():
            print(f"{method:18s} | {stats['num_features']:8d} | {stats['cv_mean']:10.3f} | {stats['cv_std']:.3f}")
        
        # Determine best method
        best_method = max(results.items(), key=lambda x: x[1]['cv_mean'])
        print(f"\n🏆 Best Method: {best_method[0]} (CV: {best_method[1]['cv_mean']:.3f})")
        
        return results
    
    def create_final_feature_set(self, integrated_df, output_file="integrated_pd_features.csv"):
        """Create and save final integrated feature set"""
        print(f"\n💾 CREATING FINAL FEATURE SET")
        print("=" * 35)
        
        # Save integrated dataset
        integrated_df.to_csv(output_file, index=False)
        
        print(f"✅ Integrated features saved to: {output_file}")
        print(f"   Total samples: {len(integrated_df)}")
        print(f"   Total features: {len(integrated_df.columns) - 2}")  # Exclude filename and group
        print(f"   Class distribution: {dict(integrated_df['group'].value_counts())}")
        
        # Feature summary
        feature_cols = [col for col in integrated_df.columns if col not in ['filename', 'group']]
        traditional_features = [col for col in feature_cols if not col.startswith('w')]
        windowed_features = [col for col in feature_cols if col.startswith('w')]
        
        print(f"\n📋 Feature Breakdown:")
        print(f"   Traditional features: {len(traditional_features)}")
        print(f"   Windowed features: {len(windowed_features)}")
        
        # Windowed feature breakdown by window size
        if windowed_features:
            window_breakdown = {}
            for feat in windowed_features:
                for ws in [5, 10, 20]:
                    if f'w{ws}ms_' in feat:
                        if ws not in window_breakdown:
                            window_breakdown[ws] = 0
                        window_breakdown[ws] += 1
                        break
            
            print(f"   Windowed feature breakdown:")
            for ws, count in sorted(window_breakdown.items()):
                print(f"     {ws}ms windows: {count} features")
        
        return output_file

def main():
    """Main integrated feature selection pipeline"""
    
    print("🚀 INTEGRATED WINDOWED + TRADITIONAL FEATURE SELECTION")
    print("=" * 60)
    
    # File paths
    traditional_csv = "comprehensive_features/pd_features_comprehensive.csv"
    windowed_csv = "comprehensive_features/windowed_pd_features.csv"
    windowed_importance_csv = "comprehensive_features/windowed_feature_importance.csv"
    
    # Check files exist
    import os
    for filepath in [traditional_csv, windowed_csv, windowed_importance_csv]:
        if not os.path.exists(filepath):
            print(f"❌ Required file not found: {filepath}")
            return
    
    # Initialize selector
    selector = IntegratedFeatureSelector(
        traditional_csv, 
        windowed_csv, 
        windowed_importance_csv
    )
    
    # Select best features from each type
    top_windowed_features = selector.select_best_windowed_features(top_k=40)
    
    # Align datasets for traditional feature selection
    traditional_aligned, windowed_aligned = selector.align_datasets()
    if traditional_aligned is None:
        print("❌ Cannot align datasets")
        return
    
    top_traditional_features = selector.select_best_traditional_features(traditional_aligned, top_k=20)
    
    # Create integrated dataset
    integrated_df = selector.create_integrated_dataset(top_traditional_features, top_windowed_features)
    
    if integrated_df is None:
        print("❌ Failed to create integrated dataset")
        return
    
    # Compare performance
    results = selector.compare_performance(integrated_df, top_traditional_features, top_windowed_features)
    
    # Save final feature set
    output_file = "comprehensive_features/integrated_pd_features.csv"
    selector.create_final_feature_set(integrated_df, output_file)
    
    print(f"\n🎉 Integrated feature selection completed!")
    print(f"📁 Final feature set: {output_file}")

if __name__ == "__main__":
    main()
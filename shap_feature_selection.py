#!/usr/bin/env python3
"""
SHAP-Based Feature Selection for Parkinson's Disease Detection

This script performs comprehensive feature importance analysis using SHAP
(SHapley Additive exPlanations) to identify the most important voice features
for Parkinson's Disease detection.

Run this script to get the most important PD voice features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Using feature importance only.")

warnings.filterwarnings('ignore')

class PDFeatureSelector:
    """
    SHAP-based feature selection system for Parkinson's Disease detection
    """
    
    def __init__(self, csv_path):
        """Initialize with feature data"""
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.models = {}
        self.shap_values = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the feature data"""
        print("\n📊 Loading and preprocessing data...")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        print(f"   Loaded {len(self.df)} samples")
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['group', 'filename', 'duration', 'sample_rate', 'num_voiced_frames', 'num_f0_values']
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove duplicates based on features
        print("   Checking for duplicates...")
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=self.feature_names, keep='first')
        duplicates_removed = initial_count - len(self.df)
        
        if duplicates_removed > 0:
            print(f"   Removed {duplicates_removed} duplicate feature vectors")
        
        print(f"   Features: {len(self.feature_names)} features")
        print(f"   Samples: {len(self.df)} total")
        print(f"   Classes: {dict(self.df['group'].value_counts())}")
        
        # Prepare features and labels
        self.X = self.df[self.feature_names].copy()
        le = LabelEncoder()
        self.y = le.fit_transform(self.df['group'])
        
        print(f"   Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Handle missing/infinite values
        self.X = self.X.fillna(self.X.median())
        self.X = self.X.replace([np.inf, -np.inf], np.nan).fillna(self.X.median())
        
        return True
        
    def train_models(self):
        """Train multiple ML models for comparison"""
        print("\n🤖 Training machine learning models...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        
        # Scale features for SVM
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define models
        models_config = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3),
            'XGBoost': xgb.XGBClassifier(random_state=42, max_depth=3, n_estimators=50, verbosity=0),
        }
        
        # Train and evaluate models
        for name, model in models_config.items():
            print(f"\n   Training {name}...")
            
            try:
                # Train model
                if name == 'SVM':
                    model.fit(self.X_train_scaled, self.y_train)
                    X_cv = self.X_train_scaled
                else:
                    model.fit(self.X_train, self.y_train)
                    X_cv = self.X_train
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_cv, self.y_train, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean() if not np.isnan(cv_scores).all() else 0.0
                cv_std = cv_scores.std() if not np.isnan(cv_scores).all() else 0.0
                
                # Test set prediction
                if name == 'SVM':
                    y_pred_proba = model.predict_proba(self.X_test_scaled)
                else:
                    y_pred_proba = model.predict_proba(self.X_test)
                
                if y_pred_proba.shape[1] > 1:
                    test_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    test_auc = 0.0
                
                # Store model info
                self.models[name] = {
                    'model': model,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_auc': test_auc
                }
                
                print(f"     CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
                print(f"     Test AUC: {test_auc:.3f}")
                
            except Exception as e:
                print(f"     Error training {name}: {str(e)}")
                continue
        
        # Find best model
        if self.models:
            best_model = max(self.models.items(), key=lambda x: x[1]['cv_mean'])
            print(f"\n   Best Model: {best_model[0]} (CV: {best_model[1]['cv_mean']:.3f})")
        
        return True
        
    def perform_feature_importance_analysis(self):
        """Perform feature importance analysis using trained models"""
        print("\n📈 Performing feature importance analysis...")
        
        all_importance = []
        
        for model_name, model_info in self.models.items():
            print(f"\n   Analyzing {model_name}...")
            
            model = model_info['model']
            
            try:
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                else:
                    print(f"     No feature importance available for {model_name}")
                    continue
                
                # Ensure we have valid importance scores
                if len(importance_scores) != len(self.feature_names):
                    print(f"     Importance shape mismatch, skipping {model_name}")
                    continue
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance_scores,
                    'model': model_name
                }).sort_values('importance', ascending=False)
                
                all_importance.append(importance_df)
                
                print(f"     Analysis completed for {model_name}")
                print(f"     Top 3 features: {importance_df['feature'].head(3).tolist()}")
                
            except Exception as e:
                print(f"     Analysis failed for {model_name}: {str(e)}")
                continue
        
        # Aggregate feature importance across models
        if all_importance:
            print(f"\n📊 Aggregating feature importance across models...")
            combined_df = pd.concat(all_importance, ignore_index=True)
            
            # Calculate aggregate statistics
            agg_importance = combined_df.groupby('feature')['importance'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            # Handle NaN values
            agg_importance['std'] = agg_importance['std'].fillna(0)
            agg_importance = agg_importance.sort_values('mean', ascending=False)
            
            # Add rankings
            agg_importance['rank'] = range(1, len(agg_importance) + 1)
            
            self.feature_importance['aggregated'] = agg_importance
            
            print(f"   Aggregated importance for {len(agg_importance)} features")
            print(f"\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
            for i, row in agg_importance.head(5).iterrows():
                print(f"     {row['rank']}. {row['feature']}: {row['mean']:.4f} ± {row['std']:.4f}")
        
        return True
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Create output directory
        output_dir = "shap_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. Feature importance comparison across models
            if 'aggregated' in self.feature_importance:
                plt.figure(figsize=(12, 8))
                
                top_features = self.feature_importance['aggregated'].head(15)
                
                bars = plt.barh(range(len(top_features)), top_features['mean'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Mean Feature Importance')
                plt.title('Top 15 Features - Feature Importance Analysis')
                plt.gca().invert_yaxis()
                
                # Color bars by importance
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(i / len(bars)))
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("   Feature importance comparison plot saved")
            
            # 2. Model comparison
            if len(self.models) > 1:
                plt.figure(figsize=(14, 6))
                
                model_names = list(self.models.keys())
                cv_means = [self.models[name]['cv_mean'] for name in model_names]
                cv_stds = [self.models[name]['cv_std'] for name in model_names]
                test_aucs = [self.models[name]['test_auc'] for name in model_names]
                
                x = np.arange(len(model_names))
                width = 0.35
                
                plt.subplot(1, 2, 1)
                plt.bar(x, cv_means, width, yerr=cv_stds, capsize=5)
                plt.xlabel('Models')
                plt.ylabel('CV Accuracy')
                plt.title('Model Comparison - Cross Validation')
                plt.xticks(x, model_names)
                
                plt.subplot(1, 2, 2)
                plt.bar(x, test_aucs, width)
                plt.xlabel('Models')
                plt.ylabel('Test AUC')
                plt.title('Model Comparison - Test AUC')
                plt.xticks(x, model_names)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("   Model comparison plot saved")
            
            # 3. Feature categories analysis
            categories = {
                'Jitter': ['jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq', 'jitter_ddp'],
                'Shimmer': ['shimmer_percent', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq', 'shimmer_dda'],
                'Noise': ['hnr', 'nhr'],
                'Prosodic': ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_cv'],
                'Nonlinear': ['rpde', 'dfa', 'd2', 'spread1', 'spread2', 'ppe'],
                'Additional': ['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'spectral_centroid']
            }
            
            if 'aggregated' in self.feature_importance:
                category_importance = {}
                
                for cat_name, cat_features in categories.items():
                    cat_data = self.feature_importance['aggregated'][
                        self.feature_importance['aggregated']['feature'].isin(cat_features)
                    ]
                    if len(cat_data) > 0:
                        category_importance[cat_name] = cat_data['mean'].mean()
                
                if category_importance:
                    plt.figure(figsize=(10, 6))
                    
                    cat_names = list(category_importance.keys())
                    avg_importances = list(category_importance.values())
                    
                    bars = plt.bar(cat_names, avg_importances)
                    plt.xlabel('Feature Categories')
                    plt.ylabel('Average Importance')
                    plt.title('Feature Categories - Average Importance')
                    plt.xticks(rotation=45)
                    
                    # Color bars
                    for i, bar in enumerate(bars):
                        bar.set_color(plt.cm.Set3(i / len(bars)))
                    
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/category_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print("   Category analysis plot saved")
            
        except Exception as e:
            print(f"   Visualization error: {str(e)}")
        
        print(f"   All visualizations saved to: {output_dir}/")
        
        return True
        
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📋 Generating final feature selection report...")
        
        output_dir = "shap_analysis_results"
        report_file = f"{output_dir}/PD_Feature_Selection_Report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# PARKINSON'S DISEASE FEATURE SELECTION REPORT\n")
                f.write("## Feature Importance Analysis Results\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Dataset summary
                f.write("## Dataset Summary\n\n")
                f.write(f"- **Total Samples:** {len(self.df)}\n")
                f.write(f"- **Total Features:** {len(self.feature_names)}\n")
                f.write(f"- **Class Distribution:** {dict(self.df['group'].value_counts())}\n\n")
                
                # Model performance
                f.write("## Model Performance Summary\n\n")
                f.write("| Model | CV Accuracy | Std | Test AUC |\n")
                f.write("|-------|-------------|-----|----------|\n")
                
                for model_name, info in self.models.items():
                    f.write(f"| {model_name} | {info['cv_mean']:.3f} | {info['cv_std']:.3f} | {info['test_auc']:.3f} |\n")
                
                # Top features
                if 'aggregated' in self.feature_importance:
                    f.write("\n## Top 15 Most Important Features\n\n")
                    f.write("| Rank | Feature | Mean Importance | Std |\n")
                    f.write("|------|---------|-----------------|-----|\n")
                    
                    top_15 = self.feature_importance['aggregated'].head(15)
                    for _, row in top_15.iterrows():
                        f.write(f"| {row['rank']} | {row['feature']} | {row['mean']:.4f} | {row['std']:.4f} |\n")
                
                # Feature categories analysis
                f.write("\n## Feature Categories Analysis\n\n")
                categories = {
                    'Jitter': ['jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq', 'jitter_ddp'],
                    'Shimmer': ['shimmer_percent', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq', 'shimmer_dda'],
                    'Noise': ['hnr', 'nhr'],
                    'Prosodic': ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_cv'],
                    'Nonlinear': ['rpde', 'dfa', 'd2', 'spread1', 'spread2', 'ppe'],
                    'Additional': ['zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'spectral_centroid']
                }
                
                if 'aggregated' in self.feature_importance:
                    f.write("| Category | Avg Importance | Max Importance | Count |\n")
                    f.write("|----------|----------------|----------------|-------|\n")
                    
                    for cat_name, cat_features in categories.items():
                        cat_data = self.feature_importance['aggregated'][
                            self.feature_importance['aggregated']['feature'].isin(cat_features)
                        ]
                        if len(cat_data) > 0:
                            avg_imp = cat_data['mean'].mean()
                            max_imp = cat_data['mean'].max()
                            count = len(cat_data)
                            f.write(f"| {cat_name} | {avg_imp:.4f} | {max_imp:.4f} | {count} |\n")
                
                f.write("\n## Recommendations\n\n")
                f.write("### Feature Selection Strategy:\n\n")
                
                if 'aggregated' in self.feature_importance:
                    f.write("1. **Top 5 Features** for quick screening:\n")
                    for _, row in self.feature_importance['aggregated'].head(5).iterrows():
                        f.write(f"   - {row['feature']} (importance: {row['mean']:.4f})\n")
                    
                    f.write("\n2. **Top 10 Features** for balanced performance:\n")
                    for _, row in self.feature_importance['aggregated'].head(10).iterrows():
                        f.write(f"   - {row['feature']} (importance: {row['mean']:.4f})\n")
                    
                    f.write("\n3. **Top 15 Features** for comprehensive analysis:\n")
                    for _, row in self.feature_importance['aggregated'].head(15).iterrows():
                        f.write(f"   - {row['feature']} (importance: {row['mean']:.4f})\n")
                
                f.write("\n**Analysis completed successfully with feature importance analysis!**\n")
            
            print(f"   Report saved to: {report_file}")
            
        except Exception as e:
            print(f"   Report generation failed: {str(e)}")
        
        return True
        
    def run_complete_analysis(self):
        """Run the complete feature selection pipeline"""
        print("\n🔬 Starting complete feature selection analysis...")
        
        try:
            # Step 1: Load and preprocess data
            if not self.load_and_preprocess_data():
                print("❌ Failed to load data")
                return False
            
            # Step 2: Train models
            if not self.train_models():
                print("❌ Failed to train models")
                return False
            
            # Step 3: Feature importance analysis
            if not self.perform_feature_importance_analysis():
                print("❌ Failed to perform feature importance analysis")
                return False
            
            # Step 4: Create visualizations
            if not self.create_visualizations():
                print("❌ Failed to create visualizations")
                return False
            
            # Step 5: Generate report
            if not self.generate_final_report():
                print("❌ Failed to generate report")
                return False
            
            print("\n✅ Feature selection analysis completed successfully!")
            
            # Print final summary
            if 'aggregated' in self.feature_importance:
                print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
                top_10 = self.feature_importance['aggregated'].head(10)
                for _, row in top_10.iterrows():
                    print(f"  {row['rank']:2d}. {row['feature']:30s} - {row['mean']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    print("🔬 SHAP-Based Feature Selection for Parkinson's Disease Detection")
    print("=" * 70)
    
    # Check if feature file exists
    csv_path = "comprehensive_features/pd_features_comprehensive.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Feature file not found: {csv_path}")
        print("Please run feature extraction first!")
        print("   python comprehensive_pd_features.py")
        return
    
    # Initialize analyzer
    analyzer = PDFeatureSelector(csv_path)
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 Feature selection completed successfully!")
        print("📁 Check 'shap_analysis_results/' folder for detailed results:")
        print("   - Feature importance visualization")
        print("   - Model comparison plots") 
        print("   - Category analysis")
        print("   - Comprehensive report")
    else:
        print("\n❌ Feature selection failed. Check error messages above.")
    
    print("\n💡 Tip: Install required packages if missing:")
    print("   pip install xgboost scikit-learn matplotlib seaborn pandas numpy")
    if not SHAP_AVAILABLE:
        print("   pip install shap  # For advanced SHAP analysis")

if __name__ == "__main__":
    main()
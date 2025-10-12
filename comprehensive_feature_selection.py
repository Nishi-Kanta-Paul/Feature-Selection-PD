#!/usr/bin/env python3
"""
COMPREHENSIVE FEATURE SELECTION FOR ALL FEATURE SETS
====================================================
Performs feature selection on:
1. Windowed features (5ms, 10ms, 20ms)
2. Non-windowed comprehensive features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveFeatureSelector:
    """Feature selection for multiple feature sets"""

    def __init__(self, csv_file, feature_set_name):
        """
        Initialize feature selector

        Args:
            csv_file: Path to feature CSV file
            feature_set_name: Name of the feature set (e.g., '5ms', '10ms', 'comprehensive')
        """
        self.df = pd.read_csv(csv_file)
        self.feature_set_name = feature_set_name
        self.csv_file = csv_file

        print(f"\n{'='*70}")
        print(f"🎯 FEATURE SELECTION: {feature_set_name}")
        print(f"{'='*70}")
        print(f"📁 File: {csv_file}")
        print(f"📊 Total samples: {len(self.df)}")
        print(f"🔢 Total columns: {len(self.df.columns)}")

    def prepare_data(self):
        """Prepare features and labels"""
        print("\n🔧 PREPARING DATA")
        print("-" * 50)

        # Identify metadata columns to exclude
        metadata_cols = ['filename', 'duration', 'original_sample_rate', 'working_sample_rate',
                         'window_size_ms', 'hop_size_ms', 'total_windows', 'voiced_windows',
                         'num_f0_values', 'group']

        # Get feature columns
        feature_cols = [
            col for col in self.df.columns if col not in metadata_cols]

        print(f"✅ Feature columns: {len(feature_cols)}")

        # Prepare X and y
        X = self.df[feature_cols].copy()

        # Handle group labels
        if 'group' in self.df.columns:
            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(self.df['group'])

            # Show label distribution
            unique, counts = np.unique(y, return_counts=True)
            print(f"📈 Label distribution:")
            for label, count in zip(le.classes_, counts):
                print(f"   {label}: {count} samples")
        else:
            print("❌ No 'group' column found!")
            return None, None, None

        # Handle missing values
        if X.isnull().any().any():
            print(f"⚠️  Missing values found: {X.isnull().sum().sum()}")
            X = X.fillna(0)
            print("   Filled with 0")

        # Remove constant features
        constant_features = [col for col in X.columns if X[col].std() == 0]
        if constant_features:
            print(f"🗑️  Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)

        print(f"✅ Final feature count: {X.shape[1]}")
        print(f"✅ Final sample count: {X.shape[0]}")

        return X, y, feature_cols

    def feature_importance_rf(self, X, y):
        """Random Forest feature importance"""
        print("\n🌲 RANDOM FOREST FEATURE IMPORTANCE")
        print("-" * 50)

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Get feature importance
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"✅ Top 10 most important features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']:<30}: {row['importance']:.4f}")

        return feature_importance_df

    def feature_importance_gb(self, X, y):
        """Gradient Boosting feature importance"""
        print("\n📈 GRADIENT BOOSTING FEATURE IMPORTANCE")
        print("-" * 50)

        # Train Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X, y)

        # Get feature importance
        importances = gb.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"✅ Top 10 most important features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']:<30}: {row['importance']:.4f}")

        return feature_importance_df

    def univariate_selection(self, X, y, k=20):
        """Univariate feature selection using F-score"""
        print(f"\n📊 UNIVARIATE FEATURE SELECTION (Top {k})")
        print("-" * 50)

        # Select K best features
        selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)

        # Get scores
        scores = selector.scores_
        feature_scores_df = pd.DataFrame({
            'feature': X.columns,
            'f_score': scores
        }).sort_values('f_score', ascending=False)

        print(f"✅ Top 10 features by F-score:")
        for idx, row in feature_scores_df.head(10).iterrows():
            print(f"   {row['feature']:<30}: {row['f_score']:.4f}")

        return feature_scores_df

    def mutual_information_selection(self, X, y):
        """Mutual information feature selection"""
        print("\n🔗 MUTUAL INFORMATION FEATURE SELECTION")
        print("-" * 50)

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        print(f"✅ Top 10 features by MI score:")
        for idx, row in mi_df.head(10).iterrows():
            print(f"   {row['feature']:<30}: {row['mi_score']:.4f}")

        return mi_df

    def evaluate_feature_subsets(self, X, y, feature_importance_df):
        """Evaluate different feature subset sizes"""
        print("\n🎯 EVALUATING FEATURE SUBSET SIZES")
        print("-" * 50)

        # Determine appropriate CV folds based on sample size
        n_samples = len(y)

        # Get minimum class count
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()

        # CV folds must be <= minimum class count
        cv_folds = min(5, min_class_count)

        if cv_folds < 2:
            print(
                f"⚠️  Insufficient samples for cross-validation (only {n_samples} samples)")
            print(f"   Skipping subset evaluation, using all features")
            return pd.DataFrame([{'k': len(feature_importance_df), 'mean_accuracy': 0.0, 'std_accuracy': 0.0}]), len(feature_importance_df)

        print(
            f"ℹ️  Using {cv_folds}-fold cross-validation ({n_samples} samples, min class count: {min_class_count})")

        # Test different subset sizes
        subset_sizes = [5, 10, 15, 20, 30, 40, 50]
        results = []

        for k in subset_sizes:
            if k > len(feature_importance_df):
                continue

            # Select top k features
            top_features = feature_importance_df.head(k)['feature'].tolist()
            X_subset = X[top_features]

            # Evaluate with cross-validation
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
            cv_scores = cross_val_score(
                rf, X_subset, y, cv=cv_folds, scoring='accuracy')

            results.append({
                'k': k,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std()
            })

            print(
                f"   K={k:2d} features: Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if len(results) == 0:
            # No valid subset sizes, use all features
            print(f"⚠️  No valid subset sizes found, using all features")
            return pd.DataFrame([{'k': len(feature_importance_df), 'mean_accuracy': 0.0, 'std_accuracy': 0.0}]), len(feature_importance_df)

        results_df = pd.DataFrame(results)

        # Find best k
        best_k = results_df.loc[results_df['mean_accuracy'].idxmax(), 'k']
        best_acc = results_df.loc[results_df['mean_accuracy'].idxmax(
        ), 'mean_accuracy']
        print(
            f"\n✅ Best subset size: K={int(best_k)} (Accuracy: {best_acc:.4f})")

        return results_df, int(best_k)

    def aggregate_feature_rankings(self, rf_importance, gb_importance, f_scores, mi_scores):
        """Aggregate rankings from different methods"""
        print("\n🔄 AGGREGATING FEATURE RANKINGS")
        print("-" * 50)

        # Normalize scores to 0-1 range
        def normalize(df, score_col):
            df = df.copy()
            df['normalized'] = (df[score_col] - df[score_col].min()) / \
                (df[score_col].max() - df[score_col].min())
            return df

        rf_norm = normalize(rf_importance, 'importance')
        gb_norm = normalize(gb_importance, 'importance')
        f_norm = normalize(f_scores, 'f_score')
        mi_norm = normalize(mi_scores, 'mi_score')

        # Merge all scores
        all_features = set(rf_norm['feature']) | set(gb_norm['feature']) | set(
            f_norm['feature']) | set(mi_norm['feature'])

        aggregated = []
        for feature in all_features:
            rf_score = rf_norm[rf_norm['feature'] ==
                               feature]['normalized'].values[0] if feature in rf_norm['feature'].values else 0
            gb_score = gb_norm[gb_norm['feature'] ==
                               feature]['normalized'].values[0] if feature in gb_norm['feature'].values else 0
            f_score = f_norm[f_norm['feature'] ==
                             feature]['normalized'].values[0] if feature in f_norm['feature'].values else 0
            mi_score = mi_norm[mi_norm['feature'] ==
                               feature]['normalized'].values[0] if feature in mi_norm['feature'].values else 0

            # Weighted average (you can adjust weights)
            aggregate_score = (0.3 * rf_score + 0.3 *
                               gb_score + 0.2 * f_score + 0.2 * mi_score)

            aggregated.append({
                'feature': feature,
                'rf_score': rf_score,
                'gb_score': gb_score,
                'f_score': f_score,
                'mi_score': mi_score,
                'aggregate_score': aggregate_score
            })

        aggregated_df = pd.DataFrame(aggregated).sort_values(
            'aggregate_score', ascending=False)

        print(f"✅ Top 15 features by aggregated ranking:")
        for idx, row in aggregated_df.head(15).iterrows():
            print(f"   {row['feature']:<30}: {row['aggregate_score']:.4f}")

        return aggregated_df

    def save_results(self, rf_importance, gb_importance, f_scores, mi_scores, aggregated, results_df, best_k):
        """Save feature selection results"""
        print("\n💾 SAVING RESULTS")
        print("-" * 50)

        # Create output directory
        output_dir = f"feature_selection_results/{self.feature_set_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save individual rankings
        rf_importance.to_csv(
            f"{output_dir}/rf_feature_importance.csv", index=False)
        print(f"✅ Saved: {output_dir}/rf_feature_importance.csv")

        gb_importance.to_csv(
            f"{output_dir}/gb_feature_importance.csv", index=False)
        print(f"✅ Saved: {output_dir}/gb_feature_importance.csv")

        f_scores.to_csv(f"{output_dir}/f_score_ranking.csv", index=False)
        print(f"✅ Saved: {output_dir}/f_score_ranking.csv")

        mi_scores.to_csv(f"{output_dir}/mi_score_ranking.csv", index=False)
        print(f"✅ Saved: {output_dir}/mi_score_ranking.csv")

        # Save aggregated rankings
        aggregated.to_csv(f"{output_dir}/aggregated_rankings.csv", index=False)
        print(f"✅ Saved: {output_dir}/aggregated_rankings.csv")

        # Save subset evaluation results
        results_df.to_csv(f"{output_dir}/subset_evaluation.csv", index=False)
        print(f"✅ Saved: {output_dir}/subset_evaluation.csv")

        # Save top K features
        top_features = aggregated.head(best_k)['feature'].tolist()
        top_features_df = pd.DataFrame({'feature': top_features})
        top_features_df.to_csv(
            f"{output_dir}/top_{best_k}_features.csv", index=False)
        print(f"✅ Saved: {output_dir}/top_{best_k}_features.csv")

        return output_dir

    def run_complete_analysis(self):
        """Run complete feature selection analysis"""
        print(f"\n{'='*70}")
        print(f"🚀 STARTING COMPLETE FEATURE SELECTION ANALYSIS")
        print(f"{'='*70}")

        # Step 1: Prepare data
        X, y, feature_cols = self.prepare_data()
        if X is None:
            return None

        # Step 2: Random Forest importance
        rf_importance = self.feature_importance_rf(X, y)

        # Step 3: Gradient Boosting importance
        gb_importance = self.feature_importance_gb(X, y)

        # Step 4: Univariate selection
        f_scores = self.univariate_selection(X, y, k=30)

        # Step 5: Mutual information
        mi_scores = self.mutual_information_selection(X, y)

        # Step 6: Aggregate rankings
        aggregated = self.aggregate_feature_rankings(
            rf_importance, gb_importance, f_scores, mi_scores)

        # Step 7: Evaluate subsets
        results_df, best_k = self.evaluate_feature_subsets(X, y, aggregated)

        # Step 8: Save results
        output_dir = self.save_results(rf_importance, gb_importance, f_scores, mi_scores,
                                       aggregated, results_df, best_k)

        print(f"\n{'='*70}")
        print(f"✅ FEATURE SELECTION COMPLETE: {self.feature_set_name}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🎯 Best K features: {best_k}")
        print(f"{'='*70}\n")

        return {
            'rf_importance': rf_importance,
            'gb_importance': gb_importance,
            'f_scores': f_scores,
            'mi_scores': mi_scores,
            'aggregated': aggregated,
            'results_df': results_df,
            'best_k': best_k,
            'output_dir': output_dir
        }


def main():
    """Main function to run feature selection on all feature sets"""

    print("\n" + "="*70)
    print("🎯 COMPREHENSIVE FEATURE SELECTION FOR ALL DATASETS")
    print("="*70)

    # Define all feature sets
    feature_sets = [
        {
            'name': 'Windowed_5ms',
            'file': 'comprehensive_features/windowed_pd_features_5ms.csv'
        },
        {
            'name': 'Windowed_10ms',
            'file': 'comprehensive_features/windowed_pd_features_10ms.csv'
        },
        {
            'name': 'Windowed_20ms',
            'file': 'comprehensive_features/windowed_pd_features_20ms.csv'
        },
        {
            'name': 'Comprehensive_Final',
            'file': 'comprehensive_features/comprehensive_pd_features_final.csv'
        }
    ]

    all_results = {}

    # Process each feature set
    for feature_set in feature_sets:
        name = feature_set['name']
        file = feature_set['file']

        if not os.path.exists(file):
            print(f"\n⚠️  Skipping {name}: File not found - {file}")
            continue

        # Run feature selection
        selector = ComprehensiveFeatureSelector(file, name)
        results = selector.run_complete_analysis()

        if results:
            all_results[name] = results

    # Summary comparison
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON ACROSS ALL FEATURE SETS")
    print("="*70)

    summary_data = []
    for name, results in all_results.items():
        summary_data.append({
            'Feature Set': name,
            'Best K': results['best_k'],
            'Best Accuracy': results['results_df'].loc[results['results_df']['mean_accuracy'].idxmax(), 'mean_accuracy'],
            'Top Feature': results['aggregated'].iloc[0]['feature'],
            'Top Score': results['aggregated'].iloc[0]['aggregate_score']
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    os.makedirs("feature_selection_results", exist_ok=True)
    summary_df.to_csv(
        "feature_selection_results/summary_comparison.csv", index=False)
    print(f"\n✅ Summary saved to: feature_selection_results/summary_comparison.csv")

    print("\n" + "="*70)
    print("🎉 ALL FEATURE SELECTION ANALYSES COMPLETE!")
    print("="*70)

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()

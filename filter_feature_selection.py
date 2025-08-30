import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, VarianceThreshold,
    f_classif, chi2, mutual_info_classif
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os

class FilterFeatureSelector:
    """
    Comprehensive Filter-based Feature Selection for Parkinson's Disease Detection
    
    Implements multiple filter methods:
    1. Statistical Tests (t-test, ANOVA F-test, Chi-square)
    2. Correlation-based Selection
    3. Variance-based Selection  
    4. Mutual Information
    5. Univariate Feature Selection
    6. Combined Filter Ranking
    """
    
    def __init__(self, features_df=None):
        self.features_df = features_df
        self.X = None
        self.y = None
        self.feature_names = None
        self.selection_results = {}
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path="extracted_features.csv"):
        """Load extracted features data"""
        try:
            self.features_df = pd.read_csv(csv_path)
            print(f"✅ Loaded features: {self.features_df.shape}")
            
            # Prepare feature matrix and target
            numerical_cols = self.features_df.select_dtypes(include=[np.number]).columns
            self.feature_names = [col for col in numerical_cols if col not in ['cohort_numeric']]
            
            self.X = self.features_df[self.feature_names].values
            self.y = self.features_df['cohort_numeric'].values
            
            print(f"✅ Feature matrix: {self.X.shape}")
            print(f"✅ Target distribution: PD={np.sum(self.y)}, HC={len(self.y)-np.sum(self.y)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def variance_threshold_selection(self, threshold=0.0):
        """Remove features with low variance"""
        print(f"\n{'='*60}")
        print("🔍 VARIANCE THRESHOLD SELECTION")
        print(f"{'='*60}")
        
        # Calculate variances
        variances = np.var(self.X, axis=0)
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(self.X)
        selected_features = np.array(self.feature_names)[selector.get_support()]
        
        # Store results
        self.selection_results['variance_threshold'] = {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'threshold': threshold,
            'variances': variances,
            'selector': selector
        }
        
        print(f"📊 Original features: {len(self.feature_names)}")
        print(f"📊 Selected features: {len(selected_features)}")
        print(f"📊 Removed features: {len(self.feature_names) - len(selected_features)}")
        print(f"📊 Variance threshold: {threshold}")
        
        # Show lowest variance features
        variance_df = pd.DataFrame({
            'feature': self.feature_names,
            'variance': variances
        }).sort_values('variance')
        
        print(f"\n🔻 10 Lowest Variance Features:")
        for i, row in variance_df.head(10).iterrows():
            print(f"   {row['feature']:<30} | Variance: {row['variance']:.6f}")
        
        return selected_features, X_selected
    
    def correlation_based_selection(self, threshold=0.95):
        """Remove highly correlated features"""
        print(f"\n{'='*60}")
        print("🔍 CORRELATION-BASED SELECTION")
        print(f"{'='*60}")
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(self.X, columns=self.feature_names).corr().abs()
        
        # Find highly correlated pairs
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = []
        
        # Find features to remove
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))
                    
                    # Remove feature with lower variance
                    var1 = np.var(self.X[:, i])
                    var2 = np.var(self.X[:, j])
                    if var1 < var2:
                        to_remove.add(feature1)
                    else:
                        to_remove.add(feature2)
        
        # Select features
        selected_features = [f for f in self.feature_names if f not in to_remove]
        selected_indices = [i for i, f in enumerate(self.feature_names) if f not in to_remove]
        X_selected = self.X[:, selected_indices]
        
        # Store results
        self.selection_results['correlation_threshold'] = {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'threshold': threshold,
            'removed_features': list(to_remove),
            'high_corr_pairs': high_corr_pairs,
            'correlation_matrix': corr_matrix
        }
        
        print(f"📊 Correlation threshold: {threshold}")
        print(f"📊 High correlation pairs found: {len(high_corr_pairs)}")
        print(f"📊 Features removed: {len(to_remove)}")
        print(f"📊 Features selected: {len(selected_features)}")
        
        if high_corr_pairs:
            print(f"\n🔗 Top 5 High Correlation Pairs:")
            sorted_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
            for i, (f1, f2, corr) in enumerate(sorted_pairs[:5]):
                print(f"   {i+1}. {f1[:25]:<25} ↔ {f2[:25]:<25} | r={corr:.4f}")
        
        return selected_features, X_selected
    
    def statistical_tests_selection(self, method='f_classif', k=50):
        """Univariate statistical tests"""
        print(f"\n{'='*60}")
        print(f"🔍 STATISTICAL TESTS SELECTION ({method.upper()})")
        print(f"{'='*60}")
        
        if method == 'f_classif':
            # ANOVA F-test
            f_scores, p_values = f_classif(self.X, self.y)
            scores = f_scores
            test_name = "ANOVA F-test"
            
        elif method == 'mutual_info':
            # Mutual Information
            scores = mutual_info_classif(self.X, self.y, random_state=42)
            p_values = np.ones(len(scores))  # MI doesn't provide p-values
            test_name = "Mutual Information"
            
        elif method == 't_test':
            # Independent t-test
            scores = []
            p_values = []
            for i in range(self.X.shape[1]):
                pd_values = self.X[self.y == 1, i]
                hc_values = self.X[self.y == 0, i]
                if len(pd_values) > 1 and len(hc_values) > 1:
                    t_stat, p_val = stats.ttest_ind(pd_values, hc_values)
                    scores.append(abs(t_stat))
                    p_values.append(p_val)
                else:
                    scores.append(0)
                    p_values.append(1)
            scores = np.array(scores)
            p_values = np.array(p_values)
            test_name = "Independent t-test"
        
        # Select top k features
        selector = SelectKBest(score_func=f_classif if method == 'f_classif' else None, k=k)
        if method == 'f_classif':
            X_selected = selector.fit_transform(self.X, self.y)
            selected_indices = selector.get_support()
        else:
            # Manual selection for other methods
            top_indices = np.argsort(scores)[-k:]
            selected_indices = np.zeros(len(scores), dtype=bool)
            selected_indices[top_indices] = True
            X_selected = self.X[:, selected_indices]
        
        selected_features = np.array(self.feature_names)[selected_indices]
        
        # Store results
        self.selection_results[f'statistical_{method}'] = {
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': scores,
            'p_values': p_values,
            'method': test_name,
            'k': k
        }
        
        print(f"📊 Method: {test_name}")
        print(f"📊 Features selected: {k}")
        print(f"📊 Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        if method != 'mutual_info':
            significant_features = np.sum(p_values < 0.05)
            print(f"📊 Significant features (p < 0.05): {significant_features}")
        
        # Show top features
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': scores,
            'p_value': p_values,
            'selected': selected_indices
        }).sort_values('score', ascending=False)
        
        print(f"\n🏆 Top 10 Features by {test_name}:")
        for i, row in feature_scores.head(10).iterrows():
            status = "✅" if row['selected'] else "❌"
            if method != 'mutual_info':
                print(f"   {status} {row['feature']:<30} | Score: {row['score']:.4f} | p: {row['p_value']:.4e}")
            else:
                print(f"   {status} {row['feature']:<30} | Score: {row['score']:.4f}")
        
        return selected_features, X_selected
    
    def combined_filter_ranking(self, methods=['f_classif', 't_test', 'mutual_info'], weights=None):
        """Combine multiple filter methods with ranking"""
        print(f"\n{'='*60}")
        print("🔍 COMBINED FILTER RANKING")
        print(f"{'='*60}")
        
        if weights is None:
            weights = [1.0] * len(methods)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Collect scores from different methods
        all_scores = {}
        for method in methods:
            if method == 'f_classif':
                scores, _ = f_classif(self.X, self.y)
            elif method == 'mutual_info':
                scores = mutual_info_classif(self.X, self.y, random_state=42)
            elif method == 't_test':
                scores = []
                for i in range(self.X.shape[1]):
                    pd_values = self.X[self.y == 1, i]
                    hc_values = self.X[self.y == 0, i]
                    if len(pd_values) > 1 and len(hc_values) > 1:
                        t_stat, _ = stats.ttest_ind(pd_values, hc_values)
                        scores.append(abs(t_stat))
                    else:
                        scores.append(0)
                scores = np.array(scores)
            
            # Normalize scores to [0, 1]
            scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            # Handle NaN values
            scores_norm = np.nan_to_num(scores_norm, nan=0.0, posinf=1.0, neginf=0.0)
            all_scores[method] = scores_norm
        
        # Calculate combined score
        combined_scores = np.zeros(len(self.feature_names))
        for i, method in enumerate(methods):
            # Handle NaN values
            method_scores = all_scores[method]
            method_scores = np.nan_to_num(method_scores, nan=0.0, posinf=1.0, neginf=0.0)
            combined_scores += weights[i] * method_scores
        
        # Rank features
        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'combined_score': combined_scores,
            'rank': len(self.feature_names) - np.argsort(np.argsort(combined_scores))
        })
        
        # Add individual method scores
        for method in methods:
            feature_ranking[f'{method}_score'] = all_scores[method]
        
        feature_ranking = feature_ranking.sort_values('combined_score', ascending=False)
        
        # Store results
        self.selection_results['combined_ranking'] = {
            'ranking': feature_ranking,
            'methods': methods,
            'weights': weights,
            'combined_scores': combined_scores
        }
        
        print(f"📊 Methods combined: {', '.join(methods)}")
        print(f"📊 Weights: {[f'{w:.2f}' for w in weights]}")
        print(f"📊 Score range: [{np.min(combined_scores):.4f}, {np.max(combined_scores):.4f}]")
        
        print(f"\n🏆 Top 15 Features by Combined Ranking:")
        for i, row in feature_ranking.head(15).iterrows():
            print(f"   {row['rank']:2d}. {row['feature']:<30} | Score: {row['combined_score']:.4f}")
        
        return feature_ranking
    
    def evaluate_feature_sets(self, output_dir="feature_selection_analysis"):
        """Evaluate different feature selection methods using cross-validation"""
        print(f"\n{'='*60}")
        print("🔍 FEATURE SET EVALUATION")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Evaluate different feature sets
        evaluation_results = {}
        
        # Original features
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, X_scaled, self.y, cv=5, scoring='accuracy')
            evaluation_results[f'Original_{clf_name}'] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_features': self.X.shape[1]
            }
        
        # Evaluate each selection method
        for method_name, results in self.selection_results.items():
            if 'selected_features' in results:
                selected_features = results['selected_features']
                feature_indices = [i for i, f in enumerate(self.feature_names) if f in selected_features]
                X_subset = X_scaled[:, feature_indices]
                
                for clf_name, clf in classifiers.items():
                    try:
                        scores = cross_val_score(clf, X_subset, self.y, cv=5, scoring='accuracy')
                        evaluation_results[f'{method_name}_{clf_name}'] = {
                            'mean_score': np.mean(scores),
                            'std_score': np.std(scores),
                            'n_features': len(selected_features)
                        }
                    except Exception as e:
                        print(f"⚠️  Error evaluating {method_name} with {clf_name}: {e}")
        
        # Store evaluation results
        self.selection_results['evaluation'] = evaluation_results
        
        # Print results
        print(f"\n📊 Cross-Validation Results (5-fold):")
        print(f"{'Method':<40} {'Classifier':<20} {'Features':<10} {'Accuracy':<12} {'Std':<8}")
        print("-" * 90)
        
        for method_clf, results in evaluation_results.items():
            method, clf = method_clf.rsplit('_', 1)
            print(f"{method:<40} {clf:<20} {results['n_features']:<10} "
                  f"{results['mean_score']:.4f}±{results['std_score']:.4f}")
        
        return evaluation_results
    
    def create_comprehensive_visualizations(self, output_dir="feature_selection_analysis"):
        """Create comprehensive visualizations for feature selection analysis"""
        print(f"\n{'='*60}")
        print("🎨 CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feature Selection Methods Comparison
        self._plot_selection_methods_comparison(output_dir)
        
        # 2. Statistical Scores Visualization
        self._plot_statistical_scores(output_dir)
        
        # 3. Correlation Analysis
        self._plot_correlation_analysis(output_dir)
        
        # 4. Feature Rankings Comparison
        self._plot_feature_rankings(output_dir)
        
        # 5. Evaluation Results
        self._plot_evaluation_results(output_dir)
        
        # 6. Feature Selection Pipeline
        self._plot_selection_pipeline(output_dir)
        
        print(f"✅ Visualizations saved to: {os.path.abspath(output_dir)}")
    
    def _plot_selection_methods_comparison(self, output_dir):
        """Plot comparison of different selection methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Method 1: Number of features selected
        methods = []
        n_features = []
        
        for method, results in self.selection_results.items():
            if 'n_selected' in results:
                methods.append(method.replace('_', ' ').title())
                n_features.append(results['n_selected'])
        
        if methods:
            axes[0, 0].bar(methods, n_features, color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Number of Features Selected by Method', fontweight='bold')
            axes[0, 0].set_ylabel('Number of Features')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(n_features):
                axes[0, 0].text(i, v + 1, str(v), ha='center', va='bottom')
        
        # Method 2: Variance distribution
        if 'variance_threshold' in self.selection_results:
            variances = self.selection_results['variance_threshold']['variances']
            axes[0, 1].hist(variances, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(variances), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(variances):.4f}')
            axes[0, 1].set_title('Feature Variance Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Variance')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # Method 3: Correlation heatmap (top features)
        if 'correlation_threshold' in self.selection_results:
            corr_matrix = self.selection_results['correlation_threshold']['correlation_matrix']
            # Show top 20x20 correlation matrix
            top_features = corr_matrix.columns[:20]
            corr_subset = corr_matrix.loc[top_features, top_features]
            
            im = axes[1, 0].imshow(corr_subset.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 0].set_title('Correlation Matrix (Top 20 Features)', fontweight='bold')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
            cbar.set_label('Correlation Coefficient')
        
        # Method 4: Statistical significance
        if 'statistical_f_classif' in self.selection_results:
            p_values = self.selection_results['statistical_f_classif']['p_values']
            
            # P-value distribution
            axes[1, 1].hist(p_values, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
            axes[1, 1].set_title('P-value Distribution (F-test)', fontweight='bold')
            axes[1, 1].set_xlabel('P-value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'selection_methods_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_scores(self, output_dir):
        """Plot statistical scores from different methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # F-scores
        if 'statistical_f_classif' in self.selection_results:
            f_scores = self.selection_results['statistical_f_classif']['scores']
            selected = self.selection_results['statistical_f_classif']['selected_features']
            
            # Sort features by F-score
            sorted_indices = np.argsort(f_scores)[-30:]  # Top 30
            top_scores = f_scores[sorted_indices]
            top_features = [self.feature_names[i] for i in sorted_indices]
            
            axes[0, 0].barh(range(len(top_scores)), top_scores, color='lightblue')
            axes[0, 0].set_yticks(range(len(top_scores)))
            axes[0, 0].set_yticklabels([f[:20] for f in top_features], fontsize=8)
            axes[0, 0].set_title('Top 30 Features by F-score', fontweight='bold')
            axes[0, 0].set_xlabel('F-score')
        
        # Mutual Information scores
        if 'statistical_mutual_info' in self.selection_results:
            mi_scores = self.selection_results['statistical_mutual_info']['scores']
            
            sorted_indices = np.argsort(mi_scores)[-30:]
            top_scores = mi_scores[sorted_indices]
            top_features = [self.feature_names[i] for i in sorted_indices]
            
            axes[0, 1].barh(range(len(top_scores)), top_scores, color='lightcoral')
            axes[0, 1].set_yticks(range(len(top_scores)))
            axes[0, 1].set_yticklabels([f[:20] for f in top_features], fontsize=8)
            axes[0, 1].set_title('Top 30 Features by Mutual Information', fontweight='bold')
            axes[0, 1].set_xlabel('Mutual Information Score')
        
        # T-test scores
        if 'statistical_t_test' in self.selection_results:
            t_scores = self.selection_results['statistical_t_test']['scores']
            
            sorted_indices = np.argsort(t_scores)[-30:]
            top_scores = t_scores[sorted_indices]
            top_features = [self.feature_names[i] for i in sorted_indices]
            
            axes[1, 0].barh(range(len(top_scores)), top_scores, color='lightgreen')
            axes[1, 0].set_yticks(range(len(top_scores)))
            axes[1, 0].set_yticklabels([f[:20] for f in top_features], fontsize=8)
            axes[1, 0].set_title('Top 30 Features by t-test', fontweight='bold')
            axes[1, 0].set_xlabel('|t-statistic|')
        
        # Combined ranking
        if 'combined_ranking' in self.selection_results:
            ranking_df = self.selection_results['combined_ranking']['ranking']
            top_30 = ranking_df.head(30)
            
            axes[1, 1].barh(range(len(top_30)), top_30['combined_score'].values, color='gold')
            axes[1, 1].set_yticks(range(len(top_30)))
            axes[1, 1].set_yticklabels([f[:20] for f in top_30['feature'].values], fontsize=8)
            axes[1, 1].set_title('Top 30 Features by Combined Ranking', fontweight='bold')
            axes[1, 1].set_xlabel('Combined Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistical_scores.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, output_dir):
        """Plot detailed correlation analysis"""
        if 'correlation_threshold' in self.selection_results:
            corr_data = self.selection_results['correlation_threshold']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # High correlation pairs
            high_corr_pairs = corr_data['high_corr_pairs']
            if high_corr_pairs:
                correlations = [pair[2] for pair in high_corr_pairs]
                
                axes[0].hist(correlations, bins=20, color='orange', alpha=0.7, edgecolor='black')
                axes[0].axvline(corr_data['threshold'], color='red', linestyle='--', 
                               label=f'Threshold: {corr_data["threshold"]}')
                axes[0].set_title('Distribution of High Correlations', fontweight='bold')
                axes[0].set_xlabel('Correlation Coefficient')
                axes[0].set_ylabel('Frequency')
                axes[0].legend()
            
            # Feature reduction effect
            original_features = len(self.feature_names)
            selected_features = corr_data['n_selected']
            removed_features = original_features - selected_features
            
            labels = ['Selected', 'Removed']
            sizes = [selected_features, removed_features]
            colors = ['lightgreen', 'lightcoral']
            
            axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1].set_title(f'Feature Reduction\n(Threshold: {corr_data["threshold"]})', 
                             fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_feature_rankings(self, output_dir):
        """Plot feature rankings comparison"""
        if 'combined_ranking' in self.selection_results:
            ranking_df = self.selection_results['combined_ranking']['ranking']
            methods = self.selection_results['combined_ranking']['methods']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Top features heatmap
            top_20 = ranking_df.head(20)
            score_columns = [f'{method}_score' for method in methods] + ['combined_score']
            heatmap_data = top_20[score_columns].T
            
            im = axes[0, 0].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
            axes[0, 0].set_xticks(range(len(top_20)))
            axes[0, 0].set_xticklabels([f[:15] for f in top_20['feature'].values], 
                                      rotation=45, ha='right', fontsize=8)
            axes[0, 0].set_yticks(range(len(score_columns)))
            axes[0, 0].set_yticklabels(score_columns)
            axes[0, 0].set_title('Score Heatmap (Top 20 Features)', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
            cbar.set_label('Normalized Score')
            
            # Ranking distribution
            combined_scores_clean = ranking_df['combined_score'].dropna()
            if len(combined_scores_clean) > 0:
                axes[0, 1].hist(combined_scores_clean, bins=30, color='purple', 
                               alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Combined Score Distribution', fontweight='bold')
                axes[0, 1].set_xlabel('Combined Score')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'No valid scores', ha='center', va='center',
                               transform=axes[0, 1].transAxes)
            
            # Method agreement
            if len(methods) >= 2:
                method1_scores = ranking_df[f'{methods[0]}_score'].values
                method2_scores = ranking_df[f'{methods[1]}_score'].values
                
                axes[1, 0].scatter(method1_scores, method2_scores, alpha=0.6, color='teal')
                axes[1, 0].set_xlabel(f'{methods[0]} Score')
                axes[1, 0].set_ylabel(f'{methods[1]} Score')
                axes[1, 0].set_title(f'Method Agreement: {methods[0]} vs {methods[1]}', 
                                    fontweight='bold')
                
                # Add correlation coefficient
                corr_coef = np.corrcoef(method1_scores, method2_scores)[0, 1]
                axes[1, 0].text(0.05, 0.95, f'r = {corr_coef:.3f}', 
                               transform=axes[1, 0].transAxes, 
                               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            # Top features bar plot
            top_15 = ranking_df.head(15)
            axes[1, 1].barh(range(len(top_15)), top_15['combined_score'].values, 
                           color='darkgreen', alpha=0.8)
            axes[1, 1].set_yticks(range(len(top_15)))
            axes[1, 1].set_yticklabels([f[:20] for f in top_15['feature'].values], fontsize=9)
            axes[1, 1].set_title('Top 15 Features (Combined Ranking)', fontweight='bold')
            axes[1, 1].set_xlabel('Combined Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_rankings.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_evaluation_results(self, output_dir):
        """Plot evaluation results"""
        if 'evaluation' in self.selection_results:
            eval_results = self.selection_results['evaluation']
            
            # Prepare data for plotting
            method_data = {}
            
            for method_clf, results in eval_results.items():
                parts = method_clf.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                    
                method, clf_part = parts
                
                # Handle classifier names
                if clf_part == 'Random':
                    clf = 'RandomForest'  
                elif clf_part == 'Logistic':
                    clf = 'LogisticRegression'
                else:
                    clf = clf_part
                
                if method not in method_data:
                    method_data[method] = {'n_features': results['n_features']}
                
                method_data[method][clf] = results['mean_score']
            
            # Extract data for plotting
            methods = list(method_data.keys())
            rf_scores = [method_data[m].get('RandomForest', 0) for m in methods]
            lr_scores = [method_data[m].get('LogisticRegression', 0) for m in methods]
            n_features = [method_data[m]['n_features'] for m in methods]
            
            if not methods:
                print("⚠️  No evaluation results to plot")
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Accuracy comparison
            x = np.arange(len(methods))
            width = 0.35
            
            axes[0].bar(x - width/2, rf_scores, width, label='Random Forest', 
                       color='lightblue', alpha=0.8)
            axes[0].bar(x + width/2, lr_scores, width, label='Logistic Regression', 
                       color='lightcoral', alpha=0.8)
            
            axes[0].set_xlabel('Feature Selection Method')
            axes[0].set_ylabel('Cross-Validation Accuracy')
            axes[0].set_title('Classification Performance by Feature Selection', fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Number of features vs accuracy (Random Forest)
            axes[1].scatter(n_features, rf_scores, color='blue', s=100, alpha=0.7)
            for i, method in enumerate(methods):
                axes[1].annotate(method.replace('_', ' ').title(), 
                               (n_features[i], rf_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1].set_xlabel('Number of Features')
            axes[1].set_ylabel('Random Forest Accuracy')
            axes[1].set_title('Features vs Performance Trade-off', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Performance improvement
            original_indices = [i for i, m in enumerate(methods) if 'original' in m.lower()]
            if original_indices:
                original_rf = rf_scores[original_indices[0]]
                improvements = [(score - original_rf) * 100 for score in rf_scores]
                
                colors = ['green' if imp > 0 else 'red' for imp in improvements]
                axes[2].bar(methods, improvements, color=colors, alpha=0.7)
                axes[2].set_xlabel('Feature Selection Method')
                axes[2].set_ylabel('Accuracy Improvement (%)')
                axes[2].set_title('Performance Improvement over Original', fontweight='bold')
                axes[2].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
                axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, 'No baseline for comparison', ha='center', va='center',
                           transform=axes[2].transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_selection_pipeline(self, output_dir):
        """Plot feature selection pipeline overview"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Pipeline stages
        stages = [
            f"Original Features\n({len(self.feature_names)} features)",
            "Variance Threshold\nFiltering",
            "Correlation-based\nFiltering", 
            "Statistical Tests\n(F-test, t-test, MI)",
            "Combined Ranking\n& Selection",
            "Evaluation\n(Cross-validation)"
        ]
        
        # Get numbers for each stage
        stage_numbers = [len(self.feature_names)]
        
        if 'variance_threshold' in self.selection_results:
            stage_numbers.append(self.selection_results['variance_threshold']['n_selected'])
        else:
            stage_numbers.append(len(self.feature_names))
            
        if 'correlation_threshold' in self.selection_results:
            stage_numbers.append(self.selection_results['correlation_threshold']['n_selected'])
        else:
            stage_numbers.append(stage_numbers[-1])
        
        # Add statistical test results
        if 'statistical_f_classif' in self.selection_results:
            stage_numbers.append(self.selection_results['statistical_f_classif']['n_selected'])
        else:
            stage_numbers.append(stage_numbers[-1])
        
        # Combined ranking (top 50)
        if 'combined_ranking' in self.selection_results:
            stage_numbers.append(50)  # Assuming top 50
        else:
            stage_numbers.append(stage_numbers[-1])
        
        # Final evaluation
        stage_numbers.append(stage_numbers[-1])
        
        # Plot pipeline
        positions = [(3, 8), (3, 6.5), (3, 5), (3, 3.5), (3, 2), (3, 0.5)]
        
        # Draw boxes and arrows
        box_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        arrow_props = dict(arrowstyle="->", connectionstyle="arc3", color="darkblue", lw=2)
        
        for i, (stage, (x, y), num) in enumerate(zip(stages, positions, stage_numbers)):
            # Draw box
            box_text = f"{stage}\n({num} features)"
            ax.text(x, y, box_text, ha='center', va='center', fontsize=11, 
                   bbox=box_props, fontweight='bold')
            
            # Draw arrow to next stage
            if i < len(stages) - 1:
                next_y = positions[i + 1][1]
                ax.annotate('', xy=(x, next_y + 0.6), xytext=(x, y - 0.6),
                           arrowprops=arrow_props)
        
        # Add filter method details on the right
        filter_details = [
            "Variance Threshold:\n• Remove low-variance features\n• Threshold: 0.0",
            "Correlation Filter:\n• Remove highly correlated pairs\n• Threshold: 0.95",
            "F-test:\n• ANOVA F-statistic\n• Select top K features",
            "t-test:\n• Independent samples t-test\n• Absolute t-statistic",
            "Mutual Information:\n• Non-linear dependencies\n• Information gain",
            "Combined Ranking:\n• Weighted combination\n• Consensus scoring"
        ]
        
        for i, detail in enumerate(filter_details):
            y_pos = 7.5 - i * 1.2
            ax.text(7, y_pos, detail, ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # Add results summary
        if 'evaluation' in self.selection_results:
            eval_results = self.selection_results['evaluation']
            best_method = None
            best_score = 0
            
            for method_clf, results in eval_results.items():
                if 'RandomForest' in method_clf and results['mean_score'] > best_score:
                    best_score = results['mean_score']
                    best_method = method_clf.replace('_RandomForest', '')
            
            if best_method:
                summary_text = f"Best Method: {best_method.replace('_', ' ').title()}\n"
                summary_text += f"Accuracy: {best_score:.4f}\n"
                summary_text += f"Dataset: {len(self.y)} samples\n"
                summary_text += f"PD: {np.sum(self.y)}, HC: {len(self.y)-np.sum(self.y)}"
                
                ax.text(7, 2, summary_text, ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                       fontweight='bold')
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-1, 9)
        ax.set_title('Filter-based Feature Selection Pipeline\nParkinson\'s Disease Detection', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'selection_pipeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_path="feature_selection_results.csv"):
        """Save feature selection results to CSV"""
        if 'combined_ranking' in self.selection_results:
            ranking_df = self.selection_results['combined_ranking']['ranking']
            ranking_df.to_csv(output_path, index=False)
            print(f"✅ Feature selection results saved to: {os.path.abspath(output_path)}")
        else:
            print("❌ No combined ranking results to save")


def main():
    """Main function to run filter-based feature selection"""
    print("🚀 FILTER-BASED FEATURE SELECTION")
    print("=" * 80)
    
    # Initialize selector
    selector = FilterFeatureSelector()
    
    # Load data
    if not selector.load_data():
        return
    
    # Apply different filter methods
    print("\n🔄 Applying Filter Methods...")
    
    # 1. Variance threshold
    selector.variance_threshold_selection(threshold=0.01)
    
    # 2. Correlation-based selection
    selector.correlation_based_selection(threshold=0.9)
    
    # 3. Statistical tests
    selector.statistical_tests_selection(method='f_classif', k=50)
    selector.statistical_tests_selection(method='t_test', k=50)
    selector.statistical_tests_selection(method='mutual_info', k=50)
    
    # 4. Combined ranking
    selector.combined_filter_ranking(
        methods=['f_classif', 't_test', 'mutual_info'],
        weights=[0.4, 0.3, 0.3]
    )
    
    # 5. Evaluate feature sets
    selector.evaluate_feature_sets()
    
    # 6. Create visualizations
    selector.create_comprehensive_visualizations()
    
    # 7. Save results
    selector.save_results()
    
    print(f"\n{'='*80}")
    print("✅ FILTER-BASED FEATURE SELECTION COMPLETE!")
    print(f"{'='*80}")
    print("📁 Output files:")
    print("   - feature_selection_results.csv")
    print("   - feature_selection_analysis/ (6 visualization plots)")


if __name__ == "__main__":
    main()

# FEATURE SELECTION PSEUDOCODE

## Comprehensive Multi-Method Feature Selection for Parkinson's Disease Detection

---

## 📋 OVERVIEW

**Purpose:** Select the most discriminative features from extracted PD voice features using multiple ranking methods

**Input:** Feature CSV files (windowed or non-windowed)

**Output:** Ranked features with importance scores and optimal feature subset

**Methods Used:**

1. Random Forest Feature Importance
2. Gradient Boosting Feature Importance
3. Univariate F-score Selection
4. Mutual Information
5. Aggregated Ranking (weighted combination)

---

## 🔄 MAIN ALGORITHM

```
ALGORITHM: ComprehensiveFeatureSelection
INPUT:
    - csv_file: Path to feature CSV file
    - feature_set_name: Name of the feature set (e.g., "5ms", "10ms", "Comprehensive")

OUTPUT:
    - rf_importance: Random Forest feature rankings
    - gb_importance: Gradient Boosting feature rankings
    - f_scores: Univariate F-score rankings
    - mi_scores: Mutual Information rankings
    - aggregated_rankings: Combined rankings from all methods
    - best_k: Optimal number of features
    - top_k_features: List of top K selected features

PROCEDURE:
1. CALL PrepareData(csv_file)
   → Returns: X (features), y (labels), feature_columns

2. CALL RandomForestImportance(X, y)
   → Returns: rf_importance_scores

3. CALL GradientBoostingImportance(X, y)
   → Returns: gb_importance_scores

4. CALL UnivariateSelection(X, y, k=30)
   → Returns: f_scores

5. CALL MutualInformationSelection(X, y)
   → Returns: mi_scores

6. CALL AggregateRankings(rf_importance, gb_importance, f_scores, mi_scores)
   → Returns: aggregated_rankings

7. CALL EvaluateFeatureSubsets(X, y, aggregated_rankings)
   → Returns: subset_evaluation_results, best_k

8. CALL SaveResults(all_rankings, best_k)
   → Saves all results to disk

9. RETURN all_results
END ALGORITHM
```

---

## 🔧 STEP 1: DATA PREPARATION

```
FUNCTION: PrepareData(csv_file)
INPUT: csv_file - Path to CSV file with features
OUTPUT: X (feature matrix), y (labels), feature_columns (list)

PROCEDURE:
1. Load CSV file into DataFrame df
   df ← READ_CSV(csv_file)

2. Print dataset statistics:
   PRINT("Total samples:", LENGTH(df))
   PRINT("Total columns:", NUMBER_OF_COLUMNS(df))

3. Define metadata columns to exclude:
   metadata_cols ← ['filename', 'duration', 'original_sample_rate',
                     'working_sample_rate', 'window_size_ms',
                     'hop_size_ms', 'total_windows', 'voiced_windows',
                     'num_f0_values', 'group']

4. Extract feature columns:
   feature_cols ← ALL_COLUMNS(df) - metadata_cols
   PRINT("Feature columns:", LENGTH(feature_cols))

5. Prepare feature matrix:
   X ← df[feature_cols]

6. Prepare labels:
   IF 'group' IN df.columns THEN:
       // Encode labels (HC, PD, SAMPLE) → (0, 1, 2)
       label_encoder ← LabelEncoder()
       y ← label_encoder.FIT_TRANSFORM(df['group'])

       // Show label distribution
       FOR each unique_label IN y:
           count ← COUNT_OCCURRENCES(y, unique_label)
           PRINT(label, ":", count, "samples")
   ELSE:
       PRINT("ERROR: No 'group' column found!")
       RETURN None, None, None

7. Handle missing values:
   IF X HAS_MISSING_VALUES THEN:
       missing_count ← COUNT_MISSING(X)
       PRINT("⚠️  Missing values found:", missing_count)
       X ← FILL_MISSING_WITH_ZERO(X)
       PRINT("   Filled with 0")

8. Remove constant features (no variance):
   constant_features ← []
   FOR each column IN X:
       IF STANDARD_DEVIATION(X[column]) == 0 THEN:
           ADD column TO constant_features

   IF LENGTH(constant_features) > 0 THEN:
       PRINT("🗑️  Removing", LENGTH(constant_features), "constant features")
       X ← DROP_COLUMNS(X, constant_features)

9. Print final statistics:
   PRINT("✅ Final feature count:", NUMBER_OF_COLUMNS(X))
   PRINT("✅ Final sample count:", NUMBER_OF_ROWS(X))

10. RETURN X, y, feature_cols
END FUNCTION
```

---

## 🌲 STEP 2: RANDOM FOREST FEATURE IMPORTANCE

```
FUNCTION: RandomForestImportance(X, y)
INPUT: X (feature matrix), y (labels)
OUTPUT: feature_importance_df (DataFrame with features and importance scores)

PROCEDURE:
1. Initialize Random Forest classifier:
   rf ← RandomForestClassifier(
       n_estimators = 100,
       random_state = 42,
       n_jobs = -1  // Use all CPU cores
   )

2. Train Random Forest:
   rf.FIT(X, y)

3. Extract feature importances:
   importances ← rf.feature_importances_

4. Create importance DataFrame:
   feature_importance_df ← DATAFRAME({
       'feature': X.columns,
       'importance': importances
   })

5. Sort by importance (descending):
   feature_importance_df ← SORT_BY(feature_importance_df, 'importance', DESCENDING)

6. Print top 10 features:
   PRINT("🌲 RANDOM FOREST FEATURE IMPORTANCE")
   PRINT("Top 10 most important features:")
   FOR i FROM 0 TO 9:
       feature ← feature_importance_df[i]['feature']
       score ← feature_importance_df[i]['importance']
       PRINT("  ", feature, ":", FORMAT(score, 4_DECIMALS))

7. RETURN feature_importance_df
END FUNCTION
```

---

## 📈 STEP 3: GRADIENT BOOSTING FEATURE IMPORTANCE

```
FUNCTION: GradientBoostingImportance(X, y)
INPUT: X (feature matrix), y (labels)
OUTPUT: feature_importance_df (DataFrame with features and importance scores)

PROCEDURE:
1. Initialize Gradient Boosting classifier:
   gb ← GradientBoostingClassifier(
       n_estimators = 100,
       random_state = 42
   )

2. Train Gradient Boosting:
   gb.FIT(X, y)

3. Extract feature importances:
   importances ← gb.feature_importances_

4. Create importance DataFrame:
   feature_importance_df ← DATAFRAME({
       'feature': X.columns,
       'importance': importances
   })

5. Sort by importance (descending):
   feature_importance_df ← SORT_BY(feature_importance_df, 'importance', DESCENDING)

6. Print top 10 features:
   PRINT("📈 GRADIENT BOOSTING FEATURE IMPORTANCE")
   PRINT("Top 10 most important features:")
   FOR i FROM 0 TO 9:
       feature ← feature_importance_df[i]['feature']
       score ← feature_importance_df[i]['importance']
       PRINT("  ", feature, ":", FORMAT(score, 4_DECIMALS))

7. RETURN feature_importance_df
END FUNCTION
```

---

## 📊 STEP 4: UNIVARIATE F-SCORE SELECTION

```
FUNCTION: UnivariateSelection(X, y, k)
INPUT: X (feature matrix), y (labels), k (number of top features to select)
OUTPUT: feature_scores_df (DataFrame with features and F-scores)

PROCEDURE:
1. Ensure k doesn't exceed number of features:
   k_actual ← MIN(k, NUMBER_OF_COLUMNS(X))

2. Initialize SelectKBest with F-score:
   selector ← SelectKBest(
       score_func = f_classif,  // ANOVA F-statistic
       k = k_actual
   )

3. Fit selector to data:
   selector.FIT(X, y)

4. Extract F-scores for all features:
   f_scores ← selector.scores_

5. Create scores DataFrame:
   feature_scores_df ← DATAFRAME({
       'feature': X.columns,
       'f_score': f_scores
   })

6. Sort by F-score (descending):
   feature_scores_df ← SORT_BY(feature_scores_df, 'f_score', DESCENDING)

7. Print top 10 features:
   PRINT("📊 UNIVARIATE FEATURE SELECTION (F-score)")
   PRINT("Top 10 features by F-score:")
   FOR i FROM 0 TO 9:
       feature ← feature_scores_df[i]['feature']
       score ← feature_scores_df[i]['f_score']
       PRINT("  ", feature, ":", FORMAT(score, 4_DECIMALS))

8. RETURN feature_scores_df
END FUNCTION
```

---

## 🔗 STEP 5: MUTUAL INFORMATION SELECTION

```
FUNCTION: MutualInformationSelection(X, y)
INPUT: X (feature matrix), y (labels)
OUTPUT: mi_df (DataFrame with features and MI scores)

PROCEDURE:
1. Calculate mutual information scores:
   mi_scores ← MUTUAL_INFO_CLASSIF(
       X, y,
       random_state = 42
   )

2. Create MI DataFrame:
   mi_df ← DATAFRAME({
       'feature': X.columns,
       'mi_score': mi_scores
   })

3. Sort by MI score (descending):
   mi_df ← SORT_BY(mi_df, 'mi_score', DESCENDING)

4. Print top 10 features:
   PRINT("🔗 MUTUAL INFORMATION FEATURE SELECTION")
   PRINT("Top 10 features by MI score:")
   FOR i FROM 0 TO 9:
       feature ← mi_df[i]['feature']
       score ← mi_df[i]['mi_score']
       PRINT("  ", feature, ":", FORMAT(score, 4_DECIMALS))

5. RETURN mi_df
END FUNCTION
```

---

## 🔄 STEP 6: AGGREGATE RANKINGS

```
FUNCTION: AggregateRankings(rf_importance, gb_importance, f_scores, mi_scores)
INPUT: Four DataFrames with feature importance from different methods
OUTPUT: aggregated_df (DataFrame with combined rankings)

PROCEDURE:
1. Define normalization function:
   FUNCTION Normalize(df, score_column):
       min_val ← MINIMUM(df[score_column])
       max_val ← MAXIMUM(df[score_column])
       df['normalized'] ← (df[score_column] - min_val) / (max_val - min_val)
       RETURN df

2. Normalize all scores to 0-1 range:
   rf_norm ← Normalize(rf_importance, 'importance')
   gb_norm ← Normalize(gb_importance, 'importance')
   f_norm ← Normalize(f_scores, 'f_score')
   mi_norm ← Normalize(mi_scores, 'mi_score')

3. Get all unique features:
   all_features ← UNION(rf_norm.features, gb_norm.features,
                        f_norm.features, mi_norm.features)

4. Aggregate scores for each feature:
   aggregated_list ← []
   FOR each feature IN all_features:
       // Get normalized score from each method (0 if not present)
       rf_score ← GET_NORMALIZED_SCORE(rf_norm, feature) OR 0
       gb_score ← GET_NORMALIZED_SCORE(gb_norm, feature) OR 0
       f_score ← GET_NORMALIZED_SCORE(f_norm, feature) OR 0
       mi_score ← GET_NORMALIZED_SCORE(mi_norm, feature) OR 0

       // Weighted average (weights: RF=30%, GB=30%, F=20%, MI=20%)
       aggregate_score ← (0.3 × rf_score +
                         0.3 × gb_score +
                         0.2 × f_score +
                         0.2 × mi_score)

       // Store all scores
       ADD TO aggregated_list: {
           'feature': feature,
           'rf_score': rf_score,
           'gb_score': gb_score,
           'f_score': f_score,
           'mi_score': mi_score,
           'aggregate_score': aggregate_score
       }

5. Create aggregated DataFrame:
   aggregated_df ← DATAFRAME(aggregated_list)

6. Sort by aggregate score (descending):
   aggregated_df ← SORT_BY(aggregated_df, 'aggregate_score', DESCENDING)

7. Print top 15 features:
   PRINT("🔄 AGGREGATED FEATURE RANKINGS")
   PRINT("Top 15 features by aggregated score:")
   FOR i FROM 0 TO 14:
       feature ← aggregated_df[i]['feature']
       score ← aggregated_df[i]['aggregate_score']
       PRINT("  ", feature, ":", FORMAT(score, 4_DECIMALS))

8. RETURN aggregated_df
END FUNCTION
```

---

## 🎯 STEP 7: EVALUATE FEATURE SUBSETS

```
FUNCTION: EvaluateFeatureSubsets(X, y, feature_importance_df)
INPUT: X (features), y (labels), feature_importance_df (ranked features)
OUTPUT: results_df (evaluation results), best_k (optimal number of features)

PROCEDURE:
1. Determine sample size and class distribution:
   n_samples ← NUMBER_OF_ROWS(X)
   unique_labels, counts ← COUNT_UNIQUE(y)
   min_class_count ← MINIMUM(counts)

2. Determine appropriate CV folds:
   cv_folds ← MIN(5, min_class_count)

   IF cv_folds < 2 THEN:
       PRINT("⚠️  Insufficient samples for cross-validation")
       PRINT("   Skipping subset evaluation")
       RETURN empty_results, NUMBER_OF_FEATURES(X)

   PRINT("ℹ️  Using", cv_folds, "-fold cross-validation")
   PRINT("   (", n_samples, "samples, min class count:", min_class_count, ")")

3. Define subset sizes to test:
   subset_sizes ← [5, 10, 15, 20, 30, 40, 50]

4. Evaluate each subset size:
   results ← []
   FOR each k IN subset_sizes:
       IF k > NUMBER_OF_FEATURES(feature_importance_df) THEN:
           CONTINUE  // Skip if k exceeds available features

       // Select top k features
       top_features ← GET_TOP_K_FEATURES(feature_importance_df, k)
       X_subset ← X[top_features]

       // Train Random Forest with cross-validation
       rf ← RandomForestClassifier(
           n_estimators = 100,
           random_state = 42,
           n_jobs = -1
       )

       cv_scores ← CROSS_VALIDATE(
           estimator = rf,
           X = X_subset,
           y = y,
           cv = cv_folds,
           scoring = 'accuracy'
       )

       // Store results
       mean_acc ← MEAN(cv_scores)
       std_acc ← STANDARD_DEVIATION(cv_scores)

       ADD TO results: {
           'k': k,
           'mean_accuracy': mean_acc,
           'std_accuracy': std_acc
       }

       PRINT("  K=", k, "features: Accuracy =",
             FORMAT(mean_acc, 4_DECIMALS), "±",
             FORMAT(std_acc, 4_DECIMALS))

5. Handle empty results:
   IF LENGTH(results) == 0 THEN:
       PRINT("⚠️  No valid subset sizes found")
       RETURN empty_results, NUMBER_OF_FEATURES(feature_importance_df)

6. Create results DataFrame:
   results_df ← DATAFRAME(results)

7. Find best k:
   best_idx ← INDEX_OF_MAX(results_df['mean_accuracy'])
   best_k ← results_df[best_idx]['k']
   best_acc ← results_df[best_idx]['mean_accuracy']

   PRINT("✅ Best subset size: K=", best_k,
         "(Accuracy:", FORMAT(best_acc, 4_DECIMALS), ")")

8. RETURN results_df, best_k
END FUNCTION
```

---

## 💾 STEP 8: SAVE RESULTS

```
FUNCTION: SaveResults(rf_importance, gb_importance, f_scores, mi_scores,
                      aggregated, results_df, best_k, feature_set_name)
INPUT: All ranking DataFrames, evaluation results, best K, feature set name
OUTPUT: output_dir (path to saved results)

PROCEDURE:
1. Create output directory:
   output_dir ← "feature_selection_results/" + feature_set_name
   CREATE_DIRECTORY(output_dir)

2. Save individual rankings:
   SAVE_CSV(rf_importance, output_dir + "/rf_feature_importance.csv")
   PRINT("✅ Saved: rf_feature_importance.csv")

   SAVE_CSV(gb_importance, output_dir + "/gb_feature_importance.csv")
   PRINT("✅ Saved: gb_feature_importance.csv")

   SAVE_CSV(f_scores, output_dir + "/f_score_ranking.csv")
   PRINT("✅ Saved: f_score_ranking.csv")

   SAVE_CSV(mi_scores, output_dir + "/mi_score_ranking.csv")
   PRINT("✅ Saved: mi_score_ranking.csv")

3. Save aggregated rankings:
   SAVE_CSV(aggregated, output_dir + "/aggregated_rankings.csv")
   PRINT("✅ Saved: aggregated_rankings.csv")

4. Save subset evaluation results:
   SAVE_CSV(results_df, output_dir + "/subset_evaluation.csv")
   PRINT("✅ Saved: subset_evaluation.csv")

5. Save top K features:
   top_features ← GET_TOP_K_FEATURES(aggregated, best_k)
   top_features_df ← DATAFRAME({'feature': top_features})
   SAVE_CSV(top_features_df, output_dir + "/top_" + best_k + "_features.csv")
   PRINT("✅ Saved: top_", best_k, "_features.csv")

6. RETURN output_dir
END FUNCTION
```

---

## 🔁 MAIN EXECUTION FLOW

```
ALGORITHM: ProcessAllFeatureSets
INPUT: None (reads predefined CSV files)
OUTPUT: all_results (dictionary), summary_df (DataFrame)

PROCEDURE:
1. Define all feature sets to process:
   feature_sets ← [
       {name: "Windowed_5ms", file: "comprehensive_features/windowed_pd_features_5ms.csv"},
       {name: "Windowed_10ms", file: "comprehensive_features/windowed_pd_features_10ms.csv"},
       {name: "Windowed_20ms", file: "comprehensive_features/windowed_pd_features_20ms.csv"},
       {name: "Comprehensive_Final", file: "comprehensive_features/comprehensive_pd_features_final.csv"}
   ]

2. Initialize results storage:
   all_results ← {}

3. Process each feature set:
   FOR each feature_set IN feature_sets:
       name ← feature_set.name
       file ← feature_set.file

       // Check file exists
       IF NOT FILE_EXISTS(file) THEN:
           PRINT("⚠️  Skipping", name, ": File not found")
           CONTINUE

       // Run feature selection
       PRINT("\n" + "="*70)
       PRINT("🎯 FEATURE SELECTION:", name)
       PRINT("="*70)

       selector ← ComprehensiveFeatureSelector(file, name)
       results ← selector.RunCompleteAnalysis()

       IF results IS NOT None THEN:
           all_results[name] ← results

4. Generate summary comparison:
   summary_data ← []
   FOR each name, results IN all_results:
       ADD TO summary_data: {
           'Feature Set': name,
           'Best K': results['best_k'],
           'Best Accuracy': MAX(results['results_df']['mean_accuracy']),
           'Top Feature': results['aggregated'][0]['feature'],
           'Top Score': results['aggregated'][0]['aggregate_score']
       }

5. Create summary DataFrame:
   summary_df ← DATAFRAME(summary_data)

6. Print summary table:
   PRINT("\n" + "="*70)
   PRINT("📊 SUMMARY COMPARISON ACROSS ALL FEATURE SETS")
   PRINT("="*70)
   PRINT(summary_df)

7. Save summary:
   SAVE_CSV(summary_df, "feature_selection_results/summary_comparison.csv")
   PRINT("✅ Summary saved to: feature_selection_results/summary_comparison.csv")

8. Final message:
   PRINT("\n" + "="*70)
   PRINT("🎉 ALL FEATURE SELECTION ANALYSES COMPLETE!")
   PRINT("="*70)

9. RETURN all_results, summary_df
END ALGORITHM
```

---

## 📊 DATA STRUCTURES

### Feature Importance DataFrame Structure:

```
| feature          | importance/score |
|------------------|------------------|
| mfcc_3_mean      | 0.8440          |
| hnr              | 0.6105          |
| mfcc_4_std       | 0.5710          |
| ...              | ...             |
```

### Aggregated Rankings Structure:

```
| feature    | rf_score | gb_score | f_score | mi_score | aggregate_score |
|------------|----------|----------|---------|----------|-----------------|
| mfcc_3_mean| 0.7543   | 0.9821   | 0.8234  | 0.9414   | 0.8440         |
| hnr        | 0.6234   | 0.7123   | 0.4532  | 0.5234   | 0.6105         |
| ...        | ...      | ...      | ...     | ...      | ...            |
```

### Subset Evaluation Structure:

```
| k  | mean_accuracy | std_accuracy |
|----|---------------|--------------|
| 5  | 1.0000       | 0.0000       |
| 10 | 1.0000       | 0.0000       |
| 15 | 1.0000       | 0.0000       |
| 20 | 0.9500       | 0.1000       |
| ...| ...          | ...          |
```

---

## 🎯 KEY PARAMETERS

### Random Forest Parameters:

- `n_estimators`: 100 trees
- `random_state`: 42 (for reproducibility)
- `n_jobs`: -1 (use all CPU cores)

### Gradient Boosting Parameters:

- `n_estimators`: 100 trees
- `random_state`: 42 (for reproducibility)

### Cross-Validation Strategy:

- Adaptive fold selection: `cv_folds = MIN(5, min_class_count)`
- Ensures proper stratification
- Minimum 2 folds required

### Aggregation Weights:

- Random Forest: 30%
- Gradient Boosting: 30%
- F-score: 20%
- Mutual Information: 20%

### Subset Sizes Tested:

- [5, 10, 15, 20, 30, 40, 50] features

---

## 📈 OUTPUT FILES

For each feature set, the following files are generated:

1. **rf_feature_importance.csv** - Random Forest rankings
2. **gb_feature_importance.csv** - Gradient Boosting rankings
3. **f_score_ranking.csv** - Univariate F-score rankings
4. **mi_score_ranking.csv** - Mutual Information rankings
5. **aggregated_rankings.csv** - Combined rankings from all methods
6. **subset_evaluation.csv** - Performance vs number of features
7. **top_K_features.csv** - Final selected top K features

Plus one summary file:

- **summary_comparison.csv** - Comparison across all feature sets

---

## 🔍 COMPLEXITY ANALYSIS

### Time Complexity:

**For each feature set:**

- Data Preparation: O(n × m) where n=samples, m=features
- Random Forest: O(n × m × t × log(n)) where t=trees (100)
- Gradient Boosting: O(n × m × t × log(n))
- F-score: O(n × m)
- Mutual Information: O(n × m × log(n))
- Aggregation: O(m)
- Subset Evaluation: O(k × n × m × t × cv) where k=subset sizes, cv=folds

**Overall:** O(k × n × m × t × cv × log(n))

### Space Complexity:

- Feature matrix: O(n × m)
- Importance arrays: O(m)
- CV results: O(k)
- **Total:** O(n × m + k)

---

## ⚠️ IMPORTANT NOTES

1. **Sample Size Dependency:**

   - CV folds automatically adjusted based on minimum class count
   - Small datasets (< 10 samples) may have limited CV reliability
   - Recommendation: At least 30 samples per class for robust results

2. **Feature Scaling:**

   - All importance scores normalized to 0-1 range before aggregation
   - Ensures fair comparison across methods
   - No need for pre-scaling input features (tree-based methods)

3. **Missing Values:**

   - Filled with 0 (conservative approach)
   - Alternative: Use median or mean imputation
   - Constant features removed automatically

4. **Reproducibility:**

   - `random_state=42` used throughout
   - Ensures consistent results across runs
   - Important for research and production

5. **Computational Efficiency:**
   - Uses `n_jobs=-1` for parallel processing
   - Significant speedup on multi-core systems
   - May need adjustment on memory-limited systems

---

## 📝 USAGE EXAMPLE

```python
# Import the module
from comprehensive_feature_selection import ComprehensiveFeatureSelector

# Create selector for a specific feature set
selector = ComprehensiveFeatureSelector(
    csv_file="comprehensive_features/windowed_pd_features_10ms.csv",
    feature_set_name="Windowed_10ms"
)

# Run complete analysis
results = selector.run_complete_analysis()

# Access results
print("Best K:", results['best_k'])
print("Top features:", results['aggregated'].head(5))
```

---

**Pseudocode Version:** 1.0  
**Date:** October 2025  
**Author:** Parkinson's Disease Feature Selection System  
**Status:** Production-Ready

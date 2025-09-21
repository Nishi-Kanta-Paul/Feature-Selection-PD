#!/usr/bin/env python3
"""
CSV Data Viewer for Feature Importance Analysis
===============================================

This script helps view and understand the CSV files generated 
by the feature importance analysis.
"""

import pandas as pd
import os

def view_detailed_scores():
    """View detailed feature importance scores from each model"""
    print("📊 DETAILED FEATURE IMPORTANCE SCORES")
    print("=" * 50)
    
    file_path = "shap_analysis_results/detailed_feature_importance_scores.csv"
    
    if not os.path.exists(file_path):
        print("❌ File not found. Please run shap_feature_selection.py first.")
        return
    
    df = pd.read_csv(file_path)
    
    print(f"📈 Total records: {len(df)}")
    print(f"🤖 Models: {df['model'].unique().tolist()}")
    print(f"🔧 Features: {df['feature'].nunique()}")
    
    print("\n🌲 RandomForest Top 10 Features:")
    rf_data = df[df['model'] == 'RandomForest'].head(10)
    for i, row in rf_data.iterrows():
        print(f"   {i+1:2d}. {row['feature']:20s} - {row['importance']:.6f}")
    
    print("\n🚀 XGBoost Top 10 Features:")
    xgb_data = df[df['model'] == 'XGBoost'].head(10)
    for i, row in xgb_data.iterrows():
        print(f"   {i+1:2d}. {row['feature']:20s} - {row['importance']:.6f}")
    
    print(f"\n💾 Full data available in: {file_path}")

def view_aggregated_scores():
    """View aggregated feature importance scores"""
    print("\n📊 AGGREGATED FEATURE IMPORTANCE SCORES")
    print("=" * 50)
    
    file_path = "shap_analysis_results/aggregated_feature_importance.csv"
    
    if not os.path.exists(file_path):
        print("❌ File not found. Please run shap_feature_selection.py first.")
        return
    
    df = pd.read_csv(file_path)
    
    print(f"📈 Total features: {len(df)}")
    print(f"🏆 Top 15 selected features")
    
    print("\n🥇 ALL FEATURES RANKED BY IMPORTANCE:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Feature':<20} {'Mean':<10} {'Std':<10} {'Status':<15}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        status = "✅ Selected" if row['rank'] <= 15 else "❌ Not selected"
        print(f"{row['rank']:<4} {row['feature']:<20} {row['mean']:<10.6f} {row['std']:<10.6f} {status:<15}")
    
    print(f"\n📊 Statistics:")
    print(f"   Highest importance: {df['mean'].max():.6f} ({df.iloc[0]['feature']})")
    print(f"   15th place: {df.iloc[14]['mean']:.6f} ({df.iloc[14]['feature']})")
    print(f"   Lowest importance: {df['mean'].min():.6f} ({df.iloc[-1]['feature']})")
    print(f"   Selection threshold: ≥ {df.iloc[14]['mean']:.6f}")
    
    print(f"\n💾 Full data available in: {file_path}")

def compare_models():
    """Compare how different models rank features"""
    print("\n🔄 MODEL COMPARISON")
    print("=" * 50)
    
    file_path = "shap_analysis_results/detailed_feature_importance_scores.csv"
    
    if not os.path.exists(file_path):
        print("❌ File not found. Please run shap_feature_selection.py first.")
        return
    
    df = pd.read_csv(file_path)
    
    # Pivot to compare models side by side
    pivot_df = df.pivot(index='feature', columns='model', values='importance')
    pivot_df = pivot_df.fillna(0)
    pivot_df['difference'] = abs(pivot_df['RandomForest'] - pivot_df['XGBoost'])
    pivot_df = pivot_df.sort_values('difference', ascending=False)
    
    print("🤔 FEATURES WITH BIGGEST MODEL DISAGREEMENT:")
    print("-" * 70)
    print(f"{'Feature':<20} {'RandomForest':<12} {'XGBoost':<12} {'Difference':<12}")
    print("-" * 70)
    
    for feature, row in pivot_df.head(10).iterrows():
        print(f"{feature:<20} {row['RandomForest']:<12.6f} {row['XGBoost']:<12.6f} {row['difference']:<12.6f}")
    
    print("\n🤝 FEATURES WITH HIGHEST CONSENSUS:")
    print("-" * 70)
    print(f"{'Feature':<20} {'RandomForest':<12} {'XGBoost':<12} {'Difference':<12}")
    print("-" * 70)
    
    for feature, row in pivot_df.tail(10).iterrows():
        print(f"{feature:<20} {row['RandomForest']:<12.6f} {row['XGBoost']:<12.6f} {row['difference']:<12.6f}")

def show_calculation_example():
    """Show step-by-step calculation for top features"""
    print("\n🧮 CALCULATION EXAMPLE")
    print("=" * 50)
    
    detailed_file = "shap_analysis_results/detailed_feature_importance_scores.csv"
    aggregated_file = "shap_analysis_results/aggregated_feature_importance.csv"
    
    if not (os.path.exists(detailed_file) and os.path.exists(aggregated_file)):
        print("❌ Files not found. Please run shap_feature_selection.py first.")
        return
    
    detailed_df = pd.read_csv(detailed_file)
    aggregated_df = pd.read_csv(aggregated_file)
    
    print("📝 Step-by-step calculation for top 5 features:")
    print("-" * 80)
    
    top_5 = aggregated_df.head(5)
    
    for _, row in top_5.iterrows():
        feature = row['feature']
        
        # Get individual scores
        rf_score = detailed_df[(detailed_df['feature'] == feature) & 
                              (detailed_df['model'] == 'RandomForest')]['importance'].iloc[0]
        xgb_score = detailed_df[(detailed_df['feature'] == feature) & 
                               (detailed_df['model'] == 'XGBoost')]['importance'].iloc[0]
        
        calculated_mean = (rf_score + xgb_score) / 2
        calculated_std = ((rf_score - calculated_mean)**2 + (xgb_score - calculated_mean)**2)**0.5
        
        print(f"\n🔢 Feature: {feature}")
        print(f"   RandomForest score: {rf_score:.6f}")
        print(f"   XGBoost score:     {xgb_score:.6f}")
        print(f"   Mean calculation:  ({rf_score:.6f} + {xgb_score:.6f}) / 2 = {calculated_mean:.6f}")
        print(f"   Stored mean:       {row['mean']:.6f}")
        print(f"   Standard deviation: {calculated_std:.6f}")
        print(f"   Stored std:        {row['std']:.6f}")
        print(f"   Rank:              {row['rank']}")

def main():
    """Main function to run all views"""
    print("🔬 FEATURE IMPORTANCE CSV DATA VIEWER")
    print("=" * 60)
    
    # Check if results exist
    if not os.path.exists("shap_analysis_results"):
        print("❌ No results found!")
        print("Please run: python shap_feature_selection.py")
        return
    
    view_detailed_scores()
    view_aggregated_scores() 
    compare_models()
    show_calculation_example()
    
    print("\n" + "=" * 60)
    print("✅ Data viewing complete!")
    print("📁 All CSV files are in: shap_analysis_results/")
    print("📖 Full documentation: shap_analysis_results/calculation_process_documentation.md")

if __name__ == "__main__":
    main()
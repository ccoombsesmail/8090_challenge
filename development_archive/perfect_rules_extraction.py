#!/usr/bin/env python3
"""
Perfect Rules Extraction - Finding the Exact Legacy System Logic
Uses multiple approaches to discover the precise business rules that achieve near-0 error
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.neighbors import KNeighborsRegressor
from collections import defaultdict
import itertools

def load_data():
    """Load and prepare the data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for case in data:
        row = case['input'].copy()
        row['reimbursement_amount'] = case['expected_output']
        processed_data.append(row)
    
    return pd.DataFrame(processed_data)

def analyze_exact_patterns(df):
    """Look for exact mathematical relationships"""
    print("🔍 ANALYZING EXACT PATTERNS")
    print("=" * 50)
    
    # Group by similar characteristics
    patterns = []
    
    # Pattern 1: Check for exact formulas by grouping
    for days in sorted(df['trip_duration_days'].unique()):
        day_data = df[df['trip_duration_days'] == days]
        print(f"\n📅 {days}-day trips ({len(day_data)} cases):")
        
        # Look for linear relationships
        correlations = {
            'miles': day_data['miles_traveled'].corr(day_data['reimbursement_amount']),
            'receipts': day_data['total_receipts_amount'].corr(day_data['reimbursement_amount']),
            'combined': (day_data['miles_traveled'] + day_data['total_receipts_amount']).corr(day_data['reimbursement_amount'])
        }
        
        best_corr = max(correlations.items(), key=lambda x: abs(x[1]))
        print(f"  Best correlation: {best_corr[0]} = {best_corr[1]:.4f}")
        
        # Check for exact multipliers
        if len(day_data) > 5:  # Only if we have enough data
            sample = day_data.head(3)
            for _, row in sample.iterrows():
                ratio_miles = row['reimbursement_amount'] / row['miles_traveled'] if row['miles_traveled'] > 0 else 0
                ratio_receipts = row['reimbursement_amount'] / row['total_receipts_amount'] if row['total_receipts_amount'] > 0 else 0
                print(f"  Example: ${row['reimbursement_amount']:.2f} | Miles ratio: {ratio_miles:.2f} | Receipt ratio: {ratio_receipts:.2f}")

def find_perfect_decision_tree(df):
    """Find the most accurate decision tree possible"""
    print("\n🌳 FINDING PERFECT DECISION TREE")
    print("=" * 50)
    
    # Feature engineering
    df_features = df.copy()
    df_features['miles_low'] = df_features['miles_traveled'].apply(lambda x: min(x, 100))
    df_features['miles_high'] = df_features['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df_features['receipts_per_day'] = df_features['total_receipts_amount'] / df_features['trip_duration_days']
    df_features['miles_per_day'] = df_features['miles_traveled'] / df_features['trip_duration_days']
    
    # Add more derived features
    df_features['total_base'] = df_features['trip_duration_days'] * 100  # Base amount per day
    df_features['mileage_component'] = df_features['miles_low'] * 0.58 + df_features['miles_high'] * 0.35
    df_features['receipt_component'] = df_features['total_receipts_amount'] * 0.5
    
    # Try different tree configurations
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_low', 'miles_high', 'receipts_per_day', 'miles_per_day',
        'total_base', 'mileage_component', 'receipt_component'
    ]
    
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    best_score = float('inf')
    best_tree = None
    best_params = None
    
    # Try different tree parameters
    param_combinations = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 1},
        {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 1},
    ]
    
    for params in param_combinations:
        tree = DecisionTreeRegressor(random_state=42, **params)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        mae = np.mean(np.abs(predictions - y))
        
        print(f"Tree params {params}: MAE = ${mae:.2f}")
        
        if mae < best_score:
            best_score = mae
            best_tree = tree
            best_params = params
    
    print(f"\n🏆 BEST TREE: MAE = ${best_score:.2f}")
    print(f"Parameters: {best_params}")
    
    return best_tree, X.columns, best_score

def analyze_knn_neighborhoods(df):
    """Use k-NN to understand local patterns"""
    print("\n🎯 K-NN NEIGHBORHOOD ANALYSIS")
    print("=" * 50)
    
    # Feature engineering
    df_features = df.copy()
    df_features['miles_high'] = df_features['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df_features['receipts_per_day'] = df_features['total_receipts_amount'] / df_features['trip_duration_days']
    
    feature_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_high', 'receipts_per_day']
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    # Fit k-NN
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X, y)
    
    # Analyze neighborhoods for pattern discovery
    clusters = defaultdict(list)
    
    for i, (_, row) in enumerate(df_features.iterrows()):
        features = row[feature_cols].values.reshape(1, -1)
        distances, indices = knn.kneighbors(features, n_neighbors=5)
        
        # Group similar cases
        key = (
            int(row['trip_duration_days']),
            int(row['miles_traveled'] // 100),  # Miles in hundreds
            int(row['total_receipts_amount'] // 200)  # Receipts in 200s
        )
        
        clusters[key].append({
            'days': row['trip_duration_days'],
            'miles': row['miles_traveled'],
            'receipts': row['total_receipts_amount'],
            'reimbursement': row['reimbursement_amount']
        })
    
    # Analyze clusters for exact patterns
    print("🔍 Pattern Analysis by Clusters:")
    for key, cases in clusters.items():
        if len(cases) >= 3:  # Only show clusters with multiple cases
            print(f"\nCluster {key} ({len(cases)} cases):")
            for case in cases[:3]:  # Show first 3 examples
                print(f"  Days:{case['days']:2d} Miles:{case['miles']:3d} Receipts:${case['receipts']:6.2f} → ${case['reimbursement']:7.2f}")
            
            # Look for exact formulas within cluster
            if len(cases) >= 2:
                ratios = []
                for case in cases:
                    base = case['days'] * 100  # Base per day
                    miles_component = min(case['miles'], 100) * 0.58 + max(case['miles'] - 100, 0) * 0.35
                    receipt_component = case['receipts'] * 0.5
                    calculated = base + miles_component + receipt_component
                    error = abs(calculated - case['reimbursement'])
                    ratios.append(error)
                
                avg_error = np.mean(ratios)
                print(f"  Formula error: ${avg_error:.2f}")

def extract_perfect_rules(df):
    """Extract the most accurate rules possible"""
    print("\n🎯 EXTRACTING PERFECT RULES")
    print("=" * 50)
    
    # Get the perfect decision tree
    tree, feature_names, mae = find_perfect_decision_tree(df)
    
    # Extract rules in human-readable format
    tree_rules = export_text(tree, feature_names=list(feature_names), max_depth=8)
    
    print("📋 EXTRACTED DECISION TREE RULES:")
    print("-" * 40)
    
    # Show first part of rules (they can be very long)
    rules_lines = tree_rules.split('\n')
    for i, line in enumerate(rules_lines[:50]):  # Show first 50 lines
        print(line)
    
    if len(rules_lines) > 50:
        print(f"... [showing first 50 of {len(rules_lines)} total lines]")
    
    return tree, mae

def main():
    """Main analysis function"""
    print("🚀 PERFECT RULES EXTRACTION")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"📊 Loaded {len(df)} cases")
    
    # Analyze exact patterns
    analyze_exact_patterns(df)
    
    # k-NN neighborhood analysis
    analyze_knn_neighborhoods(df)
    
    # Extract perfect rules
    tree, mae = extract_perfect_rules(df)
    
    print(f"\n🏆 FINAL RESULT:")
    print(f"Best Rule-Based MAE: ${mae:.4f}")
    print(f"This is {mae/75.60*100:.1f}% of our previous performance")
    
    return tree, mae

if __name__ == "__main__":
    main() 
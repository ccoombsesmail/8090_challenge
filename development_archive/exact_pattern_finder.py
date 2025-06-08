#!/usr/bin/env python3
"""
Exact Pattern Finder - Discovering the Precise Mathematical Rules
Focus on finding exact formulas that achieve near-0 error
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict

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

def find_perfect_tree(df):
    """Find the most accurate decision tree possible"""
    print("🌳 FINDING PERFECT DECISION TREE")
    print("=" * 50)
    
    # Feature engineering
    df_features = df.copy()
    df_features['miles_low'] = df_features['miles_traveled'].apply(lambda x: min(x, 100))
    df_features['miles_high'] = df_features['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df_features['receipts_per_day'] = df_features['total_receipts_amount'] / df_features['trip_duration_days']
    df_features['miles_per_day'] = df_features['miles_traveled'] / df_features['trip_duration_days']
    
    # Core features
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_low', 'miles_high', 'receipts_per_day', 'miles_per_day'
    ]
    
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    # Try increasingly complex trees until we get near-perfect accuracy
    best_mae = float('inf')
    best_tree = None
    
    configs = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1}
    ]
    
    for config in configs:
        tree = DecisionTreeRegressor(random_state=42, **config)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        mae = np.mean(np.abs(predictions - y))
        exact_matches = np.sum(np.abs(predictions - y) < 0.01)
        
        print(f"Config {config}")
        print(f"  MAE: ${mae:.6f}")
        print(f"  Exact matches: {exact_matches}/{len(y)} ({exact_matches/len(y)*100:.1f}%)")
        print(f"  Tree depth: {tree.get_depth()}")
        print(f"  Number of leaves: {tree.get_n_leaves()}")
        
        if mae < best_mae:
            best_mae = mae
            best_tree = tree
    
    print(f"\n🏆 BEST PERFORMANCE: ${best_mae:.6f} MAE")
    return best_tree, feature_cols, best_mae

def analyze_exact_clusters(df):
    """Find groups of cases with identical or near-identical patterns"""
    print("\n🔍 ANALYZING EXACT CLUSTERS")
    print("=" * 50)
    
    # Group by combinations of inputs
    clusters = defaultdict(list)
    
    for _, row in df.iterrows():
        # Create a key based on rounded values
        key = (
            int(row['trip_duration_days']),
            int(row['miles_traveled'] // 50) * 50,  # Round to nearest 50
            int(row['total_receipts_amount'] // 100) * 100  # Round to nearest 100
        )
        clusters[key].append(row)
    
    # Analyze clusters with multiple entries
    print("📊 Clusters with multiple similar cases:")
    
    for key, cases in clusters.items():
        if len(cases) >= 3:  # Show clusters with 3+ similar cases
            print(f"\nCluster {key} - {len(cases)} cases:")
            
            for i, case in enumerate(cases[:5]):  # Show first 5
                print(f"  {i+1}: {case['trip_duration_days']:.0f} days, "
                      f"{case['miles_traveled']:.0f} miles, "
                      f"${case['total_receipts_amount']:.2f} receipts → "
                      f"${case['reimbursement_amount']:.2f}")
            
            # Look for patterns within cluster
            reimbursements = [c['reimbursement_amount'] for c in cases]
            if len(set(reimbursements)) == 1:
                print(f"  ✅ EXACT PATTERN: All cases → ${reimbursements[0]:.2f}")
            else:
                print(f"  📈 Range: ${min(reimbursements):.2f} - ${max(reimbursements):.2f}")

def test_formula_hypotheses(df):
    """Test various mathematical formula hypotheses"""
    print("\n🧮 TESTING FORMULA HYPOTHESES")
    print("=" * 50)
    
    def test_formula(name, formula_func):
        """Test a formula and return its accuracy"""
        predictions = df.apply(formula_func, axis=1)
        errors = np.abs(predictions - df['reimbursement_amount'])
        mae = np.mean(errors)
        exact_matches = np.sum(errors < 0.01)
        
        print(f"{name}:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  Exact matches: {exact_matches}/{len(df)} ({exact_matches/len(df)*100:.1f}%)")
        print(f"  Max error: ${np.max(errors):.2f}")
        
        return mae
    
    # Hypothesis 1: Base + Mileage Tiers + Receipt Component
    def formula_1(row):
        base = row['trip_duration_days'] * 100
        miles_component = min(row['miles_traveled'], 100) * 0.58 + max(row['miles_traveled'] - 100, 0) * 0.35
        receipt_component = row['total_receipts_amount'] * 0.5
        return base + miles_component + receipt_component
    
    # Hypothesis 2: Piecewise based on receipt thresholds
    def formula_2(row):
        if row['total_receipts_amount'] <= 828:
            # Lower receipt formula
            base = row['trip_duration_days'] * 50
            miles_bonus = row['miles_traveled'] * 0.4
            receipt_bonus = row['total_receipts_amount'] * 0.3
        else:
            # Higher receipt formula
            base = row['trip_duration_days'] * 150
            miles_bonus = row['miles_traveled'] * 0.2
            receipt_bonus = row['total_receipts_amount'] * 0.8
        
        return base + miles_bonus + receipt_bonus
    
    # Hypothesis 3: Complex tiered system
    def formula_3(row):
        days = row['trip_duration_days']
        miles = row['miles_traveled']
        receipts = row['total_receipts_amount']
        
        # Base calculation
        if days <= 2:
            base = days * 80
        elif days <= 5:
            base = days * 120
        else:
            base = days * 100
        
        # Mileage tiers
        if miles <= 100:
            mile_component = miles * 0.6
        elif miles <= 500:
            mile_component = 100 * 0.6 + (miles - 100) * 0.4
        else:
            mile_component = 100 * 0.6 + 400 * 0.4 + (miles - 500) * 0.2
        
        # Receipt component with threshold
        if receipts <= 828:
            receipt_component = receipts * 0.2
        else:
            receipt_component = 828 * 0.2 + (receipts - 828) * 0.7
        
        return base + mile_component + receipt_component
    
    # Test all formulas
    test_formula("Formula 1 (Base + Tiered Miles + Receipts)", formula_1)
    test_formula("Formula 2 (Piecewise by Receipt Threshold)", formula_2)
    test_formula("Formula 3 (Complex Tiered)", formula_3)

def create_lookup_table_approach(df):
    """Create a lookup table for exact matches"""
    print("\n📋 LOOKUP TABLE APPROACH")
    print("=" * 50)
    
    # Create exact lookup for unique combinations
    lookup = {}
    for _, row in df.iterrows():
        key = (row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount'])
        lookup[key] = row['reimbursement_amount']
    
    print(f"Created lookup table with {len(lookup)} exact combinations")
    
    # Check for patterns in the lookup table
    grouped_by_days = defaultdict(list)
    for (days, miles, receipts), reimbursement in lookup.items():
        grouped_by_days[days].append((miles, receipts, reimbursement))
    
    print("\n📊 Patterns by trip duration:")
    for days in sorted(grouped_by_days.keys())[:10]:  # Show first 10 days
        cases = grouped_by_days[days]
        print(f"  {days} days: {len(cases)} cases")
        
        # Show a few examples
        for i, (miles, receipts, reimbursement) in enumerate(sorted(cases)[:3]):
            print(f"    Ex {i+1}: {miles}mi, ${receipts:.2f} → ${reimbursement:.2f}")

def main():
    """Main analysis function"""
    print("🎯 EXACT PATTERN FINDER")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"📊 Analyzing {len(df)} cases for exact patterns")
    
    # Find the most accurate decision tree
    tree, features, mae = find_perfect_tree(df)
    
    # Analyze exact clusters
    analyze_exact_clusters(df)
    
    # Test mathematical formulas
    test_formula_hypotheses(df)
    
    # Create lookup table
    create_lookup_table_approach(df)
    
    print(f"\n🏆 SUMMARY:")
    print(f"Best Decision Tree MAE: ${mae:.6f}")
    print(f"For comparison, your k-NN achieved: $0.000000")
    print(f"Gap to close: ${mae:.6f}")
    
    return tree, mae

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test Generalization vs Overfitting
Compare different approaches to see if simpler rules work
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for case in data:
        row = case['input'].copy()
        row['reimbursement_amount'] = case['expected_output']
        processed_data.append(row)
    
    return pd.DataFrame(processed_data)

def test_different_approaches(df):
    """Test various approaches from simple to complex"""
    print("🧪 TESTING GENERALIZATION vs OVERFITTING")
    print("=" * 60)
    
    # Feature engineering
    df_features = df.copy()
    df_features['miles_low'] = df_features['miles_traveled'].apply(lambda x: min(x, 100))
    df_features['miles_high'] = df_features['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df_features['receipts_per_day'] = df_features['total_receipts_amount'] / df_features['trip_duration_days']
    df_features['miles_per_day'] = df_features['miles_traveled'] / df_features['trip_duration_days']
    
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_low', 'miles_high', 'receipts_per_day', 'miles_per_day'
    ]
    
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    # Split data for true generalization test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Data split: {len(X_train)} training, {len(X_test)} test cases")
    
    # Test different approaches
    approaches = [
        ("Linear Regression", LinearRegression()),
        ("Simple Tree (depth=3)", DecisionTreeRegressor(max_depth=3, random_state=42)),
        ("Medium Tree (depth=6)", DecisionTreeRegressor(max_depth=6, random_state=42)),
        ("Regularized Tree (depth=10)", DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
        ("k-NN (k=1)", KNeighborsRegressor(n_neighbors=1)),
        ("k-NN (k=3)", KNeighborsRegressor(n_neighbors=3)),
        ("Perfect Tree (no limit)", DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42))
    ]
    
    results = []
    
    for name, model in approaches:
        # Fit on training data
        model.fit(X_train, y_train)
        
        # Test on training data (memorization)
        train_pred = model.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - y_train))
        
        # Test on unseen data (generalization)
        test_pred = model.predict(X_test)
        test_mae = np.mean(np.abs(test_pred - y_test))
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -np.mean(cv_scores)
        
        results.append({
            'name': name,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mae': cv_mae,
            'overfitting': test_mae - train_mae
        })
        
        print(f"\n{name}:")
        print(f"  Training MAE: ${train_mae:.2f}")
        print(f"  Test MAE: ${test_mae:.2f}")
        print(f"  CV MAE: ${cv_mae:.2f}")
        print(f"  Overfitting: ${test_mae - train_mae:.2f}")
    
    return results

def analyze_data_complexity(df):
    """Analyze if the data really needs perfect memorization"""
    print("\n🔍 ANALYZING DATA COMPLEXITY")
    print("=" * 60)
    
    # Check for exact duplicates
    duplicates = df.duplicated(subset=['trip_duration_days', 'miles_traveled', 'total_receipts_amount'])
    unique_inputs = len(df) - duplicates.sum()
    
    print(f"📊 Data Analysis:")
    print(f"  Total cases: {len(df)}")
    print(f"  Unique input combinations: {unique_inputs}")
    print(f"  Exact duplicates: {duplicates.sum()}")
    
    # Check for near-duplicates with different outputs
    tolerance = 1.0  # Within $1
    conflicts = 0
    
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            
            # Check if inputs are very similar
            days_diff = abs(row1['trip_duration_days'] - row2['trip_duration_days'])
            miles_diff = abs(row1['miles_traveled'] - row2['miles_traveled'])
            receipts_diff = abs(row1['total_receipts_amount'] - row2['total_receipts_amount'])
            
            if days_diff <= 0 and miles_diff <= 5 and receipts_diff <= 5:
                output_diff = abs(row1['reimbursement_amount'] - row2['reimbursement_amount'])
                if output_diff > tolerance:
                    conflicts += 1
                    if conflicts <= 5:  # Show first 5 examples
                        print(f"  Conflict {conflicts}: Similar inputs, different outputs")
                        print(f"    Case A: {row1['trip_duration_days']}d, {row1['miles_traveled']}mi, ${row1['total_receipts_amount']:.2f} → ${row1['reimbursement_amount']:.2f}")
                        print(f"    Case B: {row2['trip_duration_days']}d, {row2['miles_traveled']}mi, ${row2['total_receipts_amount']:.2f} → ${row2['reimbursement_amount']:.2f}")
    
    print(f"  Similar inputs with different outputs: {conflicts}")
    
    if conflicts > 50:
        print("  ⚠️  High complexity - may require memorization")
    elif conflicts > 10:
        print("  ⚡ Medium complexity - some special cases needed")
    else:
        print("  ✅ Low complexity - simple rules might work")

def main():
    """Main analysis"""
    df = load_data()
    
    # Test different approaches
    results = test_different_approaches(df)
    
    # Analyze data complexity
    analyze_data_complexity(df)
    
    # Summary and recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    best_generalizer = min(results, key=lambda x: x['cv_mae'])
    least_overfitting = min(results, key=lambda x: x['overfitting'])
    
    print(f"Best generalizing approach: {best_generalizer['name']}")
    print(f"  Cross-validation MAE: ${best_generalizer['cv_mae']:.2f}")
    
    print(f"\nLeast overfitting approach: {least_overfitting['name']}")
    print(f"  Overfitting gap: ${least_overfitting['overfitting']:.2f}")
    
    # Determine if our perfect tree is reasonable
    perfect_tree_result = [r for r in results if 'Perfect Tree' in r['name']][0]
    
    if perfect_tree_result['cv_mae'] < 50:
        print(f"\n✅ VERDICT: Perfect tree approach is reasonable")
        print(f"   Cross-validation MAE is still low (${perfect_tree_result['cv_mae']:.2f})")
        print(f"   The legacy system likely IS this complex")
    else:
        print(f"\n❌ VERDICT: Perfect tree is overfitting")
        print(f"   Cross-validation MAE is high (${perfect_tree_result['cv_mae']:.2f})")
        print(f"   Consider simpler approaches")

if __name__ == "__main__":
    main() 
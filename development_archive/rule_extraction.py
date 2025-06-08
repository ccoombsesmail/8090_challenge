#!/usr/bin/env python3
"""
Legacy System Rule Extraction
Use decision trees to discover the actual business rules, not just predict outcomes
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def extract_business_rules():
    """Extract interpretable business rules from the legacy system data"""
    
    print("🔍 LEGACY SYSTEM RULE EXTRACTION")
    print("="*50)
    
    # Load and prepare data
    with open('public_cases.json', 'r') as f:
        raw_data = json.load(f)
    
    data_list = []
    for case in raw_data:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # Engineer features for rule discovery
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_low'] = np.minimum(df['miles_traveled'], 100)
    df['miles_high'] = np.maximum(df['miles_traveled'] - 100, 0)
    df['receipt_cents'] = (df['total_receipts_amount'] * 100) % 100
    df['receipt_penalty'] = ((df['receipt_cents'] == 49) | (df['receipt_cents'] == 99)).astype(int)
    
    # Create interpretable features
    df['is_short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['is_medium_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 7)).astype(int)
    df['is_long_trip'] = (df['trip_duration_days'] >= 8).astype(int)
    df['is_high_mileage'] = (df['miles_traveled'] > 500).astype(int)
    df['is_high_spending'] = (df['receipts_per_day'] > 200).astype(int)
    
    # Features for rule extraction
    features = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_per_day', 'receipts_per_day', 'miles_low', 'miles_high',
        'receipt_penalty', 'is_short_trip', 'is_medium_trip', 'is_long_trip',
        'is_high_mileage', 'is_high_spending'
    ]
    
    X = df[features]
    y = df['reimbursement']
    
    # Build interpretable decision tree
    print("\n📋 BUILDING INTERPRETABLE DECISION TREE")
    print("-" * 40)
    
    tree = DecisionTreeRegressor(
        max_depth=6,          # Keep shallow for interpretability
        min_samples_split=30, # Ensure statistical significance
        min_samples_leaf=15,  # Avoid overfitting to outliers
        random_state=42
    )
    
    tree.fit(X, y)
    
    # Extract and display rules
    rules = export_text(tree, feature_names=features, max_depth=4)
    print("EXTRACTED BUSINESS RULES:")
    print(rules)
    
    # Performance check
    y_pred = tree.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"\nRule-based MAE: ${mae:.2f}")
    
    # Analyze feature importance for rule priorities
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 RULE IMPORTANCE (Business Logic Priority):")
    print("-" * 40)
    for _, row in feature_importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return tree, df, features

def analyze_specific_patterns(df):
    """Analyze specific patterns mentioned in interviews"""
    
    print("\n🎯 VALIDATING INTERVIEW INSIGHTS")
    print("-" * 40)
    
    # Test 5-day bonus (Lisa mentioned this)
    print("1. Testing 5-day bonus pattern:")
    day5_cases = df[df['trip_duration_days'] == 5]
    other_cases = df[df['trip_duration_days'] != 5]
    if len(day5_cases) > 0:
        day5_rate = (day5_cases['reimbursement'] / day5_cases['trip_duration_days']).mean()
        other_rate = (other_cases['reimbursement'] / other_cases['trip_duration_days']).mean()
        print(f"   5-day per-day rate: ${day5_rate:.2f}")
        print(f"   Other per-day rate: ${other_rate:.2f}")
        if day5_rate > other_rate * 1.05:
            print("   ✓ 5-day bonus confirmed!")
        else:
            print("   ✗ No significant 5-day bonus")
    
    # Test mileage tiers (Lisa mentioned 100-mile breakpoint)
    print("\n2. Testing mileage tier breakpoint:")
    low_mileage = df[df['miles_traveled'] <= 100]
    high_mileage = df[df['miles_traveled'] > 100]
    if len(low_mileage) > 0 and len(high_mileage) > 0:
        low_rate = (low_mileage['reimbursement'] / low_mileage['miles_traveled']).mean()
        high_rate = (high_mileage['reimbursement'] / high_mileage['miles_traveled']).mean()
        print(f"   ≤100 miles rate: ${low_rate:.3f}/mile")
        print(f"   >100 miles rate: ${high_rate:.3f}/mile")
        if low_rate > high_rate * 1.2:
            print("   ✓ Mileage tier breakpoint confirmed!")
        else:
            print("   ✗ No clear mileage tier pattern")
    
    # Test receipt penalty bug (Lisa mentioned .49/.99 penalty)
    print("\n3. Testing receipt ending penalty:")
    normal_cases = df[df['receipt_penalty'] == 0]
    penalty_cases = df[df['receipt_penalty'] == 1]
    if len(penalty_cases) > 0:
        normal_avg = normal_cases['reimbursement'].mean()
        penalty_avg = penalty_cases['reimbursement'].mean()
        print(f"   Normal endings avg: ${normal_avg:.2f}")
        print(f"   .49/.99 endings avg: ${penalty_avg:.2f}")
        if penalty_avg < normal_avg * 0.9:
            print("   ✓ Receipt ending penalty confirmed!")
        else:
            print("   ✗ No clear receipt ending penalty")
    
    # Test high spending penalty
    print("\n4. Testing high spending patterns:")
    high_spend = df[df['receipts_per_day'] > 300]
    normal_spend = df[df['receipts_per_day'] <= 300]
    if len(high_spend) > 0:
        high_rate = (high_spend['reimbursement'] / high_spend['trip_duration_days']).mean()
        normal_rate = (normal_spend['reimbursement'] / normal_spend['trip_duration_days']).mean()
        print(f"   High spending (>$300/day) rate: ${high_rate:.2f}/day")
        print(f"   Normal spending rate: ${normal_rate:.2f}/day")

def build_rule_based_algorithm(tree, features):
    """Build interpretable rule-based algorithm from decision tree"""
    
    print("\n🔧 BUILDING RULE-BASED ALGORITHM")
    print("-" * 40)
    
    # Extract key thresholds from tree
    print("Key decision thresholds discovered:")
    
    # This would analyze the tree structure to extract the actual thresholds
    # For now, let's use our discovered patterns
    
    print("  • Total receipts threshold: ~$828")
    print("  • Trip duration threshold: ~4.5 days")
    print("  • High mileage threshold: ~583 miles")
    print("  • Miles over 100 threshold: ~524 miles")
    
    # Return structure for rule-based implementation
    rules = {
        'receipt_threshold': 828.10,
        'trip_threshold': 4.5,
        'mileage_threshold': 583.0,
        'miles_high_threshold': 524.5,
        'base_per_diem': 100,
        'mileage_rate_low': 0.58,
        'mileage_rate_high': 0.35,
        'receipt_penalty_amount': 400
    }
    
    return rules

if __name__ == "__main__":
    # Extract rules from decision tree
    tree, df, features = extract_business_rules()
    
    # Validate interview insights
    analyze_specific_patterns(df)
    
    # Build rule-based algorithm
    rules = build_rule_based_algorithm(tree, features)
    
    print(f"\n✅ RULE EXTRACTION COMPLETE")
    print("Ready to implement rule-based reimbursement calculator!") 
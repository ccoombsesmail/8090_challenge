#!/usr/bin/env python3
"""
Analyze Remaining Outliers - What patterns are we still missing?
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

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

def engineer_enhanced_features(df):
    """Engineer the same enhanced features as the improved model"""
    
    df = df.copy()
    
    # Basic features
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df['receipt_ends_49_99'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    # Outlier-specific features
    df['receipt_mile_ratio'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    df['high_receipt_mile_ratio'] = (df['receipt_mile_ratio'] > 2).astype(int)
    df['extreme_receipt_mile_ratio'] = (df['receipt_mile_ratio'] > 10).astype(int)
    df['is_8_day_trip'] = (df['trip_duration_days'] == 8).astype(int)
    df['very_high_receipts'] = (df['total_receipts_amount'] > 2000).astype(int)
    df['high_receipts'] = (df['total_receipts_amount'] > 1500).astype(int)
    df['long_trip'] = (df['trip_duration_days'] >= 10).astype(int)
    df['very_long_trip'] = (df['trip_duration_days'] >= 14).astype(int)
    df['very_low_efficiency'] = (df['miles_per_day'] < 30).astype(int)
    df['high_efficiency'] = (df['miles_per_day'] > 300).astype(int)
    df['short_high_expense'] = ((df['trip_duration_days'] <= 3) & 
                               (df['total_receipts_amount'] > 1000)).astype(int)
    df['city_travel_pattern'] = ((df['receipt_mile_ratio'] > 1.5) & 
                                (df['miles_per_day'] < 100)).astype(int)
    df['days_x_receipt_ratio'] = df['trip_duration_days'] * df['receipt_mile_ratio']
    df['efficiency_x_receipts'] = df['miles_per_day'] * df['total_receipts_amount']
    
    return df

def train_improved_model(df):
    """Train the improved model"""
    
    df_features = engineer_enhanced_features(df)
    
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'receipts_per_day', 'miles_per_day', 'miles_low', 'miles_high',
        'receipt_ends_49_99', 'receipt_mile_ratio', 'high_receipt_mile_ratio', 
        'extreme_receipt_mile_ratio', 'is_8_day_trip', 'very_high_receipts', 
        'high_receipts', 'long_trip', 'very_long_trip', 'very_low_efficiency', 
        'high_efficiency', 'short_high_expense', 'city_travel_pattern',
        'days_x_receipt_ratio', 'efficiency_x_receipts'
    ]
    
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5, 
        min_samples_leaf=3, random_state=42
    )
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    return predictions, model, df_features

def analyze_remaining_worst_cases(df, predictions, top_n=10):
    """Analyze the worst remaining cases"""
    
    df = df.copy()
    df['predicted'] = predictions
    df['error'] = abs(df['predicted'] - df['reimbursement_amount'])
    
    worst_cases = df.nlargest(top_n, 'error')
    
    print(f"🔥 REMAINING TOP {top_n} WORST CASES")
    print("=" * 80)
    print(f"Max Error: ${worst_cases['error'].max():.2f}")
    print(f"Median Error in Worst Cases: ${worst_cases['error'].median():.2f}")
    
    print(f"\nWORST {top_n} REMAINING CASES:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(worst_cases.iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        predicted = row['predicted']
        error = row['error']
        error_pct = (error / actual) * 100
        
        ratio = receipts / max(miles, 1)
        efficiency = miles / days
        spending_per_day = receipts / days
        
        print(f"{i:2d}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f} → Expected: ${actual:7.2f}, Got: ${predicted:7.2f}")
        print(f"    Error: ${error:6.2f} ({error_pct:5.1f}%) | Ratio: {ratio:5.2f} | {efficiency:5.1f}mi/d | ${spending_per_day:6.2f}/d")
        print()
    
    return worst_cases

def deep_pattern_analysis(worst_cases):
    """Deep analysis of remaining patterns"""
    
    print(f"🧐 DEEP PATTERN ANALYSIS OF REMAINING OUTLIERS")
    print("=" * 60)
    
    # Look for very specific patterns
    patterns_found = []
    
    for _, row in worst_cases.iterrows():
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        predicted = row['predicted']
        error = row['error']
        
        ratio = receipts / max(miles, 1)
        efficiency = miles / days
        spending_per_day = receipts / days
        
        patterns = []
        
        # Very specific combinations
        if days == 8 and ratio > 1.5:
            patterns.append("8-DAY + HIGH RATIO")
        
        if days == 4 and receipts > 2000:
            patterns.append("4-DAY + VERY HIGH RECEIPTS")
        
        if efficiency < 20 and receipts > 2000:
            patterns.append("VERY LOW EFFICIENCY + VERY HIGH RECEIPTS")
        
        if days <= 4 and receipts > 1500 and miles < 100:
            patterns.append("SHORT + HIGH RECEIPTS + LOW MILES")
        
        if actual < 500 and predicted > 1000:
            patterns.append("MAJOR OVER-PREDICTION")
        
        if days >= 8 and actual < 700 and predicted > 1200:
            patterns.append("LONG TRIP OVER-PREDICTION")
        
        if receipts > 1400 and actual < 650:
            patterns.append("HIGH RECEIPTS BUT LOW REIMBURSEMENT")
        
        # Receipt ending patterns
        receipt_ending = int((receipts * 100) % 100)
        if receipt_ending in [49, 99]:
            patterns.append(f"RECEIPT ENDS .{receipt_ending:02d}")
        
        # Add to overall patterns
        for pattern in patterns:
            patterns_found.append({
                'pattern': pattern,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'actual': actual,
                'predicted': predicted,
                'error': error
            })
    
    # Summarize patterns
    if patterns_found:
        pattern_summary = {}
        for p in patterns_found:
            pattern = p['pattern']
            if pattern not in pattern_summary:
                pattern_summary[pattern] = []
            pattern_summary[pattern].append(p['error'])
        
        print("SPECIFIC PATTERNS FOUND:")
        for pattern, errors in pattern_summary.items():
            avg_error = np.mean(errors)
            count = len(errors)
            print(f"  {pattern:35s}: {count} cases, avg error ${avg_error:6.2f}")
    
    # Look for mathematical relationships
    print(f"\nMATHEMATICAL RELATIONSHIP ANALYSIS:")
    
    # Check if there are specific ratios or formulas
    for i, (_, row) in enumerate(worst_cases.head(5).iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        predicted = row['predicted']
        
        # Try to find mathematical relationships
        if actual > 0:
            actual_per_day = actual / days
            actual_per_mile = actual / max(miles, 1) if miles > 0 else 0
            actual_per_receipt = actual / receipts if receipts > 0 else 0
            
            print(f"Case {i}: Actual=${actual:.2f} | ${actual_per_day:.2f}/day | ${actual_per_mile:.3f}/mile | {actual_per_receipt:.3f}/receipt")

def suggest_next_improvements(worst_cases):
    """Suggest what to try next"""
    
    print(f"\n💡 SUGGESTED NEXT IMPROVEMENTS")
    print("=" * 50)
    
    # Analyze the worst cases for new feature ideas
    improvements = []
    
    # Check for 8-day + high ratio pattern
    eight_day_high_ratio = worst_cases[(worst_cases['trip_duration_days'] == 8) & 
                                      (worst_cases['receipt_mile_ratio'] > 1.5)]
    if len(eight_day_high_ratio) > 0:
        improvements.append("1. ADD SPECIFIC 8-DAY + HIGH-RATIO INTERACTION TERM")
    
    # Check for very high receipts with low reimbursement
    high_receipts_low_reimb = worst_cases[(worst_cases['total_receipts_amount'] > 1400) & 
                                         (worst_cases['reimbursement_amount'] < 650)]
    if len(high_receipts_low_reimb) > 0:
        improvements.append("2. ADD PENALTY FOR HIGH RECEIPTS WITH EXPECTED LOW REIMBURSEMENT")
    
    # Check for extreme over-predictions
    over_predictions = worst_cases[worst_cases['predicted'] > worst_cases['reimbursement_amount'] * 1.5]
    if len(over_predictions) > 0:
        improvements.append("3. ADD OVER-PREDICTION PENALTY MECHANISM")
    
    # Check for short high-expense trips
    short_expensive = worst_cases[(worst_cases['trip_duration_days'] <= 4) & 
                                 (worst_cases['total_receipts_amount'] > 1500)]
    if len(short_expensive) > 0:
        improvements.append("4. REFINE SHORT HIGH-EXPENSE TRIP RULES")
    
    if improvements:
        for improvement in improvements:
            print(f"   {improvement}")
    else:
        print("   No obvious systematic patterns remaining")
    
    print(f"\n🎯 POTENTIAL ADVANCED TECHNIQUES:")
    print(f"   • Ensemble multiple models with different strengths")
    print(f"   • Add polynomial features for complex interactions")
    print(f"   • Use gradient boosting for better outlier handling")
    print(f"   • Create separate models for different trip types")

def main():
    """Main analysis"""
    
    print("🔍 ANALYZING REMAINING OUTLIERS AFTER IMPROVEMENT")
    print("=" * 70)
    
    # Load data and train improved model
    df = load_data()
    predictions, model, df_features = train_improved_model(df)
    
    # Current performance
    mae = np.mean(np.abs(predictions - df['reimbursement_amount']))
    max_error = np.max(np.abs(predictions - df['reimbursement_amount']))
    print(f"Current Model Performance: MAE ${mae:.2f}, Max Error ${max_error:.2f}")
    print()
    
    # Analyze worst remaining cases
    worst_cases = analyze_remaining_worst_cases(df_features, predictions, top_n=10)
    
    # Deep pattern analysis
    deep_pattern_analysis(worst_cases)
    
    # Suggest improvements
    suggest_next_improvements(worst_cases)
    
    return worst_cases

if __name__ == "__main__":
    remaining_outliers = main() 
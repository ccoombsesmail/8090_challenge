#!/usr/bin/env python3
"""
Outlier Analysis - Find Missing Business Rules
Analyzes cases with large errors to identify patterns we missed
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

def train_current_model(df):
    """Train our current model to get predictions"""
    
    # Engineer features exactly like in our production system
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    df['receipt_ends_49_99'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'receipts_per_day', 'miles_per_day', 'miles_low', 'miles_high',
        'receipt_ends_49_99'
    ]
    
    X = df[feature_cols]
    y = df['reimbursement_amount']
    
    tree = DecisionTreeRegressor(
        max_depth=12, 
        min_samples_split=10, 
        min_samples_leaf=5,
        random_state=42
    )
    
    tree.fit(X, y)
    predictions = tree.predict(X)
    
    return predictions, tree, feature_cols

def analyze_worst_cases(df, predictions, top_n=20):
    """Analyze the cases with the largest errors"""
    
    # Calculate errors
    df = df.copy()
    df['predicted'] = predictions
    df['error'] = abs(df['predicted'] - df['reimbursement_amount'])
    df['error_pct'] = (df['error'] / df['reimbursement_amount']) * 100
    
    # Sort by error size
    worst_cases = df.nlargest(top_n, 'error')
    
    print(f"🔍 ANALYZING TOP {top_n} WORST CASES")
    print("=" * 80)
    print(f"Max Error: ${worst_cases['error'].max():.2f}")
    print(f"Median Error in Worst Cases: ${worst_cases['error'].median():.2f}")
    
    print(f"\nWORST {top_n} CASES:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(worst_cases.iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        predicted = row['predicted']
        error = row['error']
        error_pct = row['error_pct']
        
        print(f"{i:2d}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f} → Expected: ${actual:7.2f}, Got: ${predicted:7.2f}, Error: ${error:6.2f} ({error_pct:5.1f}%)")
    
    return worst_cases

def find_outlier_patterns(worst_cases):
    """Look for patterns in the worst cases"""
    
    print(f"\n📊 OUTLIER PATTERN ANALYSIS")
    print("=" * 50)
    
    # Duration analysis
    print(f"DURATION PATTERNS:")
    duration_stats = worst_cases.groupby('trip_duration_days').agg({
        'error': ['count', 'mean', 'max'],
        'reimbursement_amount': 'mean',
        'total_receipts_amount': 'mean',
        'miles_traveled': 'mean'
    }).round(2)
    
    for duration in sorted(worst_cases['trip_duration_days'].unique()):
        cases = worst_cases[worst_cases['trip_duration_days'] == duration]
        if len(cases) > 0:
            avg_error = cases['error'].mean()
            count = len(cases)
            avg_reimbursement = cases['reimbursement_amount'].mean()
            print(f"  {duration:2d} days: {count:2d} cases, avg error ${avg_error:6.2f}, avg reimbursement ${avg_reimbursement:7.2f}")
    
    # Receipt amount analysis
    print(f"\nRECEIPT AMOUNT PATTERNS:")
    receipt_bins = [0, 100, 500, 828, 1000, 1500, 2000, float('inf')]
    receipt_labels = ['$0-100', '$100-500', '$500-828', '$828-1000', '$1000-1500', '$1500-2000', '$2000+']
    
    worst_cases['receipt_category'] = pd.cut(worst_cases['total_receipts_amount'], 
                                           bins=receipt_bins, labels=receipt_labels)
    
    for category in receipt_labels:
        cases = worst_cases[worst_cases['receipt_category'] == category]
        if len(cases) > 0:
            avg_error = cases['error'].mean()
            count = len(cases)
            print(f"  {category:>12}: {count:2d} cases, avg error ${avg_error:6.2f}")
    
    # Miles analysis
    print(f"\nMILEAGE PATTERNS:")
    mile_bins = [0, 100, 300, 500, 800, 1200, float('inf')]
    mile_labels = ['0-100', '100-300', '300-500', '500-800', '800-1200', '1200+']
    
    worst_cases['mile_category'] = pd.cut(worst_cases['miles_traveled'], 
                                        bins=mile_bins, labels=mile_labels)
    
    for category in mile_labels:
        cases = worst_cases[worst_cases['mile_category'] == category]
        if len(cases) > 0:
            avg_error = cases['error'].mean()
            count = len(cases)
            print(f"  {category:>8} mi: {count:2d} cases, avg error ${avg_error:6.2f}")
    
    # Efficiency analysis
    print(f"\nEFFICIENCY PATTERNS:")
    worst_cases['miles_per_day'] = worst_cases['miles_traveled'] / worst_cases['trip_duration_days']
    worst_cases['receipts_per_day'] = worst_cases['total_receipts_amount'] / worst_cases['trip_duration_days']
    
    efficiency_bins = [0, 50, 150, 250, 400, float('inf')]
    efficiency_labels = ['0-50', '50-150', '150-250', '250-400', '400+']
    
    worst_cases['efficiency_category'] = pd.cut(worst_cases['miles_per_day'], 
                                              bins=efficiency_bins, labels=efficiency_labels)
    
    for category in efficiency_labels:
        cases = worst_cases[worst_cases['efficiency_category'] == category]
        if len(cases) > 0:
            avg_error = cases['error'].mean()
            count = len(cases)
            print(f"  {category:>8} mi/day: {count:2d} cases, avg error ${avg_error:6.2f}")

def identify_missing_rules(worst_cases):
    """Try to identify what business rules we might be missing"""
    
    print(f"\n🧩 MISSING BUSINESS RULES ANALYSIS")
    print("=" * 60)
    
    # Look for extreme cases
    extreme_cases = worst_cases.head(5)
    
    print("ANALYZING TOP 5 WORST CASES FOR MISSING RULES:")
    
    for i, (_, row) in enumerate(extreme_cases.iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        predicted = row['predicted']
        error = row['error']
        
        miles_per_day = miles / days
        receipts_per_day = receipts / days
        
        print(f"\nCASE {i}: {days}d, {miles}mi, ${receipts:.2f}")
        print(f"  Expected: ${actual:.2f}, Got: ${predicted:.2f}, Error: ${error:.2f}")
        print(f"  Efficiency: {miles_per_day:.1f} mi/day, Spending: ${receipts_per_day:.2f}/day")
        
        # Look for potential missing rules
        potential_rules = []
        
        if days >= 10 and receipts < 200:
            potential_rules.append("LONG TRIP + LOW RECEIPTS: Special handling?")
        
        if days == 1 and receipts > 1000:
            potential_rules.append("ONE DAY + HIGH RECEIPTS: Same-day return rule?")
        
        if miles_per_day > 500:
            potential_rules.append("EXTREME EFFICIENCY: Emergency travel rule?")
        
        if miles == 0 and days > 1:
            potential_rules.append("NO TRAVEL + MULTI-DAY: Conference/meeting rule?")
        
        if receipts > 2000:
            potential_rules.append("VERY HIGH RECEIPTS: Executive/special rate?")
        
        if days <= 3 and miles > 1000:
            potential_rules.append("SHORT + HIGH MILES: Rush travel rule?")
        
        if receipts_per_day > 500:
            potential_rules.append("HIGH DAILY SPENDING: Special category?")
        
        if miles > 0 and receipts / miles > 2:
            potential_rules.append("HIGH RECEIPT-TO-MILE RATIO: City/expensive area rule?")
        
        if potential_rules:
            for rule in potential_rules:
                print(f"    → {rule}")
        else:
            print("    → No obvious special patterns detected")

def suggest_improvements(worst_cases):
    """Suggest specific improvements to our model"""
    
    print(f"\n💡 SUGGESTED IMPROVEMENTS")
    print("=" * 50)
    
    # Analyze what features might help
    improvements = []
    
    # Check for extreme ratios
    worst_cases['receipt_mile_ratio'] = worst_cases['total_receipts_amount'] / np.maximum(worst_cases['miles_traveled'], 1)
    
    high_ratio_cases = worst_cases[worst_cases['receipt_mile_ratio'] > 2]
    if len(high_ratio_cases) > 0:
        improvements.append(f"1. ADD RECEIPT-TO-MILE RATIO FEATURE: {len(high_ratio_cases)} high-error cases have ratio > 2")
    
    # Check for duration-specific patterns
    long_trip_errors = worst_cases[worst_cases['trip_duration_days'] >= 10]
    if len(long_trip_errors) > 0:
        avg_error = long_trip_errors['error'].mean()
        improvements.append(f"2. IMPROVE LONG TRIP RULES: {len(long_trip_errors)} cases ≥10 days, avg error ${avg_error:.2f}")
    
    # Check for very high receipts
    high_receipt_errors = worst_cases[worst_cases['total_receipts_amount'] > 1500]
    if len(high_receipt_errors) > 0:
        avg_error = high_receipt_errors['error'].mean()
        improvements.append(f"3. ADD HIGH-RECEIPT RULES: {len(high_receipt_errors)} cases >$1500, avg error ${avg_error:.2f}")
    
    # Check for extreme efficiency
    extreme_efficiency = worst_cases[worst_cases['miles_per_day'] > 300]
    if len(extreme_efficiency) > 0:
        avg_error = extreme_efficiency['error'].mean()
        improvements.append(f"4. REFINE EFFICIENCY RULES: {len(extreme_efficiency)} cases >300 mi/day, avg error ${avg_error:.2f}")
    
    # Output suggestions
    if improvements:
        for improvement in improvements:
            print(f"   {improvement}")
    else:
        print("   No obvious systematic improvements identified")
    
    print(f"\n📈 POTENTIAL QUICK WINS:")
    print(f"   • Add more granular receipt amount thresholds")
    print(f"   • Implement separate rules for extreme cases (>10 days, >$2000, >500 mi/day)")
    print(f"   • Consider receipt-to-mile ratio as a feature")
    print(f"   • Add special handling for same-day trips with high receipts")

def main():
    """Main analysis"""
    
    print("🔍 OUTLIER ANALYSIS - FINDING MISSING BUSINESS RULES")
    print("=" * 70)
    
    # Load data and train model
    df = load_data()
    predictions, tree, feature_cols = train_current_model(df)
    
    print(f"Current Model Performance:")
    mae = np.mean(np.abs(predictions - df['reimbursement_amount']))
    print(f"  MAE: ${mae:.2f}")
    print(f"  Max Error: ${np.max(np.abs(predictions - df['reimbursement_amount'])):.2f}")
    
    # Analyze worst cases
    worst_cases = analyze_worst_cases(df, predictions, top_n=20)
    
    # Find patterns
    find_outlier_patterns(worst_cases)
    
    # Identify missing rules
    identify_missing_rules(worst_cases)
    
    # Suggest improvements
    suggest_improvements(worst_cases)
    
    return worst_cases

if __name__ == "__main__":
    outliers = main() 
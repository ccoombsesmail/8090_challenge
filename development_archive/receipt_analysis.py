#!/usr/bin/env python3
"""
Focused Receipt Analysis to understand the exact reimbursement patterns
"""

import json
import pandas as pd
import numpy as np

def analyze_receipt_patterns():
    """Analyze specific receipt patterns to improve our formula"""
    
    # Load test cases
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Convert to DataFrame for analysis
    data = []
    for case in test_cases:
        inputs = case['input']
        data.append({
            'days': inputs['trip_duration_days'],
            'miles': inputs['miles_traveled'],
            'receipts': inputs['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['receipts_per_day'] = df['receipts'] / df['days']
    df['per_day_rate'] = df['reimbursement'] / df['days']
    
    # Estimate base components by looking at zero/low receipt cases
    low_receipt_cases = df[df['receipts'] <= 50].copy()
    print("=== LOW RECEIPT CASES (≤$50) ===")
    print(f"Count: {len(low_receipt_cases)}")
    if len(low_receipt_cases) > 0:
        print("\nSample cases:")
        for i, row in low_receipt_cases.head(10).iterrows():
            estimated_base = row['reimbursement'] - (0.5 * min(row['miles'], 100)) - (0.3 * max(row['miles'] - 100, 0))
            estimated_per_day = estimated_base / row['days']
            print(f"  {row['days']} days, {row['miles']} miles, ${row['receipts']:.2f} receipts → ${row['reimbursement']:.2f}")
            print(f"    Est. base: ${estimated_base:.2f}, per day: ${estimated_per_day:.2f}")
    
    print("\n=== RECEIPT RATE ANALYSIS ===")
    
    # Try to isolate receipt component by subtracting estimated base + mileage
    def estimate_receipt_component(row):
        # Base per diem (conservative estimate)
        if row['days'] == 1:
            base = 100
        elif row['days'] <= 3:
            base = 90 * row['days']
        elif row['days'] <= 7:
            base = 80 * row['days']
        else:
            base = 70 * row['days']
        
        # Mileage component
        mileage = 0.5 * min(row['miles'], 100) + 0.3 * max(row['miles'] - 100, 0)
        
        # Receipt component (what's left)
        receipt_component = row['reimbursement'] - base - mileage
        return receipt_component
    
    df['estimated_receipt_component'] = df.apply(estimate_receipt_component, axis=1)
    df['receipt_rate'] = np.where(df['receipts'] > 0, df['estimated_receipt_component'] / df['receipts'], 0)
    
    # Analyze receipt rates by amount ranges
    receipt_ranges = [
        (0, 100, "0-100"),
        (100, 300, "100-300"), 
        (300, 600, "300-600"),
        (600, 1000, "600-1000"),
        (1000, 1500, "1000-1500"),
        (1500, 2500, "1500-2500")
    ]
    
    print("\nReceipt Rate Analysis:")
    for min_amt, max_amt, label in receipt_ranges:
        subset = df[(df['receipts'] >= min_amt) & (df['receipts'] < max_amt)]
        if len(subset) > 0:
            avg_rate = subset['receipt_rate'].mean()
            median_rate = subset['receipt_rate'].median()
            count = len(subset)
            print(f"  ${label}: {count:3d} cases, avg rate: {avg_rate:5.2f}, median: {median_rate:5.2f}")
    
    print("\n=== SPECIFIC ERROR CASES ANALYSIS ===")
    
    # Analyze specific high-error cases from our results
    problem_cases = [
        (4, 69, 2321.49, 322.00),  # Case 152
        (8, 795, 1645.99, 644.69), # Case 684  
        (1, 1082, 1809.49, 446.94), # Case 996
        (5, 516, 1878.49, 669.85), # Case 711
    ]
    
    print("High-error cases from our evaluation:")
    for days, miles, receipts, expected in problem_cases:
        # Find matching case in data
        matching = df[(df['days'] == days) & (df['miles'] == miles) & (abs(df['receipts'] - receipts) < 0.01)]
        if len(matching) > 0:
            row = matching.iloc[0]
            receipt_comp = row['estimated_receipt_component']
            implied_rate = receipt_comp / receipts if receipts > 0 else 0
            print(f"  {days} days, {miles} miles, ${receipts:.2f} receipts → ${expected:.2f}")
            print(f"    Est. receipt component: ${receipt_comp:.2f}, implied rate: {implied_rate:.3f}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("Based on analysis, suggested receipt rates:")
    
    # Calculate average rates for different ranges
    for min_amt, max_amt, label in receipt_ranges:
        subset = df[(df['receipts'] >= min_amt) & (df['receipts'] < max_amt)]
        if len(subset) > 5:  # Only ranges with sufficient data
            rates = subset['receipt_rate'].dropna()
            rates_cleaned = rates[(rates >= 0) & (rates <= 2)]  # Remove outliers
            if len(rates_cleaned) > 0:
                avg_rate = rates_cleaned.mean()
                print(f"  ${label}: use rate {avg_rate:.2f}")

if __name__ == "__main__":
    analyze_receipt_patterns() 
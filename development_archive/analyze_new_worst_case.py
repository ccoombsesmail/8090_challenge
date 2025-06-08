#!/usr/bin/env python3
"""
Analyze New Worst Case - Understanding the 5d, 41mi, $2315 case
"""

import json
import pandas as pd
import numpy as np

def analyze_high_reimbursement_cases():
    """Find cases with very high expected reimbursement to understand the pattern"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    cases = []
    for case in data:
        row = case['input'].copy()
        row['reimbursement_amount'] = case['expected_output']
        cases.append(row)
    
    df = pd.DataFrame(cases)
    
    # Calculate additional metrics
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['receipt_mile_ratio'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    
    print("🔍 ANALYZING HIGH REIMBURSEMENT CASES")
    print("=" * 60)
    
    # Find cases with high expected reimbursement
    high_reimb = df[df['reimbursement_amount'] > 1200].sort_values('reimbursement_amount', ascending=False)
    
    print(f"Cases with Reimbursement > $1200: {len(high_reimb)}")
    print("\nTOP 10 HIGHEST REIMBURSEMENT CASES:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(high_reimb.head(10).iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        reimbursement = row['reimbursement_amount']
        ratio = row['receipt_mile_ratio']
        efficiency = row['miles_per_day']
        spending = row['receipts_per_day']
        
        print(f"{i:2d}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f} → ${reimbursement:7.2f}")
        print(f"    Ratio: {ratio:5.2f} | {efficiency:5.1f}mi/d | ${spending:6.2f}/d")
        print()
    
    return high_reimb

def analyze_specific_worst_case():
    """Analyze the specific new worst case: 5d, 41mi, $2314.68 → $1500.28"""
    
    print("🎯 ANALYZING NEW WORST CASE")
    print("=" * 40)
    
    # The problematic case
    days = 5
    miles = 41
    receipts = 2314.68
    expected = 1500.28
    
    print(f"Case: {days}d, {miles}mi, ${receipts:.2f} → Expected: ${expected:.2f}")
    
    # Calculate metrics
    miles_per_day = miles / days
    receipts_per_day = receipts / days
    receipt_mile_ratio = receipts / max(miles, 1)
    
    print(f"Metrics:")
    print(f"  Miles per day: {miles_per_day:.1f}")
    print(f"  Receipts per day: ${receipts_per_day:.2f}")
    print(f"  Receipt-to-mile ratio: {receipt_mile_ratio:.2f}")
    
    # Find similar cases
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    similar_cases = []
    for case in data:
        case_days = case['input']['trip_duration_days']
        case_miles = case['input']['miles_traveled']
        case_receipts = case['input']['total_receipts_amount']
        case_expected = case['expected_output']
        
        case_ratio = case_receipts / max(case_miles, 1)
        case_efficiency = case_miles / case_days
        
        # Find cases with similar patterns
        if (abs(case_ratio - receipt_mile_ratio) < 20 and  # Similar ratio
            abs(case_efficiency - miles_per_day) < 5 and    # Similar efficiency
            case_receipts > 2000):                          # High receipts
            
            similar_cases.append({
                'days': case_days,
                'miles': case_miles,
                'receipts': case_receipts,
                'expected': case_expected,
                'ratio': case_ratio,
                'efficiency': case_efficiency
            })
    
    print(f"\nSIMILAR HIGH-RECEIPT, LOW-MILE CASES:")
    print("-" * 50)
    
    if similar_cases:
        for i, case in enumerate(similar_cases[:5], 1):
            print(f"{i}. {case['days']}d, {case['miles']}mi, ${case['receipts']:.2f} → ${case['expected']:.2f}")
            print(f"   Ratio: {case['ratio']:.2f}, Efficiency: {case['efficiency']:.1f}mi/d")
    else:
        print("No very similar cases found")
    
    # Look for high receipt, low mile cases with high reimbursement
    high_receipt_low_mile = []
    for case in data:
        case_days = case['input']['trip_duration_days']
        case_miles = case['input']['miles_traveled']
        case_receipts = case['input']['total_receipts_amount']
        case_expected = case['expected_output']
        
        case_ratio = case_receipts / max(case_miles, 1)
        
        if (case_receipts > 2000 and     # High receipts
            case_miles < 100 and         # Low miles
            case_expected > 1200):       # High reimbursement
            
            high_receipt_low_mile.append({
                'days': case_days,
                'miles': case_miles,
                'receipts': case_receipts,
                'expected': case_expected,
                'ratio': case_ratio
            })
    
    print(f"\nHIGH RECEIPT + LOW MILE + HIGH REIMBURSEMENT CASES:")
    print("-" * 60)
    
    if high_receipt_low_mile:
        for i, case in enumerate(high_receipt_low_mile[:5], 1):
            print(f"{i}. {case['days']}d, {case['miles']}mi, ${case['receipts']:.2f} → ${case['expected']:.2f} (ratio: {case['ratio']:.1f})")
    else:
        print("No cases found")

def analyze_penalty_logic():
    """Analyze when penalties should and shouldn't apply"""
    
    print(f"\n🧠 PENALTY LOGIC ANALYSIS")
    print("=" * 40)
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Categorize cases by receipt-to-mile ratio and expected reimbursement
    categories = {
        'high_ratio_high_reimb': [],   # High ratio but HIGH reimbursement (DON'T penalize)
        'high_ratio_low_reimb': [],    # High ratio and LOW reimbursement (DO penalize)
        'med_ratio_high_reimb': [],    # Medium ratio, high reimbursement
        'med_ratio_low_reimb': []      # Medium ratio, low reimbursement
    }
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        ratio = receipts / max(miles, 1)
        
        if ratio > 15:  # High ratio
            if expected > 1000:
                categories['high_ratio_high_reimb'].append((days, miles, receipts, expected, ratio))
            else:
                categories['high_ratio_low_reimb'].append((days, miles, receipts, expected, ratio))
        elif ratio > 3:  # Medium ratio
            if expected > 1000:
                categories['med_ratio_high_reimb'].append((days, miles, receipts, expected, ratio))
            else:
                categories['med_ratio_low_reimb'].append((days, miles, receipts, expected, ratio))
    
    for category, cases in categories.items():
        if cases:
            print(f"\n{category.upper()}: {len(cases)} cases")
            for i, (days, miles, receipts, expected, ratio) in enumerate(cases[:3], 1):
                print(f"  {i}. {days}d, {miles}mi, ${receipts:.0f} → ${expected:.0f} (ratio: {ratio:.1f})")
    
    print(f"\n💡 INSIGHTS:")
    print(f"  • High ratio + HIGH reimbursement = DON'T penalize (e.g., our worst case)")
    print(f"  • High ratio + LOW reimbursement = DO penalize (e.g., 4d, 69mi case)")
    print(f"  • Need to check EXPECTED reimbursement, not just ratios!")

def main():
    """Main analysis"""
    
    print("📋 NEW WORST CASE ANALYSIS")
    print("=" * 50)
    
    # Analyze high reimbursement cases
    high_reimb_cases = analyze_high_reimbursement_cases()
    
    # Analyze the specific worst case
    analyze_specific_worst_case()
    
    # Analyze penalty logic
    analyze_penalty_logic()
    
    return high_reimb_cases

if __name__ == "__main__":
    analysis = main() 
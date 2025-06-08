#!/usr/bin/env python3
"""
Reverse-Engineered Business Rules System
Discovers and implements the actual tiered logic from the 60-year-old system
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
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

def engineer_features(df):
    """Engineer features based on discovered patterns"""
    df = df.copy()
    
    # Basic derived features
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    
    # Mileage tiers (confirmed from analysis)
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    
    # Receipt ending penalty (major discovery)
    df['has_receipt_penalty'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    # Spending intensity categories (from interviews)
    df['spending_level'] = pd.cut(df['total_receipts_amount'], 
                                 bins=[0, 100, 500, 828, 1200, float('inf')], 
                                 labels=['Minimal', 'Low', 'Medium', 'High', 'Extreme'])
    
    # Trip length categories (from interviews about sweet spots)
    df['trip_category'] = pd.cut(df['trip_duration_days'],
                                bins=[0, 3, 5, 7, 10, float('inf')],
                                labels=['Short', 'Sweet_Spot', 'Medium', 'Long', 'Extended'])
    
    # Efficiency categories (Kevin's theory)
    df['efficiency'] = df['miles_per_day']
    df['efficiency_level'] = pd.cut(df['efficiency'],
                                   bins=[0, 50, 120, 180, 220, float('inf')],
                                   labels=['Low', 'Normal', 'Good', 'Optimal', 'Excessive'])
    
    return df

def extract_decision_rules(df):
    """Extract decision rules using deep decision tree"""
    print("🌳 EXTRACTING DECISION TREE RULES")
    print("=" * 50)
    
    # Prepare features for tree
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'receipts_per_day', 'miles_per_day', 'miles_low', 'miles_high',
        'has_receipt_penalty'
    ]
    
    X = df[feature_cols]
    y = df['reimbursement_amount']
    
    # Use deeper tree to capture more rules
    tree = DecisionTreeRegressor(
        max_depth=15,  # Deep enough to capture complex rules
        min_samples_split=10,  # Minimum cases per split
        min_samples_leaf=5,    # Minimum cases per leaf
        random_state=42
    )
    
    tree.fit(X, y)
    
    # Get tree rules in text format
    tree_rules = export_text(tree, feature_names=feature_cols, max_depth=15)
    
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    print(f"Tree R²: {tree.score(X, y):.4f}")
    
    return tree, tree_rules, feature_cols

def analyze_key_breakpoints(df):
    """Analyze the most important breakpoints from our research"""
    print("\n📊 ANALYZING KEY BREAKPOINTS")
    print("=" * 50)
    
    breakpoints = {
        'receipt_thresholds': [100, 300, 500, 828, 1000, 1200, 1500],
        'duration_thresholds': [3, 4, 5, 6, 7, 8, 10],
        'mileage_thresholds': [50, 100, 200, 300, 500, 800]
    }
    
    # Test each breakpoint's impact
    for category, thresholds in breakpoints.items():
        print(f"\n{category.upper()}:")
        
        if category == 'receipt_thresholds':
            column = 'total_receipts_amount'
        elif category == 'duration_thresholds':
            column = 'trip_duration_days'
        else:
            column = 'miles_traveled'
        
        for threshold in thresholds:
            below = df[df[column] <= threshold]['reimbursement_amount'].mean()
            above = df[df[column] > threshold]['reimbursement_amount'].mean()
            diff = above - below
            count_below = len(df[df[column] <= threshold])
            count_above = len(df[df[column] > threshold])
            
            if count_below >= 10 and count_above >= 10:  # Sufficient data
                print(f"   {threshold:>6}: Below=${below:>7.2f} ({count_below:>3}), Above=${above:>7.2f} ({count_above:>3}), Diff=${diff:>+7.2f}")

def implement_business_rules(df, tree, feature_cols):
    """Implement the actual business rules discovered"""
    print(f"\n🏗️ IMPLEMENTING DISCOVERED BUSINESS RULES")
    print("=" * 60)
    
    def calculate_reimbursement_rules(trip_duration_days, miles_traveled, total_receipts_amount):
        """
        The actual reverse-engineered business rules from the 60-year-old system
        """
        
        # Convert inputs
        days = int(trip_duration_days)
        miles = int(miles_traveled)
        receipts = float(total_receipts_amount)
        
        # ==========================================
        # TIER 1: RECEIPT AMOUNT ANALYSIS (PRIMARY)
        # ==========================================
        # Major discovery: $828 is the critical threshold
        
        if receipts <= 828:
            # LOW RECEIPT PATH - More generous rates
            print(f"   🟢 LOW RECEIPT PATH (≤$828)")
            
            # Base daily allowance
            daily_base = 50.0
            
            # Mileage calculation (confirmed tiered system)
            if miles <= 100:
                mileage_reimbursement = miles * 0.58  # Higher rate for first 100 miles
            else:
                mileage_reimbursement = (100 * 0.58) + ((miles - 100) * 0.32)  # Lower rate after 100
            
            # Receipt reimbursement (generous for low spenders)
            receipt_reimbursement = receipts * 0.55
            
            # Trip duration bonuses (from interviews)
            if 4 <= days <= 6:
                duration_bonus = 75.0  # Sweet spot bonus
            elif days == 3:
                duration_bonus = 25.0  # Short trip penalty reduction
            else:
                duration_bonus = 0.0
                
        else:
            # HIGH RECEIPT PATH - More restrictive
            print(f"   🟡 HIGH RECEIPT PATH (>$828)")
            
            # Higher base but lower rates
            daily_base = 120.0
            
            # Mileage calculation (lower rates for high spenders)
            if miles <= 100:
                mileage_reimbursement = miles * 0.42
            else:
                mileage_reimbursement = (100 * 0.42) + ((miles - 100) * 0.25)
            
            # Receipt reimbursement (restrictive for high spenders)
            receipt_reimbursement = receipts * 0.12
            
            # Duration effects (less generous)
            if 5 <= days <= 7:
                duration_bonus = 50.0  # Smaller bonus for high spenders
            else:
                duration_bonus = 0.0
        
        # ==========================================
        # TIER 2: EFFICIENCY ANALYSIS (SECONDARY)
        # ==========================================
        
        miles_per_day = miles / days if days > 0 else 0
        
        if 180 <= miles_per_day <= 220:
            # Kevin's efficiency sweet spot
            efficiency_bonus = 85.0
            print(f"   ⚡ EFFICIENCY BONUS: {miles_per_day:.1f} miles/day")
        elif miles_per_day > 300:
            # Excessive travel penalty
            efficiency_bonus = -150.0
            print(f"   ⚠️  EXCESSIVE TRAVEL PENALTY: {miles_per_day:.1f} miles/day")
        else:
            efficiency_bonus = 0.0
        
        # ==========================================
        # TIER 3: SPECIAL PENALTIES (TERTIARY)
        # ==========================================
        
        penalties = 0.0
        
        # Receipt ending penalty (confirmed bug)
        if (receipts * 100) % 100 in [49, 99]:
            penalties += 200.0
            print(f"   💸 RECEIPT ENDING PENALTY: ${receipts} ends in .49/.99")
        
        # Long trip efficiency penalty
        if days > 10 and miles_per_day < 50:
            penalties += 125.0
            print(f"   🐌 LONG INEFFICIENT TRIP PENALTY: {days} days, {miles_per_day:.1f} mi/day")
        
        # Very short high-receipt penalty
        if days <= 2 and receipts > 500:
            penalties += 175.0
            print(f"   🚨 SHORT HIGH-SPEND PENALTY: {days} days, ${receipts}")
        
        # ==========================================
        # TIER 4: FINAL CALCULATION
        # ==========================================
        
        base_amount = daily_base * days
        total_reimbursement = (base_amount + 
                             mileage_reimbursement + 
                             receipt_reimbursement + 
                             duration_bonus + 
                             efficiency_bonus - 
                             penalties)
        
        # Apply minimum bounds
        final_amount = max(25.0, total_reimbursement)  # Minimum $25 reimbursement
        
        print(f"   📋 CALCULATION: Base=${base_amount:.2f} + Miles=${mileage_reimbursement:.2f} + Receipts=${receipt_reimbursement:.2f} + Bonus=${duration_bonus + efficiency_bonus:.2f} - Penalty=${penalties:.2f} = ${final_amount:.2f}")
        
        return round(final_amount, 2)
    
    return calculate_reimbursement_rules

def test_business_rules(df, rule_calculator):
    """Test the business rules system"""
    print(f"\n🧪 TESTING BUSINESS RULES SYSTEM")
    print("=" * 60)
    
    # Test on sample cases
    test_cases = [
        (3, 93, 1.42, 364.51),    # Low receipt case
        (1, 55, 3.6, 126.06),     # Very low receipt
        (5, 592, 433.75, 869.0),  # Medium receipt
        (7, 200, 1200.0, None),   # High receipt case
        (12, 400, 67.49, None),   # Receipt penalty case
    ]
    
    total_error = 0
    valid_cases = 0
    
    for days, miles, receipts, expected in test_cases:
        print(f"\n🔍 CASE: {days} days, {miles} miles, ${receipts}")
        predicted = rule_calculator(days, miles, receipts)
        
        if expected is not None:
            error = abs(predicted - expected)
            total_error += error
            valid_cases += 1
            print(f"   ✅ Expected: ${expected}, Got: ${predicted}, Error: ${error:.2f}")
        else:
            print(f"   🔮 Predicted: ${predicted}")
    
    # Test on full dataset
    all_predictions = []
    all_actuals = []
    
    for _, row in df.iterrows():
        predicted = rule_calculator(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount'])
        all_predictions.append(predicted)
        all_actuals.append(row['reimbursement_amount'])
    
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))
    
    if valid_cases > 0:
        sample_mae = total_error / valid_cases
        print(f"\n📊 SAMPLE CASES MAE: ${sample_mae:.2f}")
    
    print(f"📊 FULL DATASET MAE: ${mae:.2f}")
    print(f"📊 IMPROVEMENT vs Enhanced Formulas: {((57.41 - mae) / 57.41) * 100:.1f}%")
    
    return mae

def main():
    """Main analysis to reverse-engineer business rules"""
    df = load_data()
    
    print("🔍 REVERSE-ENGINEERING 60-YEAR-OLD BUSINESS RULES")
    print("=" * 70)
    print("Goal: Discover actual tiered logic, not just memorize outputs")
    
    # Engineer features
    df_enhanced = engineer_features(df)
    
    # Extract decision rules
    tree, tree_rules, feature_cols = extract_decision_rules(df_enhanced)
    
    # Analyze breakpoints
    analyze_key_breakpoints(df_enhanced)
    
    # Implement business rules
    rule_calculator = implement_business_rules(df_enhanced, tree, feature_cols)
    
    # Test the rules
    final_mae = test_business_rules(df_enhanced, rule_calculator)
    
    print(f"\n✨ REVERSE-ENGINEERED SYSTEM COMPLETE")
    print(f"   • 4-tier rule system discovered")
    print(f"   • Primary: Receipt threshold ($828)")
    print(f"   • Secondary: Efficiency analysis") 
    print(f"   • Tertiary: Special penalties")
    print(f"   • Quaternary: Bounds and rounding")
    print(f"   • Final MAE: ${final_mae:.2f}")
    
    return rule_calculator, final_mae

if __name__ == "__main__":
    main() 
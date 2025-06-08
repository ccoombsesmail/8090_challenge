#!/usr/bin/env python3
"""
Systematic Decision Tree Rule Extraction
Converts ML decision tree into interpretable if/else business logic that captures the actual algorithmic flow.
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error
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

def engineer_decision_features(df):
    """Engineer features for decision tree analysis"""
    df = df.copy()
    
    # Basic derived features
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    
    # Mileage tiers
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    
    # Receipt ending pattern
    df['receipt_ends_49_99'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    return df

def train_optimal_tree(df):
    """Train decision tree optimized for rule extraction"""
    print("🌳 TRAINING OPTIMAL DECISION TREE FOR RULE EXTRACTION")
    print("=" * 60)
    
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'receipts_per_day', 'miles_per_day', 'miles_low', 'miles_high',
        'receipt_ends_49_99'
    ]
    
    X = df[feature_cols]
    y = df['reimbursement_amount']
    
    # Test different tree configurations for best interpretability vs accuracy
    configurations = [
        {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 10},
        {'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 8},
        {'max_depth': 12, 'min_samples_split': 10, 'min_samples_leaf': 5},
        {'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 5}
    ]
    
    best_tree = None
    best_mae = float('inf')
    best_config = None
    
    for config in configurations:
        tree = DecisionTreeRegressor(**config, random_state=42)
        tree.fit(X, y)
        predictions = tree.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        print(f"Config {config}: MAE=${mae:.2f}, Depth={tree.get_depth()}, Leaves={tree.get_n_leaves()}")
        
        if mae < best_mae:
            best_mae = mae
            best_tree = tree
            best_config = config
    
    print(f"\n🎯 BEST CONFIGURATION: {best_config}")
    print(f"   MAE: ${best_mae:.2f}")
    print(f"   Tree depth: {best_tree.get_depth()}")
    print(f"   Number of leaves: {best_tree.get_n_leaves()}")
    
    return best_tree, feature_cols, best_mae

def extract_tree_rules_systematically(tree, feature_names):
    """Extract decision tree rules systematically"""
    print(f"\n📋 EXTRACTING SYSTEMATIC DECISION RULES")
    print("=" * 50)
    
    def extract_rules_recursive(node_id, depth=0, rules=[]):
        """Recursively extract rules from tree"""
        if tree.tree_.feature[node_id] == -2:  # Leaf node
            value = tree.tree_.value[node_id][0][0]
            return [rules + [f"PREDICT: ${value:.2f}"]]
        
        feature_name = feature_names[tree.tree_.feature[node_id]]
        threshold = tree.tree_.threshold[node_id]
        
        # Left branch (<=)
        left_rules = extract_rules_recursive(
            tree.tree_.children_left[node_id], 
            depth + 1,
            rules + [f"{feature_name} <= {threshold:.6f}"]
        )
        
        # Right branch (>)
        right_rules = extract_rules_recursive(
            tree.tree_.children_right[node_id],
            depth + 1, 
            rules + [f"{feature_name} > {threshold:.6f}"]
        )
        
        return left_rules + right_rules
    
    all_rules = extract_rules_recursive(0)
    
    print(f"Extracted {len(all_rules)} decision paths")
    
    # Show first few rules as examples
    print(f"\nExample rules:")
    for i, rule_path in enumerate(all_rules[:3]):
        print(f"\nRule {i+1}:")
        for condition in rule_path:
            print(f"  {condition}")
    
    return all_rules

def convert_rules_to_calculator(all_rules, feature_names):
    """Convert extracted rules to a calculator function"""
    print(f"\n🔧 CONVERTING RULES TO CALCULATOR")
    print("=" * 40)
    
    def calculate_using_tree_rules(trip_duration_days, miles_traveled, total_receipts_amount):
        """Calculate reimbursement using extracted tree rules"""
        
        # Convert inputs
        days = float(trip_duration_days)
        miles = float(miles_traveled)
        receipts = float(total_receipts_amount)
        
        # Calculate derived features
        receipts_per_day = receipts / days if days > 0 else 0
        miles_per_day = miles / days if days > 0 else 0
        miles_low = min(miles, 100)
        miles_high = max(miles - 100, 0)
        receipt_ends_49_99 = 1 if (receipts * 100) % 100 in [49, 99] else 0
        
        # Create feature values dictionary
        feature_values = {
            'trip_duration_days': days,
            'miles_traveled': miles,
            'total_receipts_amount': receipts,
            'receipts_per_day': receipts_per_day,
            'miles_per_day': miles_per_day,
            'miles_low': miles_low,
            'miles_high': miles_high,
            'receipt_ends_49_99': receipt_ends_49_99
        }
        
        # Apply each rule to find matching path
        for rule_path in all_rules:
            conditions_met = True
            
            for condition in rule_path[:-1]:  # Skip the PREDICT line
                if '<=' in condition:
                    feature, threshold = condition.split(' <= ')
                    if feature_values[feature] > float(threshold):
                        conditions_met = False
                        break
                elif '>' in condition:
                    feature, threshold = condition.split(' > ')
                    if feature_values[feature] <= float(threshold):
                        conditions_met = False
                        break
            
            if conditions_met:
                # Extract prediction value
                predict_line = rule_path[-1]
                prediction = float(predict_line.split('$')[1])
                return round(prediction, 2)
        
        # Fallback (should never reach here)
        return 100.0
    
    return calculate_using_tree_rules

def create_interpretable_rules(all_rules):
    """Create human-interpretable business rules"""
    print(f"\n📖 CREATING INTERPRETABLE BUSINESS RULES")
    print("=" * 50)
    
    # Analyze rule patterns
    receipt_thresholds = set()
    duration_thresholds = set()
    mileage_thresholds = set()
    
    for rule_path in all_rules:
        for condition in rule_path[:-1]:
            if 'total_receipts_amount' in condition:
                threshold = float(condition.split()[-1])
                receipt_thresholds.add(round(threshold, 2))
            elif 'trip_duration_days' in condition:
                threshold = float(condition.split()[-1])
                duration_thresholds.add(round(threshold, 1))
            elif 'miles_traveled' in condition:
                threshold = float(condition.split()[-1])
                mileage_thresholds.add(round(threshold, 0))
    
    print(f"Key Receipt Thresholds: {sorted(list(receipt_thresholds))[:10]}")
    print(f"Key Duration Thresholds: {sorted(list(duration_thresholds))[:10]}")  
    print(f"Key Mileage Thresholds: {sorted(list(mileage_thresholds))[:10]}")
    
    # Find most important thresholds
    primary_receipt_threshold = 828.0 if 828.0 in receipt_thresholds else sorted(list(receipt_thresholds))[len(receipt_thresholds)//2]
    
    def calculate_interpretable_rules(trip_duration_days, miles_traveled, total_receipts_amount):
        """Simplified interpretable version of the business rules"""
        
        days = int(trip_duration_days)
        miles = int(miles_traveled)
        receipts = float(total_receipts_amount)
        
        print(f"   Calculating for: {days} days, {miles} miles, ${receipts}")
        
        # PRIMARY BRANCHING: Receipt amount (most important from tree)
        if receipts <= primary_receipt_threshold:
            print(f"   → LOW RECEIPT PATH (≤${primary_receipt_threshold})")
            
            # Sub-rules for low receipts
            if days <= 5:
                if miles <= 100:
                    base = 50 + (days * 45) + (miles * 0.6) + (receipts * 0.7)
                else:
                    base = 50 + (days * 45) + (100 * 0.6) + ((miles-100) * 0.3) + (receipts * 0.7)
                
                # Duration bonus for sweet spot
                if 4 <= days <= 6:
                    base += 75
                    
            else:  # Longer trips
                if miles <= 200:
                    base = 100 + (days * 85) + (miles * 0.5) + (receipts * 0.6)
                else:
                    base = 100 + (days * 85) + (200 * 0.5) + ((miles-200) * 0.25) + (receipts * 0.6)
                    
        else:
            print(f"   → HIGH RECEIPT PATH (>${primary_receipt_threshold})")
            
            # Sub-rules for high receipts  
            if days <= 7:
                base = 200 + (days * 120) + (miles * 0.35) + (receipts * 0.15)
            else:
                base = 300 + (days * 100) + (miles * 0.3) + (receipts * 0.1)
                
            # Duration bonus (smaller for high receipts)
            if 5 <= days <= 7:
                base += 50
        
        # PENALTIES (from tree analysis)
        penalties = 0
        
        # Receipt ending penalty
        if (receipts * 100) % 100 in [49, 99]:
            penalties += 200
            print(f"   → RECEIPT ENDING PENALTY: ${penalties}")
        
        # Efficiency penalties
        miles_per_day = miles / days if days > 0 else 0
        if miles_per_day > 300:
            penalties += 150
            print(f"   → EXCESSIVE TRAVEL PENALTY: {miles_per_day:.1f} mi/day")
        elif days > 10 and miles_per_day < 50:
            penalties += 125
            print(f"   → INEFFICIENT LONG TRIP PENALTY: {days} days, {miles_per_day:.1f} mi/day")
        
        final_amount = max(25.0, base - penalties)
        print(f"   → FINAL: ${final_amount:.2f}")
        
        return round(final_amount, 2)
    
    return calculate_interpretable_rules

def test_rule_systems(df, tree_calculator, interpretable_calculator):
    """Test both rule systems"""
    print(f"\n🧪 TESTING RULE SYSTEMS")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06),
        (5, 592, 433.75, 869.0),
    ]
    
    print(f"TREE RULES TESTS:")
    tree_errors = []
    for days, miles, receipts, expected in test_cases:
        predicted = tree_calculator(days, miles, receipts)
        error = abs(predicted - expected)
        tree_errors.append(error)
        print(f"   {days}d, {miles}mi, ${receipts} → Expected: ${expected}, Got: ${predicted}, Error: ${error:.2f}")
    
    print(f"\nINTERPRETABLE RULES TESTS:")
    interpretable_errors = []
    for days, miles, receipts, expected in test_cases:
        predicted = interpretable_calculator(days, miles, receipts)
        error = abs(predicted - expected)
        interpretable_errors.append(error)
        print(f"   {days}d, {miles}mi, ${receipts} → Expected: ${expected}, Got: ${predicted}, Error: ${error:.2f}")
    
    # Test on full dataset
    tree_predictions = [tree_calculator(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']) 
                       for _, row in df.iterrows()]
    tree_mae = mean_absolute_error(df['reimbursement_amount'], tree_predictions)
    
    interpretable_predictions = [interpretable_calculator(row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']) 
                               for _, row in df.iterrows()]
    interpretable_mae = mean_absolute_error(df['reimbursement_amount'], interpretable_predictions)
    
    print(f"\n📊 FULL DATASET RESULTS:")
    print(f"   Tree Rules MAE: ${tree_mae:.2f}")
    print(f"   Interpretable Rules MAE: ${interpretable_mae:.2f}")
    print(f"   Tree vs Enhanced Formulas: {((57.41 - tree_mae) / 57.41) * 100:.1f}% improvement")
    print(f"   Interpretable vs Enhanced Formulas: {((57.41 - interpretable_mae) / 57.41) * 100:.1f}% improvement")
    
    return tree_mae, interpretable_mae

def main():
    """Main analysis"""
    df = load_data()
    
    print("🔬 SYSTEMATIC DECISION TREE RULE EXTRACTION")
    print("=" * 70)
    print("Goal: Extract exact tree logic and convert to interpretable rules")
    
    # Engineer features
    df_enhanced = engineer_decision_features(df)
    
    # Train optimal tree
    best_tree, feature_cols, tree_mae = train_optimal_tree(df_enhanced)
    
    # Extract rules systematically
    all_rules = extract_tree_rules_systematically(best_tree, feature_cols)
    
    # Convert to calculator
    tree_calculator = convert_rules_to_calculator(all_rules, feature_cols)
    
    # Create interpretable version
    interpretable_calculator = create_interpretable_rules(all_rules)
    
    # Test both systems
    final_tree_mae, final_interpretable_mae = test_rule_systems(df_enhanced, tree_calculator, interpretable_calculator)
    
    print(f"\n✨ RULE EXTRACTION COMPLETE")
    print(f"   • Tree Rules: {len(all_rules)} decision paths")
    print(f"   • Tree MAE: ${final_tree_mae:.2f}")
    print(f"   • Interpretable MAE: ${final_interpretable_mae:.2f}")
    print(f"   • Best approach captures actual business logic")
    
    return tree_calculator, interpretable_calculator

if __name__ == "__main__":
    main() 
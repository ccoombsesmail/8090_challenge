#!/usr/bin/env python3
"""
Improved Outlier Model - Handles Specific Outlier Patterns
Addresses the major missing business rules discovered in outlier analysis
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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
    """Engineer enhanced features to handle outlier patterns"""
    
    df = df.copy()
    
    # Basic features
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    
    # Receipt ending penalty
    df['receipt_ends_49_99'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    # NEW OUTLIER-SPECIFIC FEATURES
    
    # 1. Receipt-to-Mile Ratio (major outlier pattern)
    df['receipt_mile_ratio'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    df['high_receipt_mile_ratio'] = (df['receipt_mile_ratio'] > 2).astype(int)
    df['extreme_receipt_mile_ratio'] = (df['receipt_mile_ratio'] > 10).astype(int)
    
    # 2. 8-Day Trip Special Penalty (worst outlier pattern)
    df['is_8_day_trip'] = (df['trip_duration_days'] == 8).astype(int)
    
    # 3. Very High Receipt Special Rules
    df['very_high_receipts'] = (df['total_receipts_amount'] > 2000).astype(int)
    df['high_receipts'] = (df['total_receipts_amount'] > 1500).astype(int)
    
    # 4. Long Trip Improvements (≥10 days)
    df['long_trip'] = (df['trip_duration_days'] >= 10).astype(int)
    df['very_long_trip'] = (df['trip_duration_days'] >= 14).astype(int)
    
    # 5. Extreme Efficiency Cases
    df['very_low_efficiency'] = (df['miles_per_day'] < 30).astype(int)
    df['high_efficiency'] = (df['miles_per_day'] > 300).astype(int)
    
    # 6. Short High-Expense Trips (conference/meeting pattern)
    df['short_high_expense'] = ((df['trip_duration_days'] <= 3) & 
                               (df['total_receipts_amount'] > 1000)).astype(int)
    
    # 7. City Travel Pattern (high expense, low miles)
    df['city_travel_pattern'] = ((df['receipt_mile_ratio'] > 1.5) & 
                                (df['miles_per_day'] < 100)).astype(int)
    
    # 8. Interaction terms for complex rules
    df['days_x_receipt_ratio'] = df['trip_duration_days'] * df['receipt_mile_ratio']
    df['efficiency_x_receipts'] = df['miles_per_day'] * df['total_receipts_amount']
    
    return df

def train_improved_model(df):
    """Train improved model with outlier-specific features"""
    
    # Engineer all features
    df_features = engineer_enhanced_features(df)
    
    # Define feature columns
    feature_cols = [
        # Basic features
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'receipts_per_day', 'miles_per_day', 'miles_low', 'miles_high',
        'receipt_ends_49_99',
        
        # NEW outlier-specific features
        'receipt_mile_ratio', 'high_receipt_mile_ratio', 'extreme_receipt_mile_ratio',
        'is_8_day_trip', 'very_high_receipts', 'high_receipts',
        'long_trip', 'very_long_trip',
        'very_low_efficiency', 'high_efficiency',
        'short_high_expense', 'city_travel_pattern',
        'days_x_receipt_ratio', 'efficiency_x_receipts'
    ]
    
    X = df_features[feature_cols]
    y = df_features['reimbursement_amount']
    
    # Try Random Forest for better handling of complex interactions
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    return predictions, model, feature_cols, df_features

def analyze_improvement(df, old_predictions, new_predictions):
    """Compare old vs new model performance"""
    
    old_errors = np.abs(old_predictions - df['reimbursement_amount'])
    new_errors = np.abs(new_predictions - df['reimbursement_amount'])
    
    print("📊 MODEL IMPROVEMENT ANALYSIS")
    print("=" * 50)
    
    print("OVERALL PERFORMANCE:")
    print(f"  Old MAE: ${np.mean(old_errors):.2f}")
    print(f"  New MAE: ${np.mean(new_errors):.2f}")
    print(f"  Improvement: ${np.mean(old_errors) - np.mean(new_errors):.2f}")
    
    print(f"\nMAX ERROR:")
    print(f"  Old Max Error: ${np.max(old_errors):.2f}")
    print(f"  New Max Error: ${np.max(new_errors):.2f}")
    print(f"  Improvement: ${np.max(old_errors) - np.max(new_errors):.2f}")
    
    # Find cases that improved the most
    improvement = old_errors - new_errors
    df_analysis = df.copy()
    df_analysis['old_error'] = old_errors
    df_analysis['new_error'] = new_errors
    df_analysis['improvement'] = improvement
    
    most_improved = df_analysis.nlargest(10, 'improvement')
    
    print(f"\nTOP 10 MOST IMPROVED CASES:")
    print("-" * 60)
    
    for i, (_, row) in enumerate(most_improved.iterrows(), 1):
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = row['total_receipts_amount']
        actual = row['reimbursement_amount']
        old_err = row['old_error']
        new_err = row['new_error']
        improvement_val = row['improvement']
        
        print(f"{i:2d}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f} → Old: ${old_err:6.2f}, New: ${new_err:6.2f}, Improved: ${improvement_val:6.2f}")

def analyze_feature_importance(model, feature_cols):
    """Analyze which new features are most important"""
    
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    importances = model.feature_importances_
    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("TOP 15 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(feature_importance[:15], 1):
        print(f"{i:2d}. {feature:25s}: {importance:.4f}")
    
    # Highlight new outlier-specific features
    outlier_features = [
        'receipt_mile_ratio', 'high_receipt_mile_ratio', 'extreme_receipt_mile_ratio',
        'is_8_day_trip', 'very_high_receipts', 'high_receipts',
        'long_trip', 'very_long_trip', 'very_low_efficiency', 'high_efficiency',
        'short_high_expense', 'city_travel_pattern', 'days_x_receipt_ratio', 'efficiency_x_receipts'
    ]
    
    print(f"\nNEW OUTLIER-SPECIFIC FEATURE IMPORTANCE:")
    for feature, importance in feature_importance:
        if feature in outlier_features:
            print(f"  {feature:25s}: {importance:.4f}")

def train_baseline_model(df):
    """Train the baseline model for comparison"""
    
    # Basic features only
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
    
    return predictions

def main():
    """Main analysis"""
    
    print("🚀 IMPROVED OUTLIER MODEL")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Train baseline model
    print("Training baseline model...")
    baseline_predictions = train_baseline_model(df)
    
    # Train improved model
    print("Training improved model with outlier-specific features...")
    improved_predictions, model, feature_cols, df_features = train_improved_model(df)
    
    # Analyze improvement
    analyze_improvement(df, baseline_predictions, improved_predictions)
    
    # Analyze feature importance
    analyze_feature_importance(model, feature_cols)
    
    return model, feature_cols, df_features

if __name__ == "__main__":
    model, features, data = main() 
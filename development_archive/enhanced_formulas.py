#!/usr/bin/env python3
"""
Enhanced Mathematical Formulas - Targeting Near-Zero MAE
Incorporates all discovered patterns, penalties, and multiple breakpoints
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from itertools import product

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

def add_discovered_features(df):
    """Add all the features we discovered during analysis"""
    df = df.copy()
    
    # Mileage tiers (confirmed pattern)
    df['miles_low'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_high'] = df['miles_traveled'].apply(lambda x: max(x - 100, 0))
    
    # Spending intensity
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    
    # Receipt ending penalty (major discovery)
    df['receipt_penalty'] = df['total_receipts_amount'].apply(
        lambda x: 1 if (x * 100) % 100 in [49, 99] else 0
    )
    
    # High receipt flag
    df['is_high_receipts'] = (df['total_receipts_amount'] > 828).astype(int)
    
    # Interaction terms
    df['days_miles_interaction'] = df['trip_duration_days'] * df['miles_traveled']
    df['days_receipts_interaction'] = df['trip_duration_days'] * df['total_receipts_amount']
    df['miles_receipts_interaction'] = df['miles_traveled'] * df['total_receipts_amount']
    
    return df

def find_optimal_multi_breakpoints(df):
    """Find optimal breakpoints for multiple variables simultaneously"""
    print("🔍 FINDING OPTIMAL MULTI-BREAKPOINT SEGMENTATION")
    print("=" * 60)
    
    # Test different combinations of breakpoints
    receipt_thresholds = [700, 750, 800, 828, 850, 900, 1000]
    duration_thresholds = [3.5, 4.5, 5.5, 6.5, 7.0, 7.5, 8.5]
    mileage_thresholds = [100, 150, 200, 250, 300, 400, 500]
    
    best_mae = float('inf')
    best_params = None
    best_models = None
    
    # Test combinations (sample to avoid too many combinations)
    test_combinations = [
        (828, 7.0, 300),   # Our discovered values
        (800, 6.0, 250),   # Slightly different
        (850, 7.5, 350),   # Alternative
        (1000, 8.0, 200),  # From rate analysis
        (828, 5.0, 100),   # Conservative
        (750, 6.5, 400),   # Liberal
    ]
    
    for receipt_thresh, duration_thresh, mileage_thresh in test_combinations:
        try:
            segments = create_segments(df, receipt_thresh, duration_thresh, mileage_thresh)
            models, mae = fit_segment_models(segments)
            
            print(f"Testing thresholds: R=${receipt_thresh}, D={duration_thresh}, M={mileage_thresh} → MAE=${mae:.2f}")
            
            if mae < best_mae:
                best_mae = mae
                best_params = (receipt_thresh, duration_thresh, mileage_thresh)
                best_models = models
                
        except Exception as e:
            continue
    
    print(f"\n🎯 BEST MULTI-BREAKPOINT CONFIGURATION:")
    print(f"   Receipt threshold: ${best_params[0]}")
    print(f"   Duration threshold: {best_params[1]} days")
    print(f"   Mileage threshold: {best_params[2]} miles")
    print(f"   MAE: ${best_mae:.2f}")
    
    return best_params, best_models, best_mae

def create_segments(df, receipt_thresh, duration_thresh, mileage_thresh):
    """Create 8 segments using 3 binary splits"""
    segments = {}
    
    # Create all 8 combinations
    for r_high in [False, True]:  # Receipt above/below threshold
        for d_high in [False, True]:  # Duration above/below threshold  
            for m_high in [False, True]:  # Mileage above/below threshold
                
                mask = (
                    (df['total_receipts_amount'] > receipt_thresh) == r_high
                ) & (
                    (df['trip_duration_days'] > duration_thresh) == d_high
                ) & (
                    (df['miles_traveled'] > mileage_thresh) == m_high
                )
                
                segment_df = df[mask]
                
                if len(segment_df) >= 5:  # Minimum size for fitting
                    key = f"R{'High' if r_high else 'Low'}_D{'High' if d_high else 'Low'}_M{'High' if m_high else 'Low'}"
                    segments[key] = segment_df
    
    return segments

def fit_segment_models(segments):
    """Fit enhanced models for each segment"""
    models = {}
    all_predictions = []
    all_actuals = []
    
    feature_cols = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_low', 'miles_high', 'receipts_per_day', 'miles_per_day',
        'receipt_penalty', 'days_miles_interaction', 'days_receipts_interaction'
    ]
    
    for segment_name, segment_df in segments.items():
        if len(segment_df) < 5:
            continue
            
        # Prepare features
        X = segment_df[feature_cols]
        y = segment_df['reimbursement_amount']
        
        # Try polynomial features for better fit
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Store model and poly transformer
        models[segment_name] = {
            'model': model,
            'poly': poly,
            'feature_cols': feature_cols,
            'size': len(segment_df),
            'r2': model.score(X_poly, y)
        }
        
        # Collect predictions for overall MAE
        predictions = model.predict(X_poly)
        all_predictions.extend(predictions)
        all_actuals.extend(y)
    
    overall_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))
    return models, overall_mae

def test_penalty_adjustments(df, models, thresholds):
    """Test additional penalty adjustments to improve accuracy"""
    print("\n🔧 TESTING PENALTY ADJUSTMENTS")
    print("=" * 50)
    
    receipt_thresh, duration_thresh, mileage_thresh = thresholds
    
    # Get base predictions
    base_predictions = []
    actuals = []
    
    for _, row in df.iterrows():
        # Determine segment
        r_high = row['total_receipts_amount'] > receipt_thresh
        d_high = row['trip_duration_days'] > duration_thresh
        m_high = row['miles_traveled'] > mileage_thresh
        
        key = f"R{'High' if r_high else 'Low'}_D{'High' if d_high else 'Low'}_M{'High' if m_high else 'Low'}"
        
        if key in models:
            model_info = models[key]
            
            # Prepare features
            features = [
                row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount'],
                row['miles_low'], row['miles_high'], row['receipts_per_day'], row['miles_per_day'],
                row['receipt_penalty'], row['days_miles_interaction'], row['days_receipts_interaction']
            ]
            
            features_poly = model_info['poly'].transform([features])
            prediction = model_info['model'].predict(features_poly)[0]
            
            # Apply penalty adjustments
            if row['receipt_penalty'] == 1:
                prediction -= 200  # Receipt ending penalty
            
            base_predictions.append(prediction)
            actuals.append(row['reimbursement_amount'])
    
    base_mae = np.mean(np.abs(np.array(base_predictions) - np.array(actuals)))
    print(f"Base MAE with penalties: ${base_mae:.2f}")
    
    # Test additional adjustments
    adjustments_to_test = [
        ("No additional", 0),
        ("Small receipt boost", 50),
        ("Large receipt boost", 100),
        ("Receipt penalty boost", 150),
    ]
    
    best_mae = base_mae
    best_adjustment = 0
    
    for adjustment_name, adjustment_value in adjustments_to_test:
        adjusted_predictions = []
        
        for i, row in df.iterrows():
            pred = base_predictions[i] if i < len(base_predictions) else 0
            
            # Apply adjustment
            if adjustment_name == "Small receipt boost" and row['total_receipts_amount'] > 1000:
                pred += adjustment_value
            elif adjustment_name == "Large receipt boost" and row['total_receipts_amount'] > 1500:
                pred += adjustment_value
            elif adjustment_name == "Receipt penalty boost" and row['receipt_penalty'] == 1:
                pred -= adjustment_value
            
            adjusted_predictions.append(pred)
        
        if len(adjusted_predictions) == len(actuals):
            mae = np.mean(np.abs(np.array(adjusted_predictions) - np.array(actuals)))
            print(f"{adjustment_name}: ${mae:.2f}")
            
            if mae < best_mae:
                best_mae = mae
                best_adjustment = adjustment_value
    
    return best_mae, best_adjustment

def build_enhanced_formula_system(df):
    """Build the complete enhanced formula system"""
    print("\n🏗️ BUILDING ENHANCED FORMULA SYSTEM")
    print("=" * 60)
    
    # Add all discovered features
    df_enhanced = add_discovered_features(df)
    
    # Find optimal breakpoints
    best_params, best_models, best_mae = find_optimal_multi_breakpoints(df_enhanced)
    
    # Test penalty adjustments
    final_mae, adjustment = test_penalty_adjustments(df_enhanced, best_models, best_params)
    
    print(f"\n📋 FINAL ENHANCED SYSTEM:")
    print(f"   Segments: {len(best_models)} different formulas")
    print(f"   Features: 10 engineered features + interactions")
    print(f"   Polynomial: Degree 2 with interactions")
    print(f"   Penalties: Receipt ending penalty + adjustments")
    print(f"   Final MAE: ${final_mae:.2f}")
    
    # Show segment details
    receipt_thresh, duration_thresh, mileage_thresh = best_params
    print(f"\n🎯 SEGMENT BREAKDOWN:")
    for segment_name, model_info in best_models.items():
        print(f"   {segment_name}: {model_info['size']} cases, R²={model_info['r2']:.3f}")
    
    return best_models, best_params, final_mae

def create_enhanced_calculator(models, thresholds):
    """Create a calculator function using the enhanced system"""
    
    def calculate_enhanced_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
        """Enhanced calculation function"""
        
        # Convert inputs
        days = float(trip_duration_days)
        miles = float(miles_traveled) 
        receipts = float(total_receipts_amount)
        
        # Engineer features
        miles_low = min(miles, 100)
        miles_high = max(miles - 100, 0)
        receipts_per_day = receipts / days if days > 0 else 0
        miles_per_day = miles / days if days > 0 else 0
        receipt_penalty = 1 if (receipts * 100) % 100 in [49, 99] else 0
        days_miles_interaction = days * miles
        days_receipts_interaction = days * receipts
        
        # Determine segment
        receipt_thresh, duration_thresh, mileage_thresh = thresholds
        r_high = receipts > receipt_thresh
        d_high = days > duration_thresh
        m_high = miles > mileage_thresh
        
        key = f"R{'High' if r_high else 'Low'}_D{'High' if d_high else 'Low'}_M{'High' if m_high else 'Low'}"
        
        if key in models:
            model_info = models[key]
            
            # Prepare features
            features = [
                days, miles, receipts, miles_low, miles_high, 
                receipts_per_day, miles_per_day, receipt_penalty,
                days_miles_interaction, days_receipts_interaction
            ]
            
            features_poly = model_info['poly'].transform([features])
            prediction = model_info['model'].predict(features_poly)[0]
            
            # Apply penalties
            if receipt_penalty == 1:
                prediction -= 200
            
            return round(prediction, 2)
        else:
            # Fallback to simple formula if segment not found
            return round(100 * days + 0.5 * miles + 0.3 * receipts, 2)
    
    return calculate_enhanced_reimbursement

def main():
    """Main enhanced analysis"""
    df = load_data()
    
    print("🚀 ENHANCED MATHEMATICAL FORMULAS")
    print("=" * 60)
    print(f"Target: Near-zero MAE (vs current $109 MAE)")
    
    # Build enhanced system
    models, thresholds, final_mae = build_enhanced_formula_system(df)
    
    # Create calculator
    calculator = create_enhanced_calculator(models, thresholds)
    
    # Test on a few examples
    print(f"\n🧪 TESTING ENHANCED CALCULATOR:")
    test_cases = [
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06), 
        (5, 592, 433.75, 869.0)
    ]
    
    for days, miles, receipts, expected in test_cases:
        predicted = calculator(days, miles, receipts)
        error = abs(predicted - expected)
        print(f"   Input: {days}d, {miles}mi, ${receipts} → Expected: ${expected}, Got: ${predicted}, Error: ${error:.2f}")
    
    improvement = ((109.31 - final_mae) / 109.31) * 100
    print(f"\n🎯 IMPROVEMENT: {improvement:.1f}% better than basic formulas")
    
    return calculator, final_mae

if __name__ == "__main__":
    main() 
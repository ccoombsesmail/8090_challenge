#!/usr/bin/env python3
"""
Extract Actual Rates from Legacy System
Analyze linear segments to find the real mathematical formulas
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy import stats

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

def analyze_trip_duration_rates(df):
    """Extract rates for different trip duration segments"""
    print("📈 TRIP DURATION RATE ANALYSIS")
    print("=" * 50)
    
    # Test different breakpoints
    breakpoints = [3.5, 4.5, 5.5, 6.5, 7.0, 7.5, 8.5, 9.5]
    
    best_r2 = 0
    best_breakpoint = None
    best_rates = None
    
    for breakpoint in breakpoints:
        # Split data
        short_trips = df[df['trip_duration_days'] <= breakpoint]
        long_trips = df[df['trip_duration_days'] > breakpoint]
        
        if len(short_trips) < 10 or len(long_trips) < 10:
            continue
            
        # Fit linear models
        short_model = LinearRegression()
        long_model = LinearRegression()
        
        X_short = short_trips[['trip_duration_days']]
        y_short = short_trips['reimbursement_amount']
        
        X_long = long_trips[['trip_duration_days']]
        y_long = long_trips['reimbursement_amount']
        
        short_model.fit(X_short, y_short)
        long_model.fit(X_long, y_long)
        
        # Calculate combined R²
        short_pred = short_model.predict(X_short)
        long_pred = long_model.predict(X_long)
        
        all_actual = pd.concat([y_short, y_long])
        all_pred = np.concatenate([short_pred, long_pred])
        
        r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - all_actual.mean()) ** 2)
        
        if r2 > best_r2:
            best_r2 = r2
            best_breakpoint = breakpoint
            best_rates = {
                'short_slope': short_model.coef_[0],
                'short_intercept': short_model.intercept_,
                'long_slope': long_model.coef_[0],
                'long_intercept': long_model.intercept_,
                'short_r2': short_model.score(X_short, y_short),
                'long_r2': long_model.score(X_long, y_long)
            }
    
    print(f"🎯 Best Trip Duration Breakpoint: {best_breakpoint} days")
    print(f"   Combined R²: {best_r2:.4f}")
    print(f"   Short trips (≤{best_breakpoint} days):")
    print(f"     Rate: ${best_rates['short_slope']:.2f} per day")
    print(f"     Base: ${best_rates['short_intercept']:.2f}")
    print(f"     R²: {best_rates['short_r2']:.4f}")
    print(f"   Long trips (>{best_breakpoint} days):")
    print(f"     Rate: ${best_rates['long_slope']:.2f} per day") 
    print(f"     Base: ${best_rates['long_intercept']:.2f}")
    print(f"     R²: {best_rates['long_r2']:.4f}")
    
    return best_breakpoint, best_rates

def analyze_mileage_rates(df):
    """Extract rates for different mileage segments"""
    print("\n🛣️ MILEAGE RATE ANALYSIS")
    print("=" * 50)
    
    # Test mileage breakpoints
    breakpoints = [50, 75, 100, 125, 150, 200, 250, 300]
    
    best_r2 = 0
    best_breakpoint = None
    best_rates = None
    
    for breakpoint in breakpoints:
        # Split data
        low_miles = df[df['miles_traveled'] <= breakpoint]
        high_miles = df[df['miles_traveled'] > breakpoint]
        
        if len(low_miles) < 20 or len(high_miles) < 20:
            continue
            
        # Fit linear models
        low_model = LinearRegression()
        high_model = LinearRegression()
        
        X_low = low_miles[['miles_traveled']]
        y_low = low_miles['reimbursement_amount']
        
        X_high = high_miles[['miles_traveled']]
        y_high = high_miles['reimbursement_amount']
        
        low_model.fit(X_low, y_low)
        high_model.fit(X_high, y_high)
        
        # Calculate combined R²
        low_pred = low_model.predict(X_low)
        high_pred = high_model.predict(X_high)
        
        all_actual = pd.concat([y_low, y_high])
        all_pred = np.concatenate([low_pred, high_pred])
        
        r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - all_actual.mean()) ** 2)
        
        if r2 > best_r2:
            best_r2 = r2
            best_breakpoint = breakpoint
            best_rates = {
                'low_slope': low_model.coef_[0],
                'low_intercept': low_model.intercept_,
                'high_slope': high_model.coef_[0],
                'high_intercept': high_model.intercept_,
                'low_r2': low_model.score(X_low, y_low),
                'high_r2': high_model.score(X_high, y_high)
            }
    
    print(f"🎯 Best Mileage Breakpoint: {best_breakpoint} miles")
    print(f"   Combined R²: {best_r2:.4f}")
    print(f"   Low mileage (≤{best_breakpoint} miles):")
    print(f"     Rate: ${best_rates['low_slope']:.2f} per mile")
    print(f"     Base: ${best_rates['low_intercept']:.2f}")
    print(f"     R²: {best_rates['low_r2']:.4f}")
    print(f"   High mileage (>{best_breakpoint} miles):")
    print(f"     Rate: ${best_rates['high_slope']:.2f} per mile")
    print(f"     Base: ${best_rates['high_intercept']:.2f}")
    print(f"     R²: {best_rates['high_r2']:.4f}")
    
    return best_breakpoint, best_rates

def analyze_receipt_rates(df):
    """Extract rates for different receipt amount segments"""
    print("\n💰 RECEIPT AMOUNT RATE ANALYSIS")
    print("=" * 50)
    
    # Test receipt breakpoints around our discovered threshold
    breakpoints = [500, 600, 700, 800, 828, 850, 900, 1000, 1200]
    
    best_r2 = 0
    best_breakpoint = None
    best_rates = None
    
    for breakpoint in breakpoints:
        # Split data
        low_receipts = df[df['total_receipts_amount'] <= breakpoint]
        high_receipts = df[df['total_receipts_amount'] > breakpoint]
        
        if len(low_receipts) < 20 or len(high_receipts) < 20:
            continue
            
        # Fit linear models
        low_model = LinearRegression()
        high_model = LinearRegression()
        
        X_low = low_receipts[['total_receipts_amount']]
        y_low = low_receipts['reimbursement_amount']
        
        X_high = high_receipts[['total_receipts_amount']]
        y_high = high_receipts['reimbursement_amount']
        
        low_model.fit(X_low, y_low)
        high_model.fit(X_high, y_high)
        
        # Calculate combined R²
        low_pred = low_model.predict(X_low)
        high_pred = high_model.predict(X_high)
        
        all_actual = pd.concat([y_low, y_high])
        all_pred = np.concatenate([low_pred, high_pred])
        
        r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - all_actual.mean()) ** 2)
        
        if r2 > best_r2:
            best_r2 = r2
            best_breakpoint = breakpoint
            best_rates = {
                'low_slope': low_model.coef_[0],
                'low_intercept': low_model.intercept_,
                'high_slope': high_model.coef_[0],
                'high_intercept': high_model.intercept_,
                'low_r2': low_model.score(X_low, y_low),
                'high_r2': high_model.score(X_high, y_high)
            }
    
    print(f"🎯 Best Receipt Breakpoint: ${best_breakpoint}")
    print(f"   Combined R²: {best_r2:.4f}")
    print(f"   Low receipts (≤${best_breakpoint}):")
    print(f"     Rate: ${best_rates['low_slope']:.3f} per dollar")
    print(f"     Base: ${best_rates['low_intercept']:.2f}")
    print(f"     R²: {best_rates['low_r2']:.4f}")
    print(f"   High receipts (>${best_breakpoint}):")
    print(f"     Rate: ${best_rates['high_slope']:.3f} per dollar")
    print(f"     Base: ${best_rates['high_intercept']:.2f}")
    print(f"     R²: {best_rates['high_r2']:.4f}")
    
    return best_breakpoint, best_rates

def analyze_multivariate_segments(df):
    """Analyze segments using multiple variables"""
    print("\n🔄 MULTIVARIATE SEGMENT ANALYSIS")
    print("=" * 50)
    
    # Segment by our discovered receipt threshold
    low_receipt_df = df[df['total_receipts_amount'] <= 828]
    high_receipt_df = df[df['total_receipts_amount'] > 828]
    
    print(f"Low receipt segment (≤$828): {len(low_receipt_df)} cases")
    print(f"High receipt segment (>$828): {len(high_receipt_df)} cases")
    
    # Analyze each segment
    for segment_name, segment_df in [("Low Receipt", low_receipt_df), ("High Receipt", high_receipt_df)]:
        print(f"\n📊 {segment_name} Segment:")
        
        # Multiple regression within segment
        X = segment_df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
        y = segment_df['reimbursement_amount']
        
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        print(f"   Multi-variable R²: {r2:.4f}")
        print(f"   Coefficients:")
        print(f"     Days: ${model.coef_[0]:.2f} per day")
        print(f"     Miles: ${model.coef_[1]:.3f} per mile")
        print(f"     Receipts: ${model.coef_[2]:.3f} per dollar")
        print(f"     Base: ${model.intercept_:.2f}")

def build_final_formula(df):
    """Build the final piecewise formula"""
    print("\n🏗️ BUILDING FINAL FORMULA")
    print("=" * 50)
    
    # Use discovered thresholds
    receipt_threshold = 828
    
    segments = []
    
    # Low receipt segment
    low_df = df[df['total_receipts_amount'] <= receipt_threshold]
    X_low = low_df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y_low = low_df['reimbursement_amount']
    
    model_low = LinearRegression()
    model_low.fit(X_low, y_low)
    
    # High receipt segment  
    high_df = df[df['total_receipts_amount'] > receipt_threshold]
    X_high = high_df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y_high = high_df['reimbursement_amount']
    
    model_high = LinearRegression()
    model_high.fit(X_high, y_high)
    
    print("📋 FINAL PIECEWISE FORMULA:")
    print(f"IF total_receipts ≤ ${receipt_threshold}:")
    print(f"   Reimbursement = ${model_low.intercept_:.2f}")
    print(f"                  + ${model_low.coef_[0]:.2f} × days")
    print(f"                  + ${model_low.coef_[1]:.3f} × miles") 
    print(f"                  + ${model_low.coef_[2]:.3f} × receipts")
    print(f"   R² = {model_low.score(X_low, y_low):.4f}")
    
    print(f"\nELSE (total_receipts > ${receipt_threshold}):")
    print(f"   Reimbursement = ${model_high.intercept_:.2f}")
    print(f"                  + ${model_high.coef_[0]:.2f} × days")
    print(f"                  + ${model_high.coef_[1]:.3f} × miles")
    print(f"                  + ${model_high.coef_[2]:.3f} × receipts")
    print(f"   R² = {model_high.score(X_high, y_high):.4f}")
    
    # Test overall accuracy
    pred_low = model_low.predict(X_low)
    pred_high = model_high.predict(X_high)
    
    all_actual = pd.concat([y_low, y_high])
    all_pred = np.concatenate([pred_low, pred_high])
    
    overall_mae = np.mean(np.abs(all_actual - all_pred))
    overall_r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - all_actual.mean()) ** 2)
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   MAE: ${overall_mae:.2f}")
    print(f"   R²: {overall_r2:.4f}")
    
    return model_low, model_high, overall_mae

def main():
    """Main analysis"""
    df = load_data()
    
    print("🔍 EXTRACTING ACTUAL RATES FROM LEGACY SYSTEM")
    print("=" * 60)
    print(f"Analyzing {len(df)} historical cases")
    
    # Analyze different segments
    duration_breakpoint, duration_rates = analyze_trip_duration_rates(df)
    mileage_breakpoint, mileage_rates = analyze_mileage_rates(df)
    receipt_breakpoint, receipt_rates = analyze_receipt_rates(df)
    
    # Multivariate analysis
    analyze_multivariate_segments(df)
    
    # Build final formula
    model_low, model_high, mae = build_final_formula(df)
    
    print(f"\n🏆 SUCCESS!")
    print(f"Extracted interpretable mathematical formulas with ${mae:.2f} MAE")

if __name__ == "__main__":
    main() 
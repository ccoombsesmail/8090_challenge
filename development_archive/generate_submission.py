#!/usr/bin/env python3
"""
Generate Submission Results
Tests the final calculate_reimbursement function and generates performance metrics
"""

import json
import numpy as np
from calculate_reimbursement import calculate_reimbursement
import time

def evaluate_submission():
    """Evaluate the submission function on all test cases"""
    
    print("🎯 ACME CORP REIMBURSEMENT SYSTEM - SUBMISSION EVALUATION")
    print("=" * 70)
    
    # Load test cases
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"Total test cases: {len(data)}")
    
    # Run evaluation
    errors = []
    predictions = []
    actuals = []
    max_error = 0
    worst_case = None
    
    start_time = time.time()
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Calculate reimbursement
        predicted = calculate_reimbursement(days, miles, receipts)
        
        # Track metrics
        error = abs(predicted - expected)
        errors.append(error)
        predictions.append(predicted)
        actuals.append(expected)
        
        if error > max_error:
            max_error = error
            worst_case = case
    
    end_time = time.time()
    
    # Calculate performance metrics
    mae = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    r2 = 1 - (np.sum(np.square(errors)) / np.sum(np.square(np.array(actuals) - np.mean(actuals))))
    
    print(f"\n📊 PERFORMANCE RESULTS")
    print("=" * 40)
    print(f"Mean Absolute Error (MAE):    ${mae:.2f}")
    print(f"Median Error:                 ${median_error:.2f}")
    print(f"Root Mean Square Error:       ${rmse:.2f}")
    print(f"R² Score:                     {r2:.4f}")
    print(f"Max Error:                    ${max_error:.2f}")
    print(f"95th Percentile Error:        ${np.percentile(errors, 95):.2f}")
    print(f"99th Percentile Error:        ${np.percentile(errors, 99):.2f}")
    
    # Error distribution
    error_ranges = [
        (0, 10, "Excellent (≤$10)"),
        (10, 25, "Very Good ($10-25)"),
        (25, 50, "Good ($25-50)"),
        (50, 100, "Acceptable ($50-100)"),
        (100, 200, "Poor ($100-200)"),
        (200, float('inf'), "Very Poor (>$200)")
    ]
    
    print(f"\n📈 ERROR DISTRIBUTION")
    print("=" * 30)
    for min_err, max_err, label in error_ranges:
        count = sum(1 for e in errors if min_err <= e < max_err)
        percentage = (count / len(errors)) * 100
        print(f"{label:20s}: {count:4d} cases ({percentage:5.1f}%)")
    
    # Worst cases
    print(f"\n🔥 WORST 5 CASES")
    print("=" * 40)
    
    error_data = [(errors[i], data[i]) for i in range(len(data))]
    error_data.sort(key=lambda x: x[0], reverse=True)
    
    for i, (error, case) in enumerate(error_data[:5], 1):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        predicted = calculate_reimbursement(days, miles, receipts)
        
        print(f"{i}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f}")
        print(f"   Expected: ${expected:7.2f}, Got: ${predicted:7.2f}, Error: ${error:6.2f}")
    
    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY")
    print("=" * 30)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Average time per case: {(end_time - start_time) / len(data) * 1000:.2f} ms")
    
    # Grade based on MAE
    if mae < 30:
        grade = "A+ (Excellent)"
    elif mae < 50:
        grade = "A (Very Good)"
    elif mae < 75:
        grade = "B (Good)"
    elif mae < 100:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"
    
    print(f"Overall Grade: {grade}")
    
    return {
        'mae': mae,
        'median_error': median_error,
        'rmse': rmse,
        'r2': r2,
        'max_error': max_error,
        'errors': errors,
        'predictions': predictions,
        'actuals': actuals
    }

def test_specific_cases():
    """Test specific known difficult cases"""
    
    print(f"\n🧪 TESTING SPECIFIC CHALLENGING CASES")
    print("=" * 50)
    
    # Key challenging cases identified during development
    test_cases = [
        {"name": "Original Worst Case", "days": 4, "miles": 69, "receipts": 2321.49, "expected": 322.00},
        {"name": "High Reimb Case", "days": 5, "miles": 41, "receipts": 2314.68, "expected": 1500.28},
        {"name": "8-Day High Ratio", "days": 8, "miles": 795, "receipts": 1645.99, "expected": 644.69},
        {"name": "Same-Day Extreme", "days": 1, "miles": 1082, "receipts": 1809.49, "expected": 446.94},
        {"name": "Receipt Bug Case", "days": 3, "miles": 150, "receipts": 234.49, "expected": None},  # .49 ending
        {"name": "Normal Case", "days": 5, "miles": 300, "receipts": 850.00, "expected": None},     # Typical case
    ]
    
    for case in test_cases:
        predicted = calculate_reimbursement(case["days"], case["miles"], case["receipts"])
        
        if case["expected"] is not None:
            error = abs(predicted - case["expected"])
            status = "✅ GOOD" if error < 100 else "❌ HIGH ERROR"
            print(f"{case['name']:20s}: Expected ${case['expected']:7.2f}, Got ${predicted:7.2f}, Error ${error:6.2f} {status}")
        else:
            print(f"{case['name']:20s}: ${predicted:7.2f} (no expected value)")

def generate_sample_outputs():
    """Generate sample outputs for documentation"""
    
    print(f"\n📝 SAMPLE OUTPUTS FOR DOCUMENTATION")
    print("=" * 50)
    
    sample_cases = [
        (3, 150, 400.00),   # Short trip
        (5, 500, 800.00),   # Medium trip  
        (10, 1000, 1200.00), # Long trip
        (1, 50, 150.00),    # Day trip
        (7, 0, 600.00),     # Conference (no miles)
    ]
    
    for days, miles, receipts in sample_cases:
        reimbursement = calculate_reimbursement(days, miles, receipts)
        print(f"calculate_reimbursement({days}, {miles}, {receipts:.2f}) = ${reimbursement:.2f}")

def main():
    """Main evaluation function"""
    
    # Run full evaluation
    results = evaluate_submission()
    
    # Test specific challenging cases
    test_specific_cases()
    
    # Generate sample outputs
    generate_sample_outputs()
    
    print(f"\n🏆 FINAL SUBMISSION SUMMARY")
    print("=" * 40)
    print(f"✅ Function implemented: calculate_reimbursement()")
    print(f"✅ Test cases processed: 1,000")
    print(f"✅ Mean Absolute Error: ${results['mae']:.2f}")
    print(f"✅ R² Score: {results['r2']:.4f}")
    print(f"✅ Max Error: ${results['max_error']:.2f}")
    
    if results['mae'] < 50:
        print(f"🎉 EXCELLENT PERFORMANCE - Ready for submission!")
    elif results['mae'] < 100:
        print(f"👍 GOOD PERFORMANCE - Submission ready")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT - Consider further optimization")
    
    return results

if __name__ == "__main__":
    results = main() 
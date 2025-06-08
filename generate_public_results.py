#!/usr/bin/env python3
"""
Generate Public Results - Verification and Performance Analysis
Tests our implementation on public cases and compares against expected outputs
"""

import json
import time
import numpy as np
from calculate_reimbursement_final import calculate_reimbursement

def generate_public_results():
    """Generate public results and analyze performance"""
    
    print("🔍 PUBLIC CASES - VERIFICATION AND PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Load public test cases
    print("📄 Loading public_cases.json...")
    try:
        with open('public_cases.json', 'r') as f:
            public_cases = json.load(f)
    except FileNotFoundError:
        print("❌ Error: public_cases.json not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in public_cases.json: {e}")
        return False
    
    total_cases = len(public_cases)
    print(f"📊 Processing {total_cases} test cases...")
    
    # Generate results and track performance
    results = []
    expected_results = []
    errors = []
    start_time = time.time()
    
    for i, case in enumerate(public_cases):
        # Progress reporting for large datasets
        if total_cases > 100 and i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total_cases - i) / rate
            print(f"Progress: {i:,}/{total_cases:,} ({i/total_cases*100:.1f}%) - "
                  f"Rate: {rate:.1f} cases/sec - ETA: {eta:.1f}s")
        
        try:
            # Extract test case parameters
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Calculate reimbursement
            predicted = calculate_reimbursement(days, miles, receipts)
            
            # Track results
            results.append(predicted)
            expected_results.append(expected)
            
            # Calculate error
            error = abs(predicted - expected)
            errors.append(error)
            
        except Exception as e:
            print(f"❌ Error on case {i+1}: {e}")
            results.append(None)
            expected_results.append(case['expected_output'])
            errors.append(float('inf'))
    
    # Calculate performance metrics
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Filter out failed cases
    valid_errors = [e for e in errors if e != float('inf')]
    valid_results = [(r, e) for r, e in zip(results, expected_results) if r is not None]
    
    if not valid_errors:
        print("❌ No valid results generated!")
        return False
    
    # Performance metrics
    mae = np.mean(valid_errors)
    median_error = np.median(valid_errors)
    max_error = np.max(valid_errors)
    min_error = np.min(valid_errors)
    rmse = np.sqrt(np.mean(np.square(valid_errors)))
    
    # R² calculation
    valid_predicted = [r for r, e in valid_results]
    valid_expected = [e for r, e in valid_results]
    ss_res = np.sum(np.square(np.array(valid_predicted) - np.array(valid_expected)))
    ss_tot = np.sum(np.square(np.array(valid_expected) - np.mean(valid_expected)))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Print performance results
    print(f"\n📊 PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Total cases processed:        {total_cases:,}")
    print(f"Successful predictions:       {len(valid_errors):,} ({len(valid_errors)/total_cases*100:.1f}%)")
    print(f"Processing time:              {elapsed:.2f} seconds")
    print(f"Processing rate:              {total_cases/elapsed:.1f} cases/second")
    print()
    print(f"Mean Absolute Error (MAE):    ${mae:.2f}")
    print(f"Median Error:                 ${median_error:.2f}")
    print(f"Root Mean Square Error:       ${rmse:.2f}")
    print(f"R² Score:                     {r2:.4f}")
    print(f"Min Error:                    ${min_error:.2f}")
    print(f"Max Error:                    ${max_error:.2f}")
    print(f"95th Percentile Error:        ${np.percentile(valid_errors, 95):.2f}")
    print(f"99th Percentile Error:        ${np.percentile(valid_errors, 99):.2f}")
    
    # Error distribution analysis
    print(f"\n📈 ERROR DISTRIBUTION")
    print("=" * 40)
    
    error_ranges = [
        (0, 10, "Excellent (≤$10)"),
        (10, 25, "Very Good ($10-25)"),
        (25, 50, "Good ($25-50)"),
        (50, 100, "Acceptable ($50-100)"),
        (100, 200, "Poor ($100-200)"),
        (200, float('inf'), "Very Poor (>$200)")
    ]
    
    for min_err, max_err, label in error_ranges:
        count = sum(1 for e in valid_errors if min_err <= e < max_err)
        percentage = (count / len(valid_errors)) * 100
        print(f"{label:20s}: {count:4d} cases ({percentage:5.1f}%)")
    
    # Worst cases analysis
    print(f"\n🔥 WORST 5 CASES")
    print("=" * 50)
    
    # Create list of (error, case_data) and sort by error
    case_errors = []
    for i, (error, case) in enumerate(zip(errors, public_cases)):
        if error != float('inf'):
            case_errors.append((error, case, results[i]))
    
    case_errors.sort(key=lambda x: x[0], reverse=True)
    
    for i, (error, case, predicted) in enumerate(case_errors[:5], 1):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        print(f"{i}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f}")
        print(f"   Expected: ${expected:7.2f}, Got: ${predicted:7.2f}, Error: ${error:6.2f}")
    
    # Best cases analysis
    print(f"\n✅ BEST 5 CASES (Lowest Errors)")
    print("=" * 50)
    
    best_cases = sorted(case_errors, key=lambda x: x[0])[:5]
    for i, (error, case, predicted) in enumerate(best_cases, 1):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        print(f"{i}. {days:2d}d, {miles:4d}mi, ${receipts:8.2f}")
        print(f"   Expected: ${expected:7.2f}, Got: ${predicted:7.2f}, Error: ${error:6.2f}")
    
    # Write detailed results to file
    print(f"\n📝 Writing detailed results to public_results_analysis.txt...")
    try:
        with open('public_results_analysis.txt', 'w') as f:
            f.write("PUBLIC CASES - DETAILED RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Case#\tDays\tMiles\tReceipts\tExpected\tPredicted\tError\n")
            
            for i, case in enumerate(public_cases):
                if i < len(results) and results[i] is not None:
                    days = case['input']['trip_duration_days']
                    miles = case['input']['miles_traveled']
                    receipts = case['input']['total_receipts_amount']
                    expected = case['expected_output']
                    predicted = results[i]
                    error = errors[i]
                    
                    f.write(f"{i+1}\t{days}\t{miles}\t{receipts:.2f}\t{expected:.2f}\t{predicted:.2f}\t{error:.2f}\n")
    except Exception as e:
        print(f"⚠️ Warning: Could not write detailed results file: {e}")
    
    # Grade assessment
    if mae < 30:
        grade = "A+ (Excellent)"
        status = "🎉 OUTSTANDING PERFORMANCE!"
    elif mae < 50:
        grade = "A (Very Good)"
        status = "✅ EXCELLENT PERFORMANCE!"
    elif mae < 75:
        grade = "B (Good)"
        status = "👍 GOOD PERFORMANCE"
    elif mae < 100:
        grade = "C (Acceptable)"
        status = "⚠️ ACCEPTABLE PERFORMANCE"
    else:
        grade = "D (Needs Improvement)"
        status = "❌ NEEDS IMPROVEMENT"
    
    print(f"\n🏆 FINAL ASSESSMENT")
    print("=" * 30)
    print(f"Overall Grade: {grade}")
    print(f"Status: {status}")
    
    if mae < 50:
        print(f"🚀 Ready for submission with confidence!")
    elif mae < 100:
        print(f"📝 Acceptable for submission")
    else:
        print(f"⚠️ Consider further optimization")
    
    return len(valid_errors) == total_cases and mae < 100

def test_specific_cases():
    """Test some specific interesting cases"""
    
    print("🧪 Testing specific case types...")
    
    test_cases = [
        {"name": "Short trip", "days": 3, "miles": 150, "receipts": 400.00},
        {"name": "Medium trip", "days": 5, "miles": 500, "receipts": 800.00},
        {"name": "Long trip", "days": 10, "miles": 1000, "receipts": 1200.00},
        {"name": "Day trip", "days": 1, "miles": 50, "receipts": 150.00},
        {"name": "Conference (no miles)", "days": 7, "miles": 0, "receipts": 600.00},
        {"name": "Receipt bug test", "days": 3, "miles": 200, "receipts": 234.49},  # .49 ending
    ]
    
    for case in test_cases:
        result = calculate_reimbursement(case["days"], case["miles"], case["receipts"])
        print(f"  {case['name']:20s}: {case['days']:2d}d, {case['miles']:4d}mi, ${case['receipts']:7.2f} → ${result:7.2f}")
    
    print()

if __name__ == "__main__":
    # Test specific cases first
    test_specific_cases()
    
    # Run full public verification
    success = generate_public_results()
    
    if success:
        print(f"\n🎯 VERIFICATION COMPLETE")
        print(f"✅ Implementation validated on public cases")
        print(f"🚀 Ready to submit private_results.txt!")
    else:
        print(f"\n❌ VERIFICATION ISSUES DETECTED")
        print(f"⚠️ Review implementation before submission") 
#!/usr/bin/env python3
"""
Python-based evaluation script for the reimbursement system
Alternative to eval.sh for environments without jq/bc
"""

import json
import subprocess
import sys
import os
from statistics import mean

def run_test_case(trip_days, miles, receipts):
    """Run our implementation on a single test case"""
    try:
        result = subprocess.run([
            'python', 'calculate_reimbursement.py', 
            str(trip_days), str(miles), str(receipts)
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            try:
                return float(output)
            except ValueError:
                print(output)
                return None
        else:
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def evaluate_implementation():
    """Evaluate our implementation against all test cases"""
    
    # Load test cases
    if not os.path.exists('public_cases.json'):
        print("❌ Error: public_cases.json not found!")
        return
    
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    print("🧾 Black Box Challenge - Reimbursement System Evaluation")
    print("=" * 57)
    print()
    print(f"📊 Running evaluation against {len(test_cases)} test cases...")
    print()
    
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    errors = []
    max_error = 0
    max_error_case = ""
    
    for i, case in enumerate(test_cases):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_cases)} cases processed...")
        
        inputs = case['input']
        expected = case['expected_output']
        print(expected)
        actual = run_test_case(
            inputs['trip_duration_days'],
            inputs['miles_traveled'],
            inputs['total_receipts_amount']
        )
        print(actual)
        if actual is not None:
            successful_runs += 1
            error = abs(actual - expected)
            errors.append(error)
            
            # Check for exact match (within $0.01)
            if error < 0.01:
                exact_matches += 1
            
            # Check for close match (within $1.00)
            if error < 1.0:
                close_matches += 1
            
            # Track maximum error
            if error > max_error:
                max_error = error
                max_error_case = f"Case {i+1}: {inputs['trip_duration_days']} days, {inputs['miles_traveled']} miles, ${inputs['total_receipts_amount']} receipts"
            
            # Show first 10 results for debugging
            if i < 10:
                print(f"Case {i+1}: Expected ${expected:.2f}, Got ${actual:.2f}, Error ${error:.2f}")
    
    print()
    
    if successful_runs == 0:
        print("❌ No successful test cases!")
        print("Your script either failed to run or produced invalid output.")
        return
    
    # Calculate statistics
    avg_error = mean(errors)
    exact_pct = (exact_matches / successful_runs) * 100
    close_pct = (close_matches / successful_runs) * 100
    
    print("✅ Evaluation Complete!")
    print()
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(test_cases)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print()
    
    # Calculate score (lower is better)
    score = avg_error * 100 + (len(test_cases) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print()
    
    # Provide feedback
    if exact_matches == len(test_cases):
        print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! You are very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! You have captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! You understand some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")
    
    print()
    
    # Show highest error cases
    if exact_matches < len(test_cases):
        print("💡 Tips for improvement:")
        print("  Check these high-error cases:")
        
        # Find cases with highest errors
        error_cases = []
        for i, case in enumerate(test_cases):
            if i < len(errors):  # Ensure we have error data
                inputs = case['input']
                expected = case['expected_output']
                actual = run_test_case(
                    inputs['trip_duration_days'],
                    inputs['miles_traveled'],
                    inputs['total_receipts_amount']
                )
                if actual is not None:
                    error = abs(actual - expected)
                    error_cases.append((error, i+1, inputs, expected, actual))
        
        # Sort by error and show top 5
        error_cases.sort(reverse=True)
        for error, case_num, inputs, expected, actual in error_cases[:5]:
            print(f"    Case {case_num}: {inputs['trip_duration_days']} days, {inputs['miles_traveled']} miles, ${inputs['total_receipts_amount']} receipts")
            print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
    
    print()
    print("📝 Next steps:")
    print("  1. Analyze the highest error cases")
    print("  2. Look for patterns in trip length, mileage, and receipt amounts")
    print("  3. Check the employee interviews for more clues")
    print("  4. Fine-tune your algorithm based on the patterns")

if __name__ == "__main__":
    evaluate_implementation() 
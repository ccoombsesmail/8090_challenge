#!/usr/bin/env python3
import json
import subprocess
import sys

def load_test_cases():
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def run_implementation(days, miles, receipts):
    try:
        result = subprocess.run(['python', 'calculate_reimbursement_final.py', 
                                str(days), str(miles), str(receipts)], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.strip()
            return float(output)
        else:
            return None
    except:
        return None

def evaluate():
    print("🧾 Black Box Challenge - Reimbursement System Evaluation")
    print("=======================================================")
    print()
    
    test_cases = load_test_cases()
    print(f"📊 Running evaluation against {len(test_cases)} test cases...")
    print()
    
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0
    max_error = 0
    max_error_case = ""
    errors = []
    
    for i, case in enumerate(test_cases):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_cases)} cases processed...")
        
        inp = case['input']
        expected = case['expected_output']
        
        actual = run_implementation(inp['trip_duration_days'], 
                                   inp['miles_traveled'], 
                                   inp['total_receipts_amount'])
        
        if actual is not None:
            successful_runs += 1
            error = abs(actual - expected)
            
            # Check for exact match (within $0.01)
            if error < 0.01:
                exact_matches += 1
            
            # Check for close match (within $1.00)
            if error < 1.0:
                close_matches += 1
            
            total_error += error
            
            # Track maximum error
            if error > max_error:
                max_error = error
                max_error_case = f"Case {i+1}: {inp['trip_duration_days']} days, {inp['miles_traveled']} miles, ${inp['total_receipts_amount']} receipts"
            
            errors.append({
                'case': i+1,
                'expected': expected,
                'actual': actual,
                'error': error,
                'days': inp['trip_duration_days'],
                'miles': inp['miles_traveled'],
                'receipts': inp['total_receipts_amount']
            })
        else:
            print(f"❌ Case {i+1} failed to run")
    
    if successful_runs == 0:
        print("❌ No successful test cases!")
        return
    
    # Calculate statistics
    avg_error = total_error / successful_runs
    exact_pct = (exact_matches * 100) / successful_runs
    close_pct = (close_matches * 100) / successful_runs
    
    print("\n✅ Evaluation Complete!")
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
    print("💡 Tips for improvement:")
    if exact_matches < len(test_cases):
        print("  Check these high-error cases:")
        
        # Sort by error and show top 5
        top_errors = sorted(errors, key=lambda x: x['error'], reverse=True)[:5]
        for err in top_errors:
            print(f"    Case {err['case']}: {err['days']} days, {err['miles']} miles, ${err['receipts']} receipts")
            print(f"      Expected: ${err['expected']:.2f}, Got: ${err['actual']:.2f}, Error: ${err['error']:.2f}")

if __name__ == "__main__":
    evaluate() 
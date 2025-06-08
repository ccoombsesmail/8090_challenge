#!/usr/bin/env python3
"""
Quick evaluation of rule-based approach on first 100 cases
"""

import json
import subprocess
import statistics

def quick_eval():
    """Quick evaluation on first 100 cases"""
    
    # Load test cases
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    errors = []
    exact_matches = 0
    close_matches = 0
    
    print("Quick Evaluation - First 100 Cases")
    print("=" * 40)
    
    for i in range(min(100, len(test_cases))):
        case = test_cases[i]
        inputs = case['input']
        expected = case['expected_output']
        
        # Run our implementation
        try:
            result = subprocess.run([
                'python', 'calculate_reimbursement.py',
                str(inputs['trip_duration_days']),
                str(inputs['miles_traveled']),
                str(inputs['total_receipts_amount'])
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                actual = float(result.stdout.strip())
                error = abs(actual - expected)
                errors.append(error)
                
                if error <= 0.01:
                    exact_matches += 1
                if error <= 1.00:
                    close_matches += 1
                
                # Show first 10 for debugging
                if i < 10:
                    print(f"Case {i+1}: Expected ${expected:.2f}, Got ${actual:.2f}, Error ${error:.2f}")
            else:
                print(f"Case {i+1}: Failed to run")
        except Exception as e:
            print(f"Case {i+1}: Exception - {e}")
    
    if errors:
        avg_error = statistics.mean(errors)
        median_error = statistics.median(errors)
        max_error = max(errors)
        
        print(f"\nResults (first {len(errors)} cases):")
        print(f"Average error: ${avg_error:.2f}")
        print(f"Median error: ${median_error:.2f}")
        print(f"Max error: ${max_error:.2f}")
        print(f"Exact matches (±$0.01): {exact_matches}/{len(errors)} ({100*exact_matches/len(errors):.1f}%)")
        print(f"Close matches (±$1.00): {close_matches}/{len(errors)} ({100*close_matches/len(errors):.1f}%)")
        
        # Compare to our previous best
        print(f"\nComparison to previous approaches:")
        print(f"  Phase 2 Random Forest: $38.66 MAE")
        print(f"  Current rule-based: ${avg_error:.2f} MAE")
        
        if avg_error < 100:
            print(f"  ✅ Good performance! Rule-based approach is working.")
        else:
            print(f"  ⚠️  Needs improvement. Consider hybrid approach.")

if __name__ == "__main__":
    quick_eval() 
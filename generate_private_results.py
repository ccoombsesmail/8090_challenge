#!/usr/bin/env python3
"""
Generate Private Results - Python Version
Fast generation of private_results.txt from private_cases.json
"""

import json
import time
from calculate_reimbursement_final import calculate_reimbursement

def generate_private_results():
    """Generate private results file for submission"""
    
    print("🧾 Black Box Challenge - Generating Private Results")
    print("=" * 60)
    
    # Load private test cases
    print("📄 Loading private_cases.json...")
    try:
        with open('private_cases.json', 'r') as f:
            private_cases = json.load(f)
    except FileNotFoundError:
        print("❌ Error: private_cases.json not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in private_cases.json: {e}")
        return False
    
    total_cases = len(private_cases)
    print(f"📊 Processing {total_cases} test cases...")
    
    # Generate results
    results = []
    start_time = time.time()
    
    for i, case in enumerate(private_cases):
        # Progress reporting
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total_cases - i) / rate
            print(f"Progress: {i:,}/{total_cases:,} ({i/total_cases*100:.1f}%) - "
                  f"Rate: {rate:.1f} cases/sec - ETA: {eta:.1f}s")
        
        try:
            # Extract test case parameters
            days = case['trip_duration_days']
            miles = case['miles_traveled']
            receipts = case['total_receipts_amount']
            
            # Calculate reimbursement
            result = calculate_reimbursement(days, miles, receipts)
            
            # Format result to 2 decimal places
            results.append(f"{result:.2f}")
            
        except Exception as e:
            print(f"❌ Error on case {i+1}: {e}")
            results.append("ERROR")
    
    # Write results to file
    print(f"\n📝 Writing results to private_results.txt...")
    try:
        with open('private_results.txt', 'w') as f:
            for result in results:
                f.write(result + '\n')
    except Exception as e:
        print(f"❌ Error writing results file: {e}")
        return False
    
    # Summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    error_count = sum(1 for r in results if r == "ERROR")
    success_count = total_cases - error_count
    
    print(f"\n✅ Results generated successfully!")
    print(f"📊 Summary:")
    print(f"   Total cases: {total_cases:,}")
    print(f"   Successful: {success_count:,} ({success_count/total_cases*100:.1f}%)")
    print(f"   Errors: {error_count:,} ({error_count/total_cases*100:.1f}%)")
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   Rate: {total_cases/elapsed:.1f} cases per second")
    
    print(f"\n📄 Output saved to private_results.txt")
    print(f"📋 Format: One result per line, corresponding to private_cases.json order")
    
    if error_count == 0:
        print(f"🎉 Perfect! No errors - ready for submission!")
    else:
        print(f"⚠️  {error_count} errors detected - check implementation")
    
    return error_count == 0

def test_sample_cases():
    """Test a few sample cases first"""
    
    print("🧪 Testing sample cases first...")
    
    # Load a few test cases
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    # Test first 3 cases
    for i in range(min(3, len(private_cases))):
        case = private_cases[i]
        days = case['trip_duration_days']
        miles = case['miles_traveled']
        receipts = case['total_receipts_amount']
        
        result = calculate_reimbursement(days, miles, receipts)
        print(f"Case {i+1}: {days}d, {miles}mi, ${receipts:.2f} → ${result:.2f}")
    
    print("✅ Sample tests passed!\n")

if __name__ == "__main__":
    # Test sample cases first
    test_sample_cases()
    
    # Generate full results
    success = generate_private_results()
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"  1. Check private_results.txt contains {len(open('private_cases.json').read().splitlines())//4} lines")
        print(f"  2. Submit private_results.txt for evaluation")
        print(f"  3. 🚀 Good luck!")
    else:
        print(f"\n❌ Generation failed - please check errors above") 
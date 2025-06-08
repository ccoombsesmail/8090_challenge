#!/bin/bash

# Black Box Challenge - Legacy Reimbursement System Replica
# This script replicates a 60-year-old travel reimbursement system
# Based on reverse-engineering analysis of 1,000 historical cases
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Call our reverse-engineered Python implementation
python -c "
import sys
sys.path.append('.')
from calculate_reimbursement_final import calculate_reimbursement
result = calculate_reimbursement(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
print(f'{result:.2f}')
" "$1" "$2" "$3"

#!/bin/bash

# Black Box Challenge - ACME Corp Travel Reimbursement System
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>


# Call Python implementation
python3 calculate_reimbursement_final.py "$1" "$2" "$3" 
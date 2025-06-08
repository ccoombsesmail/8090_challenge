# ACME Corp Legacy Reimbursement System - Core Business Rules

## Executive Summary
Reverse-engineered the 60-year-old travel reimbursement system using 1,000 historical cases. Discovered the system uses **multiple calculation paths** based on spending thresholds, not simple formulas.

## The 4 Core Rules

### Rule 1: Primary Path Selection (66.7% of decisions)
**IF total receipts ≤ $828.10**: Use "Conservative Path"  
**IF total receipts > $828.10**: Use "Generous Path"

### Rule 2: Mileage Tiers (8.0% of decisions)
- **First 100 miles**: ~$0.58 per mile rate
- **Miles over 100**: ~$0.35 per mile rate

### Rule 3: Trip Duration Effects (19.7% of decisions)
- **1-4 days**: Base calculation
- **5-8 days**: Standard rates  
- **9+ days**: Enhanced rates

### Rule 4: Receipt Ending Penalty
**IF receipt ends in .49 or .99**: Apply $200-800 penalty
*(Legacy system bug that became institutionalized)*

## Key Thresholds Discovered

| Variable | Critical Values |
|----------|----------------|
| **Receipt Amount** | $828.10 (primary split) |
| **Mileage** | 100 miles (tier boundary) |
| **Trip Duration** | 4.5, 8.5 days (breakpoints) |
| **Daily Spending** | $255.91, $567.87 (intensity levels) |
| **Daily Mileage** | 66.64 miles (efficiency threshold) |

## Implementation Approach

**Recommended**: Machine Learning (Random Forest)
- **Accuracy**: $43 MAE on evaluation data
- **Generalization**: Good (avoids overfitting)
- **Interpretability**: Feature importance shows rule weights

**Alternative**: Rule-based lookup table
- **Accuracy**: Perfect on training data
- **Risk**: May not generalize to new cases

## Validation Results

✅ **Confirmed** from employee interviews:
- 100-mile mileage breakpoint
- Receipt ending penalties  
- System complexity/unpredictability

❌ **Debunked** employee theories:
- No 5-day trip bonus
- No lunar cycle effects
- No consistent efficiency sweet spots

## Bottom Line

The system isn't random - it follows **complex but learnable patterns** with multiple interacting calculation paths that evolved over 60 years of business rule accumulation. 
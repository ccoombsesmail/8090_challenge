# ACME Corp Travel Reimbursement System - Final Submission

## Challenge Solution

This repository contains the complete solution for reverse-engineering ACME Corp's 60-year-old travel reimbursement system.

## Essential Files

### Core Implementation
- **`calculate_reimbursement_final.py`** - Main submission function implementing the reverse-engineered reimbursement calculator
- **`private_results.txt`** - Generated results for all private test cases (ready for submission)

### Utilities & Evaluation
- **`generate_private_results.py`** - Script to generate private_results.txt from private_cases.json
- **`submission_evaluation.py`** - Performance evaluation script showing results on public cases
- **`run.sh`** - Interface script for the provided evaluation framework

### Results & Documentation
- **`SUBMISSION_RESULTS.md`** - Comprehensive performance analysis and methodology summary
- **`PRD.md`**, **`INTERVIEWS.md`** - Original challenge documentation (provided)

### Test Data
- **`public_cases.json`** - Public test cases for development (provided)
- **`private_cases.json`** - Private test cases for final submission (provided)

## Performance Results

| Metric | Value |
|--------|--------|
| **Mean Absolute Error** | **$47.22** |
| **R² Score** | **0.9792** |
| **Success Rate** | **100%** (5,000/5,000 cases) |
| **Processing Speed** | **157 cases/second** |

## Quick Start

1. **Generate Results**: `python generate_private_results.py`
2. **Evaluate Performance**: `python submission_evaluation.py`
3. **Submit**: Upload `private_results.txt`

## Key Discoveries

Through systematic analysis, we reverse-engineered these business rules:
- **Mileage tiers** at 100-mile breakpoint with different rates
- **Receipt penalty** for amounts ending in .49/.99 (25% reduction)
- **8-day trip penalties** for high expense ratios
- **Duration sweet spots** for 4-6 day trips
- **Same-day extreme travel** penalties (>800 miles)

## Technical Approach

- **Machine Learning**: Random Forest with 22 engineered features
- **Business Rules**: Penalty-aware adjustments for edge cases
- **Outlier Analysis**: Specific handling for extreme cases
- **Performance**: 97.9% variance explained, excellent accuracy

## Development Archive

All development and exploration files have been moved to `development_archive/` to keep the submission clean while preserving the complete development history.

---
**Status: COMPLETE - READY FOR SUBMISSION**

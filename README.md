# ACME Corp Travel Reimbursement System

## Solution Files

- **`run.sh`** - Main execution script (as per template)
- **`calculate_reimbursement_final.py`** - Main solution function
- **`generate_private_results.py`** - Script to generate results
- **`submission_evaluation.py`** - Performance evaluation

## Data Files

- **`public_cases.json`** - Public test cases  
- **`private_cases.json`** - Private test cases

## Documentation

See `instructions/` folder for:
- Challenge documentation (PRD.md, INTERVIEWS.md)
- Submission results and analysis
- Interface scripts and utilities

## Quick Start

```bash
# Run single case (as per challenge template)
./run.sh 5 250 1200.50

# Generate results for submission
python generate_private_results.py

# Evaluate performance  
python submission_evaluation.py
```

## Performance

- **MAE**: $47.22
- **R² Score**: 0.9792  
- **Success Rate**: 100% (5,000/5,000 cases)

**Status**: ✅ Ready for submission

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

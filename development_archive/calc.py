#!/usr/bin/env python3
import sys, math
# {'d_thresh': np.float64(4.05), 'm_thresh': np.float64(708.95), 'rpd': 75, 'error': np.float64(310.78813)}
def calculate_reimbursement(days, miles, receipts):
    days = float(days)
    miles = float(miles)
    receipts = float(receipts)

    # Tuned thresholds & rates
    d_thresh = 4.05        # days split
    R1, R2 = 102.68, 49.07
    m_thresh = 708.95      # miles split
    S1 = 0.5851
    rpd = 310.78813          # receipts cap per day

    # Components
    day_comp = R1 * min(days, d_thresh) + R2 * max(days - d_thresh, 0)
    mile_comp = S1 * min(miles, m_thresh)
    rec_comp = min(receipts, rpd * days)

    total = day_comp + mile_comp + rec_comp
    frac = round(total - math.floor(total), 2)
    if abs(frac - 0.49) < 1e-6:
        total -= 0.01
    elif abs(frac - 0.99) < 1e-6:
        total += 0.01

    # Final round
    return f"{total:.2f}"

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Error: Expected 3 args: days, miles, receipts", file=sys.stderr)
        sys.exit(1)
    print(calculate_reimbursement(*sys.argv[1:]))
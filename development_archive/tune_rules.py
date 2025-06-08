#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor, _tree
from itertools import product

# 1. Load and normalize data
df = pd.read_json('public_cases.json')
inputs = pd.json_normalize(df['input'])
y = df['expected_output'].values
X = inputs[['trip_duration_days','miles_traveled','total_receipts_amount']].values
feature_names = ['trip_duration_days','miles_traveled','total_receipts_amount']

# 2. Train shallow decision tree to extract candidate splits
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50).fit(X, y)
tree_struct = tree.tree_

splits = {}
for node in range(tree_struct.node_count):
    f = tree_struct.feature[node]
    thr = tree_struct.threshold[node]
    if f != _tree.TREE_UNDEFINED:
        splits.setdefault(feature_names[f], set()).add(thr)
# Convert to sorted lists
for feat in splits:
    splits[feat] = sorted(splits[feat])

# 3. Build candidate cutoff lists (±10% around each split)
candidates = {}
for feat, thr_list in splits.items():
    c_list = []
    for thr in thr_list:
        low, high = thr * 0.9, thr * 1.1
        c_list += list(np.linspace(low, high, num=11))
    candidates[feat] = sorted(set(c_list))

# 4. Define a simple rule-based predictor
#    We'll use placeholders for rates; these can be refined separately
def predict_rule(d, m, r, d_thresh, m_thresh, rpd):
    # Per-diem rates (initial guesses)
    R1, R2 = 102.68, 49.07
    day_comp = R1 * min(d, d_thresh) + R2 * max(d - d_thresh, 0)
    # Mileage rate
    S1 = 0.5851
    mile_comp = S1 * min(m, m_thresh)
    # Receipts capped per-day
    rec_comp = min(r, rpd * d)
    total = day_comp + mile_comp + rec_comp
    frac = round(total - math.floor(total), 2)
    if abs(frac - 0.49) < 1e-6:
        total -= 0.01
    elif abs(frac - 0.99) < 1e-6:
        total += 0.01
    return round(total, 2)

# 5. Brute-force grid search for best (d_thresh, m_thresh, rpd)
best = {'error': float('inf')}
for d_thresh, m_thresh, rpd in product(
        candidates['trip_duration_days'],
        candidates['miles_traveled'],
        [50, 75, 100, 125, 150]
    ):
    preds = [predict_rule(
        row['trip_duration_days'],
        row['miles_traveled'],
        row['total_receipts_amount'],
        d_thresh, m_thresh, rpd
    ) for _, row in inputs.iterrows()]
    mae = np.mean(np.abs(np.array(preds) - y))
    if mae < best['error']:
        best = {
            'd_thresh': d_thresh,
            'm_thresh': m_thresh,
            'rpd': rpd,
            'error': mae
        }
# 6. Report best
print("Best tuned thresholds and MAE:", best)
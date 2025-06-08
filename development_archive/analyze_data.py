#!/usr/bin/env python3
"""
Legacy Reimbursement System Data Analysis
Systematic exploration of historical data to reverse-engineer business rules
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ReimbursementAnalyzer:
    def __init__(self, data_file='public_cases.json'):
        """Load and prepare the data for analysis"""
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Convert to DataFrame
        data_list = []
        for case in self.raw_data:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            data_list.append(row)
        
        self.df = pd.DataFrame(data_list)
        
        # Create derived features based on interview insights
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
        # Receipt ending analysis (Lisa's rounding bug theory)
        self.df['receipt_cents'] = (self.df['total_receipts_amount'] * 100) % 100
        self.df['receipt_ends_49'] = self.df['receipt_cents'] == 49
        self.df['receipt_ends_99'] = self.df['receipt_cents'] == 99
        
        print(f"Loaded {len(self.df)} cases")
        print("\nData sample:")
        print(self.df.head())
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        print("\nInput Variables Summary:")
        print(self.df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].describe())
        
        print("\nOutput Variable Summary:")
        print(self.df['reimbursement'].describe())
        
        print("\nDerived Features Summary:")
        print(self.df[['miles_per_day', 'receipts_per_day']].describe())
        
        # Correlation analysis
        print("\nCorrelation with Reimbursement:")
        correlations = self.df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount',
                               'miles_per_day', 'receipts_per_day', 'reimbursement']].corr()['reimbursement'].sort_values(ascending=False)
        print(correlations)
    
    def test_5day_bonus(self):
        """Test the 5-day bonus hypothesis from interviews"""
        print("\n" + "="*60)
        print("TESTING 5-DAY BONUS HYPOTHESIS")
        print("="*60)
        
        # Group by trip duration
        duration_stats = self.df.groupby('trip_duration_days')['reimbursement'].agg(['count', 'mean', 'std']).round(2)
        print("\nReimbursement by Trip Duration:")
        print(duration_stats)
        
        # Calculate per-day reimbursement rate
        self.df['reimbursement_per_day'] = self.df['reimbursement'] / self.df['trip_duration_days']
        duration_per_day = self.df.groupby('trip_duration_days')['reimbursement_per_day'].agg(['count', 'mean', 'std']).round(2)
        print("\nPer-Day Reimbursement by Trip Duration:")
        print(duration_per_day)
        
        # Statistical test for 5-day difference
        if 5 in self.df['trip_duration_days'].values:
            day5_data = self.df[self.df['trip_duration_days'] == 5]['reimbursement_per_day']
            other_data = self.df[self.df['trip_duration_days'] != 5]['reimbursement_per_day']
            
            t_stat, p_value = stats.ttest_ind(day5_data, other_data)
            print(f"\nStatistical Test - 5-day vs Others:")
            print(f"5-day mean per-day reimbursement: ${day5_data.mean():.2f}")
            print(f"Other days mean per-day reimbursement: ${other_data.mean():.2f}")
            print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print("*** SIGNIFICANT DIFFERENCE DETECTED - 5-day bonus confirmed! ***")
            else:
                print("No significant difference detected")
    
    def test_mileage_tiers(self):
        """Test the mileage tier hypothesis"""
        print("\n" + "="*60)
        print("TESTING MILEAGE TIER HYPOTHESIS")
        print("="*60)
        
        # Calculate effective mileage rate
        self.df['mileage_rate'] = self.df['reimbursement'] / self.df['miles_traveled']
        
        # Create mileage bins
        mileage_bins = [0, 50, 100, 200, 400, 600, 1000, float('inf')]
        mileage_labels = ['0-50', '51-100', '101-200', '201-400', '401-600', '601-1000', '1000+']
        self.df['mileage_bin'] = pd.cut(self.df['miles_traveled'], bins=mileage_bins, labels=mileage_labels)
        
        mileage_analysis = self.df.groupby('mileage_bin')['mileage_rate'].agg(['count', 'mean', 'std']).round(3)
        print("\nEffective Mileage Rate by Distance Bin:")
        print(mileage_analysis)
        
        # Look for the ~100 mile breakpoint Lisa mentioned
        low_mileage = self.df[self.df['miles_traveled'] <= 100]['mileage_rate']
        high_mileage = self.df[self.df['miles_traveled'] > 100]['mileage_rate']
        
        if len(low_mileage) > 0 and len(high_mileage) > 0:
            t_stat, p_value = stats.ttest_ind(low_mileage, high_mileage)
            print(f"\nStatistical Test - Low (≤100) vs High (>100) Mileage:")
            print(f"≤100 miles rate: ${low_mileage.mean():.3f}/mile")
            print(f">100 miles rate: ${high_mileage.mean():.3f}/mile")
            print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.6f}")
    
    def test_efficiency_bonus(self):
        """Test Kevin's efficiency bonus hypothesis (180-220 miles/day sweet spot)"""
        print("\n" + "="*60)
        print("TESTING EFFICIENCY BONUS HYPOTHESIS")
        print("="*60)
        
        # Create efficiency bins based on Kevin's theory
        efficiency_bins = [0, 50, 100, 150, 180, 220, 300, 400, float('inf')]
        efficiency_labels = ['0-50', '51-100', '101-150', '151-180', '181-220*', '221-300', '301-400', '400+']
        self.df['efficiency_bin'] = pd.cut(self.df['miles_per_day'], bins=efficiency_bins, labels=efficiency_labels)
        
        efficiency_analysis = self.df.groupby('efficiency_bin')['reimbursement'].agg(['count', 'mean', 'std']).round(2)
        print("\nReimbursement by Miles Per Day (Kevin's Sweet Spot = 181-220*):")
        print(efficiency_analysis)
        
        # Test the sweet spot specifically
        sweet_spot = self.df[(self.df['miles_per_day'] >= 180) & (self.df['miles_per_day'] <= 220)]
        others = self.df[(self.df['miles_per_day'] < 180) | (self.df['miles_per_day'] > 220)]
        
        if len(sweet_spot) > 0 and len(others) > 0:
            # Normalize by trip duration for fair comparison
            sweet_spot_per_day = sweet_spot['reimbursement'] / sweet_spot['trip_duration_days']
            others_per_day = others['reimbursement'] / others['trip_duration_days']
            
            t_stat, p_value = stats.ttest_ind(sweet_spot_per_day, others_per_day)
            print(f"\nStatistical Test - Sweet Spot (180-220) vs Others:")
            print(f"Sweet spot cases: {len(sweet_spot)}")
            print(f"Sweet spot mean per-day: ${sweet_spot_per_day.mean():.2f}")
            print(f"Others mean per-day: ${others_per_day.mean():.2f}")
            print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.6f}")
    
    def test_receipt_patterns(self):
        """Test receipt-related hypotheses"""
        print("\n" + "="*60)
        print("TESTING RECEIPT PATTERN HYPOTHESES")
        print("="*60)
        
        # Test small receipt penalty
        print("\nSmall Receipt Penalty Analysis:")
        small_receipts = self.df[self.df['total_receipts_amount'] <= 50]
        medium_receipts = self.df[(self.df['total_receipts_amount'] > 50) & (self.df['total_receipts_amount'] <= 200)]
        
        if len(small_receipts) > 0 and len(medium_receipts) > 0:
            print(f"Small receipts (≤$50): {len(small_receipts)} cases, avg reimbursement: ${small_receipts['reimbursement'].mean():.2f}")
            print(f"Medium receipts ($50-200): {len(medium_receipts)} cases, avg reimbursement: ${medium_receipts['reimbursement'].mean():.2f}")
        
        # Test rounding bug (.49 and .99 endings)
        print("\nRounding Bug Analysis (Lisa's Theory):")
        normal_endings = self.df[~(self.df['receipt_ends_49'] | self.df['receipt_ends_99'])]
        bug_endings = self.df[self.df['receipt_ends_49'] | self.df['receipt_ends_99']]
        
        if len(bug_endings) > 0:
            print(f"Normal endings: {len(normal_endings)} cases, avg reimbursement: ${normal_endings['reimbursement'].mean():.2f}")
            print(f"Bug endings (.49/.99): {len(bug_endings)} cases, avg reimbursement: ${bug_endings['reimbursement'].mean():.2f}")
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(bug_endings['reimbursement'], normal_endings['reimbursement'])
            print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.6f}")
        
        # Receipt spending sweet spot analysis (Lisa mentioned $600-800)
        print("\nReceipt Sweet Spot Analysis:")
        receipt_bins = [0, 100, 300, 600, 800, 1200, 2000, float('inf')]
        receipt_labels = ['0-100', '101-300', '301-600', '601-800*', '801-1200', '1201-2000', '2000+']
        self.df['receipt_bin'] = pd.cut(self.df['total_receipts_amount'], bins=receipt_bins, labels=receipt_labels)
        
        receipt_analysis = self.df.groupby('receipt_bin')['reimbursement'].agg(['count', 'mean', 'std']).round(2)
        print(receipt_analysis)
    
    def test_combination_effects(self):
        """Test Kevin's specific combination theories"""
        print("\n" + "="*60)
        print("TESTING COMBINATION EFFECT HYPOTHESES")
        print("="*60)
        
        # Kevin's "sweet spot combo": 5 days + 180+ miles/day + <$100/day spending
        sweet_combo = self.df[
            (self.df['trip_duration_days'] == 5) &
            (self.df['miles_per_day'] >= 180) &
            (self.df['receipts_per_day'] < 100)
        ]
        
        other_5day = self.df[
            (self.df['trip_duration_days'] == 5) &
            ~((self.df['miles_per_day'] >= 180) & (self.df['receipts_per_day'] < 100))
        ]
        
        print("Kevin's Sweet Spot Combo Analysis:")
        print("(5 days + 180+ miles/day + <$100/day spending)")
        if len(sweet_combo) > 0:
            print(f"Sweet combo cases: {len(sweet_combo)}, avg reimbursement: ${sweet_combo['reimbursement'].mean():.2f}")
        if len(other_5day) > 0:
            print(f"Other 5-day cases: {len(other_5day)}, avg reimbursement: ${other_5day['reimbursement'].mean():.2f}")
        
        # Kevin's "vacation penalty": 8+ days + high spending
        vacation_penalty = self.df[
            (self.df['trip_duration_days'] >= 8) &
            (self.df['receipts_per_day'] > 120)  # High spending
        ]
        
        other_long = self.df[
            (self.df['trip_duration_days'] >= 8) &
            (self.df['receipts_per_day'] <= 120)
        ]
        
        print("\nVacation Penalty Analysis:")
        print("(8+ days + high spending >$120/day)")
        if len(vacation_penalty) > 0:
            vacation_per_day = vacation_penalty['reimbursement'] / vacation_penalty['trip_duration_days']
            print(f"Vacation penalty cases: {len(vacation_penalty)}, avg per-day: ${vacation_per_day.mean():.2f}")
        if len(other_long) > 0:
            other_per_day = other_long['reimbursement'] / other_long['trip_duration_days']
            print(f"Other long trips: {len(other_long)}, avg per-day: ${other_per_day.mean():.2f}")
    
    def identify_outliers(self):
        """Identify potential bugs or edge cases"""
        print("\n" + "="*60)
        print("OUTLIER ANALYSIS - POTENTIAL BUGS/EDGE CASES")
        print("="*60)
        
        # Calculate z-scores for reimbursement amounts
        self.df['reimbursement_zscore'] = np.abs(stats.zscore(self.df['reimbursement']))
        outliers = self.df[self.df['reimbursement_zscore'] > 2.5]  # More than 2.5 standard deviations
        
        print(f"Found {len(outliers)} potential outliers (|z-score| > 2.5):")
        if len(outliers) > 0:
            print("\nTop 10 unusual cases:")
            outlier_display = outliers.nlargest(10, 'reimbursement_zscore')[
                ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement', 'reimbursement_zscore']
            ]
            print(outlier_display.round(2))
    
    def generate_summary(self):
        """Generate summary findings"""
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS")
        print("="*60)
        
        findings = []
        
        # Check if 5-day bonus exists
        if 5 in self.df['trip_duration_days'].values:
            day5_rate = self.df[self.df['trip_duration_days'] == 5]['reimbursement_per_day'].mean()
            other_rate = self.df[self.df['trip_duration_days'] != 5]['reimbursement_per_day'].mean()
            if day5_rate > other_rate * 1.05:  # 5% threshold
                findings.append(f"✓ 5-day bonus confirmed: ${day5_rate:.2f}/day vs ${other_rate:.2f}/day")
        
        # Check mileage tiers
        if len(self.df[self.df['miles_traveled'] <= 100]) > 0:
            low_rate = self.df[self.df['miles_traveled'] <= 100]['mileage_rate'].mean()
            high_rate = self.df[self.df['miles_traveled'] > 100]['mileage_rate'].mean()
            if low_rate > high_rate * 1.1:  # 10% threshold
                findings.append(f"✓ Mileage tier confirmed: ${low_rate:.3f}/mile (≤100mi) vs ${high_rate:.3f}/mile (>100mi)")
        
        # Check rounding bug
        bug_count = len(self.df[self.df['receipt_ends_49'] | self.df['receipt_ends_99']])
        if bug_count > 0:
            findings.append(f"? Rounding bug candidates: {bug_count} cases end in .49 or .99")
        
        if findings:
            print("\nConfirmed patterns:")
            for finding in findings:
                print(finding)
        else:
            print("No clear patterns confirmed yet - need deeper analysis")
        
        print(f"\nNext steps:")
        print("1. Build decision tree model to identify rule structures")
        print("2. Test more specific threshold hypotheses")
        print("3. Analyze interaction effects between variables")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("LEGACY REIMBURSEMENT SYSTEM - REVERSE ENGINEERING ANALYSIS")
        print("="*70)
        
        self.basic_statistics()
        self.test_5day_bonus()
        self.test_mileage_tiers()
        self.test_efficiency_bonus()
        self.test_receipt_patterns()
        self.test_combination_effects()
        self.identify_outliers()
        self.generate_summary()

if __name__ == "__main__":
    analyzer = ReimbursementAnalyzer()
    analyzer.run_full_analysis() 
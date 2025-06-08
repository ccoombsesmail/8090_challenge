#!/usr/bin/env python3
"""
Deep Analysis - Reverse Engineering the Mathematical Formula
Focus on building the exact reimbursement calculation logic
"""

import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DeepReimbursementAnalyzer:
    def __init__(self, data_file='public_cases.json'):
        """Load and prepare data"""
        print("Loading data for deep analysis...")
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Convert to DataFrame with enhanced features
        data_list = []
        for case in self.raw_data:
            row = case['input'].copy()
            row['reimbursement'] = case['expected_output']
            data_list.append(row)
        
        self.df = pd.DataFrame(data_list)
        
        # Enhanced feature engineering based on discoveries
        self.df['miles_per_day'] = self.df['miles_traveled'] / self.df['trip_duration_days']
        self.df['receipts_per_day'] = self.df['total_receipts_amount'] / self.df['trip_duration_days']
        
        # Mileage tiers (major discovery)
        self.df['miles_low'] = np.minimum(self.df['miles_traveled'], 100)  # First 100 miles
        self.df['miles_high'] = np.maximum(self.df['miles_traveled'] - 100, 0)  # Miles over 100
        
        # Receipt ending analysis
        self.df['receipt_cents'] = (self.df['total_receipts_amount'] * 100) % 100
        self.df['receipt_penalty'] = ((self.df['receipt_cents'] == 49) | (self.df['receipt_cents'] == 99)).astype(int)
        
        # Trip length categories
        self.df['trip_short'] = (self.df['trip_duration_days'] <= 3).astype(int)
        self.df['trip_medium'] = ((self.df['trip_duration_days'] >= 4) & (self.df['trip_duration_days'] <= 7)).astype(int)
        self.df['trip_long'] = (self.df['trip_duration_days'] >= 8).astype(int)
        
        print(f"Loaded {len(self.df)} cases with enhanced features")
    
    def analyze_mileage_structure(self):
        """Deep dive into the mileage calculation structure"""
        print("\n" + "="*60)
        print("DEEP ANALYSIS: MILEAGE CALCULATION STRUCTURE")
        print("="*60)
        
        # Test the two-tier hypothesis
        # If reimbursement = base_per_diem + (miles_low * high_rate) + (miles_high * low_rate) + receipt_component
        
        # Look at cases with zero receipts to isolate mileage + per diem
        zero_receipt_cases = self.df[self.df['total_receipts_amount'] < 10].copy()
        
        if len(zero_receipt_cases) > 0:
            print(f"\nAnalyzing {len(zero_receipt_cases)} cases with minimal receipts:")
            
            zero_receipt_cases['estimated_base_per_diem'] = zero_receipt_cases['reimbursement'] - \
                                                          (zero_receipt_cases['miles_low'] * 0.58) - \
                                                          (zero_receipt_cases['miles_high'] * 0.35)
            
            zero_receipt_cases['per_diem_per_day'] = zero_receipt_cases['estimated_base_per_diem'] / zero_receipt_cases['trip_duration_days']
            
            print("\nEstimated per-diem rates by trip length:")
            per_diem_by_length = zero_receipt_cases.groupby('trip_duration_days')['per_diem_per_day'].agg(['count', 'mean', 'std']).round(2)
            print(per_diem_by_length)
            
            # Test if it's a simple linear per diem
            print(f"\nTesting linear per diem hypothesis:")
            base_rate = zero_receipt_cases['per_diem_per_day'].mean()
            print(f"Average per diem rate: ${base_rate:.2f}/day")
            
    def build_decision_tree(self):
        """Build decision tree to identify exact rules"""
        print("\n" + "="*60)
        print("DECISION TREE ANALYSIS")
        print("="*60)
        
        # Prepare features for decision tree
        features = [
            'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
            'miles_per_day', 'receipts_per_day', 'miles_low', 'miles_high',
            'receipt_penalty', 'trip_short', 'trip_medium', 'trip_long'
        ]
        
        X = self.df[features]
        y = self.df['reimbursement']
        
        # Build a shallow tree to see main rules
        tree = DecisionTreeRegressor(
            max_depth=8,  # Keep it interpretable
            min_samples_split=20,  # Avoid overfitting
            min_samples_leaf=10,
            random_state=42
        )
        
        tree.fit(X, y)
        
        # Display the tree rules
        tree_rules = export_text(tree, feature_names=features, max_depth=3)
        print("\nDecision Tree Rules (top 3 levels):")
        print(tree_rules)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': tree.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Test accuracy
        y_pred = tree.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\nDecision Tree Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return tree
        
    def test_formula_hypotheses(self):
        """Test specific mathematical formula hypotheses"""
        print("\n" + "="*60)
        print("TESTING MATHEMATICAL FORMULA HYPOTHESES")
        print("="*60)
        
        # Hypothesis 1: Base per diem + tiered mileage + receipt function
        print("\nHypothesis 1: Base Per Diem + Tiered Mileage + Receipt Function")
        
        # Test different base per diem rates
        base_rates = [100, 120, 150, 200]
        mileage_rates_low = [0.58, 0.60, 0.55, 0.50]
        mileage_rates_high = [0.35, 0.40, 0.30, 0.25]
        
        best_mae = float('inf')
        best_params = None
        
        for base_rate in base_rates:
            for low_rate in mileage_rates_low:
                for high_rate in mileage_rates_high:
                    # Simple formula test
                    predicted = (
                        base_rate * self.df['trip_duration_days'] +  # Base per diem
                        low_rate * self.df['miles_low'] +  # First 100 miles
                        high_rate * self.df['miles_high'] +  # Miles over 100
                        0.5 * self.df['total_receipts_amount']  # 50% receipt reimbursement
                    )
                    
                    mae = mean_absolute_error(self.df['reimbursement'], predicted)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            'base_rate': base_rate,
                            'low_rate': low_rate,
                            'high_rate': high_rate,
                            'receipt_rate': 0.5
                        }
        
        print(f"Best simple formula:")
        print(f"Parameters: {best_params}")
        print(f"Mean Absolute Error: ${best_mae:.2f}")
        
        # Test the best formula
        if best_params:
            predicted_best = (
                best_params['base_rate'] * self.df['trip_duration_days'] +
                best_params['low_rate'] * self.df['miles_low'] +
                best_params['high_rate'] * self.df['miles_high'] +
                best_params['receipt_rate'] * self.df['total_receipts_amount']
            )
            
            # Apply penalty for .49/.99 endings
            penalty_mask = self.df['receipt_penalty'] == 1
            if penalty_mask.sum() > 0:
                print(f"\nTesting receipt penalty for .49/.99 endings...")
                penalty_cases = self.df[penalty_mask]
                penalty_predicted = predicted_best[penalty_mask]
                penalty_actual = penalty_cases['reimbursement']
                
                # Calculate average penalty amount
                penalty_amount = (penalty_predicted - penalty_actual).mean()
                print(f"Average penalty amount: ${penalty_amount:.2f}")
                
                # Apply penalty to predictions
                predicted_best[penalty_mask] -= penalty_amount
                
                # Recalculate MAE
                final_mae = mean_absolute_error(self.df['reimbursement'], predicted_best)
                print(f"MAE after penalty adjustment: ${final_mae:.2f}")
    
    def analyze_receipt_function(self):
        """Deep analysis of receipt reimbursement patterns"""
        print("\n" + "="*60)
        print("RECEIPT REIMBURSEMENT FUNCTION ANALYSIS")
        print("="*60)
        
        # Group by receipt ranges and calculate effective rates
        receipt_bins = np.arange(0, 2500, 100)  # $100 increments
        self.df['receipt_bin'] = pd.cut(self.df['total_receipts_amount'], bins=receipt_bins)
        
        receipt_analysis = self.df.groupby('receipt_bin').agg({
            'total_receipts_amount': ['count', 'mean'],
            'reimbursement': 'mean',
            'trip_duration_days': 'mean',
            'miles_traveled': 'mean'
        }).round(2)
        
        print("\nReceipt reimbursement by amount ranges:")
        print(receipt_analysis.head(15))  # Show first 15 ranges
        
        # Calculate implied receipt reimbursement rate
        # Subtract estimated base + mileage to isolate receipt component
        self.df['estimated_receipt_component'] = (
            self.df['reimbursement'] - 
            (150 * self.df['trip_duration_days']) -  # Estimated base per diem
            (0.58 * self.df['miles_low']) -  # Estimated mileage component
            (0.35 * self.df['miles_high'])
        )
        
        self.df['receipt_rate'] = np.where(
            self.df['total_receipts_amount'] > 0,
            self.df['estimated_receipt_component'] / self.df['total_receipts_amount'],
            0
        )
        
        # Analyze receipt rates by amount
        print("\nImplied receipt reimbursement rates:")
        receipt_rate_analysis = self.df[self.df['total_receipts_amount'] > 50].groupby('receipt_bin')['receipt_rate'].agg(['count', 'mean', 'std']).round(3)
        print(receipt_rate_analysis.head(15))
    
    def test_comprehensive_model(self):
        """Test a comprehensive model incorporating all discoveries"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL TEST")
        print("="*60)
        
        # Use Random Forest to capture complex interactions
        features = [
            'trip_duration_days', 'miles_low', 'miles_high', 
            'total_receipts_amount', 'receipt_penalty'
        ]
        
        X = self.df[features]
        y = self.df['reimbursement']
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        print(f"Random Forest Performance:")
        print(f"Training MAE: ${mean_absolute_error(y_train, y_pred_train):.2f}")
        print(f"Test MAE: ${mean_absolute_error(y_test, y_pred_test):.2f}")
        print(f"Training R²: {r2_score(y_train, y_pred_train):.4f}")
        print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(feature_importance)
        
        return rf
    
    def identify_edge_cases(self):
        """Identify specific edge cases and patterns"""
        print("\n" + "="*60)
        print("EDGE CASE ANALYSIS")
        print("="*60)
        
        # Look at the worst prediction errors
        # Use simple formula as baseline
        predicted_simple = (
            150 * self.df['trip_duration_days'] +  # Base per diem
            0.58 * self.df['miles_low'] +  # First 100 miles
            0.35 * self.df['miles_high'] +  # Miles over 100
            0.5 * self.df['total_receipts_amount']  # Receipt reimbursement
        )
        
        self.df['prediction_error'] = abs(self.df['reimbursement'] - predicted_simple)
        worst_errors = self.df.nlargest(20, 'prediction_error')
        
        print("Top 20 cases with largest prediction errors:")
        print(worst_errors[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                           'reimbursement', 'prediction_error', 'receipt_penalty']].round(2))
        
        # Look for patterns in errors
        print(f"\nError analysis:")
        print(f"Average error: ${self.df['prediction_error'].mean():.2f}")
        print(f"Median error: ${self.df['prediction_error'].median():.2f}")
        print(f"90th percentile error: ${self.df['prediction_error'].quantile(0.9):.2f}")
        
        # Check if penalty cases are among the worst errors
        penalty_cases = self.df[self.df['receipt_penalty'] == 1]
        if len(penalty_cases) > 0:
            print(f"\nReceipt penalty cases (n={len(penalty_cases)}):")
            print(f"Average error: ${penalty_cases['prediction_error'].mean():.2f}")
            print(f"These cases tend to be {'high' if penalty_cases['prediction_error'].mean() > self.df['prediction_error'].mean() else 'low'} error")
    
    def run_deep_analysis(self):
        """Run the complete deep analysis"""
        print("DEEP REIMBURSEMENT SYSTEM ANALYSIS")
        print("="*70)
        
        self.analyze_mileage_structure()
        tree = self.build_decision_tree()
        self.test_formula_hypotheses()
        self.analyze_receipt_function()
        rf = self.test_comprehensive_model()
        self.identify_edge_cases()
        
        return tree, rf

if __name__ == "__main__":
    analyzer = DeepReimbursementAnalyzer()
    tree, rf = analyzer.run_deep_analysis() 
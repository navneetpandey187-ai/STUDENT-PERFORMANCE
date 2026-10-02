"""
eda_walkthrough.py
------------------
Exploratory Data Analysis script for the Student Performance Analyzer.
Executes descriptive statistical checks, generates distribution plots,
and analyzes socio-demographic correlation with academic achievement.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_data, get_data_summary

def run_eda():
    print("=" * 60)
    print("STUDENT PERFORMANCE ANALYZER - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    print(f"\n1. Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("2. Missing Value Check:")
    print(df.isnull().sum())
    
    print("\n3. Statistical Overview:")
    summary = get_data_summary(df)
    for k, v in summary.items():
        print(f"   - {k.replace('_', ' ').title()}: {v}")
        
    print("\n4. Score Distribution Summary:")
    print(df[["math_score", "reading_score", "writing_score", "final_score"]].describe())
    
    print("\n5. Average Final Score by Parental Education:")
    print(df.groupby("parental_education")["final_score"].mean().sort_values(ascending=False))
    
    print("\n6. Average Final Score by Test Prep Course:")
    print(df.groupby("test_prep_course")["final_score"].mean())

    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    run_eda()

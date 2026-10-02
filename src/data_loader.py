"""
data_loader.py
--------------
Handles loading, validation, and feature splitting for the Student Performance Analyzer.
Ensures data integrity and consistent column typing across training and inferencing.
"""

import os
import pandas as pd
from typing import Tuple, List

# Feature schema definitions
NUMERICAL_FEATURES = [
    "study_hours_per_week",
    "attendance_rate",
    "past_failures",
    "sleep_hours",
    "family_support"
]

CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "lunch_type",
    "test_prep_course",
    "school_type",
    "internet_access",
    "extracurricular"
]

TARGET_FEATURE = "final_score"

def get_default_data_path() -> str:
    """Return absolute path to default dataset CSV."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, "data", "student_performance_data.csv")

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load student dataset from file path or fallback to default location.
    
    Parameters:
        file_path (str, optional): Custom file path to student CSV.
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe.
    """
    if file_path is None:
        file_path = get_default_data_path()
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run data_generator.py first.")
        
    df = pd.read_csv(file_path)
    return df

def get_feature_and_target_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract predictor matrix X and target vector y.
    
    Parameters:
        df (pd.DataFrame): Raw or processed dataframe.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Predictor features X and continuous target y.
    """
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    X = df[feature_cols].copy()
    y = df[TARGET_FEATURE].copy()
    return X, y

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Compute key statistical summaries for presentation and dashboards.
    """
    return {
        "total_students": len(df),
        "avg_final_score": round(df["final_score"].mean(), 2),
        "median_final_score": round(df["final_score"].median(), 2),
        "pass_rate_pct": round((df["status"] == "Pass").mean() * 100, 1),
        "avg_attendance": round(df["attendance_rate"].mean(), 1),
        "avg_study_hours": round(df["study_hours_per_week"].mean(), 1),
        "test_prep_completion_rate": round((df["test_prep_course"] == "Completed").mean() * 100, 1)
    }

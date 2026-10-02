"""
test_pipeline.py
----------------
Unit and integration tests for data generation, preprocessing,
model inference, and recommendation generation.
"""

import os
import sys
import unittest
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_generator import generate_student_dataset
from data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_FEATURE
from predict import score_to_grade, generate_academic_insights

class TestStudentPerformanceAnalyzer(unittest.TestCase):
    
    def test_dataset_generation(self):
        """Verify synthetic dataset generator produces correct shape and columns."""
        df = generate_student_dataset(n_samples=50, random_seed=42)
        self.assertEqual(len(df), 50)
        
        expected_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [
            "math_score", "reading_score", "writing_score", TARGET_FEATURE, "grade", "status"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")
            
    def test_score_ranges(self):
        """Verify scores are within valid 0-100 range."""
        df = generate_student_dataset(n_samples=50, random_seed=42)
        self.assertTrue((df["final_score"] >= 0).all() and (df["final_score"] <= 100).all())
        
    def test_grade_mapping(self):
        """Verify grade categorization boundaries."""
        grade_a, risk_a = score_to_grade(95.0)
        self.assertEqual(grade_a, "A")
        
        grade_c, risk_c = score_to_grade(72.0)
        self.assertEqual(grade_c, "C")
        
        grade_f, risk_f = score_to_grade(45.0)
        self.assertEqual(grade_f, "F")
        self.assertIn("At-Risk", risk_f)
        
    def test_insights_generation(self):
        """Verify low attendance and low study hours trigger relevant warning recommendations."""
        student = {
            "attendance_rate": 60.0,
            "study_hours_per_week": 4.0,
            "test_prep_course": "Not Completed",
            "sleep_hours": 5.0,
            "past_failures": 1
        }
        insights = generate_academic_insights(student, predicted_score=52.0)
        warning_titles = [i["title"] for i in insights if i["type"] == "warning"]
        self.assertTrue(any("Attendance" in t for t in warning_titles))
        self.assertTrue(any("Study" in t for t in warning_titles))

if __name__ == "__main__":
    unittest.main()

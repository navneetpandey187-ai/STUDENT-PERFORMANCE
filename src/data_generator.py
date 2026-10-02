"""
data_generator.py
-----------------
Generates a realistic, statistically grounded student performance dataset
based on educational research indicators (study habits, parental education,
attendance, test preparation, and psychological factors).

This dataset mimics real-world exam data (similar to UCI and Kaggle benchmarks)
while providing rich, non-linear signals suitable for both regression and classification.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic yet realistic student performance records.
    
    Parameters:
        n_samples (int): Number of student records to generate.
        random_seed (int): Seed for reproducible data generation.
        
    Returns:
        pd.DataFrame: Processed student performance dataframe.
    """
    np.random.seed(random_seed)
    
    # Demographic & Socio-economic variables
    genders = np.random.choice(["Female", "Male"], size=n_samples, p=[0.51, 0.49])
    
    parental_education_levels = [
        "Some High School",
        "High School",
        "Some College",
        "Associate's Degree",
        "Bachelor's Degree",
        "Master's Degree"
    ]
    parental_weights = [0.15, 0.22, 0.23, 0.20, 0.14, 0.06]
    parental_education = np.random.choice(parental_education_levels, size=n_samples, p=parental_weights)
    
    lunch_types = np.random.choice(["Standard", "Free/Reduced"], size=n_samples, p=[0.65, 0.35])
    test_prep = np.random.choice(["Completed", "Not Completed"], size=n_samples, p=[0.36, 0.64])
    school_type = np.random.choice(["Public", "Private"], size=n_samples, p=[0.78, 0.22])
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])

    # Behavioral & Academic Habits
    # Weekly study hours: Log-normal distribution clipped between 1 and 35 hours
    study_hours = np.clip(np.random.gamma(shape=3.5, scale=3.0, size=n_samples), 1.0, 35.0).round(1)
    
    # Attendance rate: Beta distribution centered around 85-90%
    attendance_rate = (np.random.beta(a=9, b=2, size=n_samples) * 100).clip(40.0, 100.0).round(1)
    
    # Previous academic failures: 0, 1, 2, or 3
    past_failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    
    # Sleep hours per night (Normal distribution around 6.8 hrs)
    sleep_hours = np.clip(np.random.normal(loc=6.8, scale=1.1, size=n_samples), 4.0, 10.0).round(1)
    
    # Peer / Family support score (1 to 5 scale)
    family_support = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.14, 0.28, 0.32, 0.18])

    # Latent academic ability calculation (Base formula with realistic weights)
    parental_edu_bonus_map = {
        "Some High School": -4.0,
        "High School": -1.5,
        "Some College": 1.0,
        "Associate's Degree": 3.0,
        "Bachelor's Degree": 6.5,
        "Master's Degree": 10.0
    }
    edu_effect = np.array([parental_edu_bonus_map[lvl] for lvl in parental_education])
    
    lunch_effect = np.where(lunch_types == "Standard", 4.0, -3.5)
    prep_effect = np.where(test_prep == "Completed", 8.5, -2.0)
    internet_effect = np.where(internet_access == "Yes", 2.0, -4.0)
    extra_effect = np.where(extracurricular == "Yes", 1.5, 0.0)
    
    # Non-linear diminishing returns for study hours
    study_effect = 2.2 * np.sqrt(study_hours * 5.0)
    
    # Attendance penalty: severe drop if attendance < 75%
    attendance_effect = (attendance_rate - 75.0) * 0.45
    attendance_penalty = np.where(attendance_rate < 70, -8.0, 0.0)
    
    failure_penalty = past_failures * -7.5
    sleep_optimum_effect = -2.5 * np.abs(sleep_hours - 7.5)  # optimal sleep ~7.5 hours
    
    # Base capability score (mean around 65)
    latent_score = (
        52.0
        + study_effect
        + attendance_effect
        + attendance_penalty
        + edu_effect
        + lunch_effect
        + prep_effect
        + internet_effect
        + extra_effect
        + failure_penalty
        + sleep_optimum_effect
        + (family_support - 3) * 1.5
    )
    
    # Generate 3 subject scores with domain-specific correlation and noise
    # Math: slightly higher variance, slight male tendency in typical standardized exam datasets
    math_gender_shift = np.where(genders == "Male", 2.0, -2.0)
    math_noise = np.random.normal(0, 7.5, size=n_samples)
    math_score = np.clip(latent_score + math_gender_shift + math_noise, 15, 100).round().astype(int)
    
    # Reading: slight female tendency in standard literature benchmarks
    reading_gender_shift = np.where(genders == "Female", 3.0, -3.0)
    reading_noise = np.random.normal(0, 6.5, size=n_samples)
    reading_score = np.clip(latent_score + reading_gender_shift + reading_noise, 20, 100).round().astype(int)
    
    # Writing: strongly correlated with reading, benefits from study hours
    writing_noise = np.random.normal(0, 5.0, size=n_samples)
    writing_score = np.clip(0.65 * reading_score + 0.35 * latent_score + writing_noise, 18, 100).round().astype(int)
    
    # Overall average score
    final_score = ((math_score + reading_score + writing_score) / 3.0).round(1)
    
    # Grade Categorization
    # Standard grading scale
    def assign_grade(score):
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
            
    grades = [assign_grade(s) for s in final_score]
    pass_fail = ["Pass" if s >= 60 else "Fail" for s in final_score]

    df = pd.DataFrame({
        "gender": genders,
        "parental_education": parental_education,
        "lunch_type": lunch_types,
        "test_prep_course": test_prep,
        "school_type": school_type,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance_rate,
        "past_failures": past_failures,
        "sleep_hours": sleep_hours,
        "family_support": family_support,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
        "final_score": final_score,
        "grade": grades,
        "status": pass_fail
    })
    
    return df

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "student_performance_data.csv")
    
    print("Generating student performance benchmark dataset...")
    df = generate_student_dataset(n_samples=1000, random_seed=42)
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())

if __name__ == "__main__":
    main()

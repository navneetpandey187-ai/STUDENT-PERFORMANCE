"""
train_model.py
--------------
End-to-End Machine Learning Pipeline:
1. Loads student academic & demographic dataset.
2. Constructs robust preprocessing pipeline (ColumnTransformer: OneHotEncoder + StandardScaler).
3. Benchmarks multiple candidate regressors:
   - Linear Regression
   - Ridge Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Evaluates performance across R2, RMSE, and MAE.
5. Performs hyperparameter optimization on the top candidate.
6. Serializes the champion model pipeline and diagnostic metadata for the UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_loader import (
    load_data,
    get_feature_and_target_matrices,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    get_default_data_path
)
from data_generator import generate_student_dataset

def build_preprocessor() -> ColumnTransformer:
    """
    Construct preprocessing ColumnTransformer.
    - Scales numerical features with StandardScaler.
    - Encodes categorical features with OneHotEncoder (handle_unknown='ignore').
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def evaluate_candidate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate candidate models, collecting performance benchmarks.
    """
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n" + "="*70)
    print(f"{'Model':<22} | {'Train R2':<10} | {'Test R2':<10} | {'Test RMSE':<10} | {'Test MAE':<10}")
    print("="*70)
    
    for name, regressor in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        
        pipe.fit(X_train, y_train)
        
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results.append({
            "model_name": name,
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_rmse": round(test_rmse, 3),
            "test_mae": round(test_mae, 3)
        })
        trained_pipelines[name] = pipe
        
        print(f"{name:<22} | {train_r2:<10.4f} | {test_r2:<10.4f} | {test_rmse:<10.3f} | {test_mae:<10.3f}")
        
    print("="*70)
    
    comparison_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
    best_model_name = comparison_df.iloc[0]["model_name"]
    print(f"\nChampion Candidate: {best_model_name} (Test R2: {comparison_df.iloc[0]['test_r2']:.4f})")
    
    return comparison_df, trained_pipelines, best_model_name

def extract_feature_importances(pipeline, preprocessor) -> list:
    """
    Extract feature importances or coefficients from the trained model pipeline.
    """
    regressor = pipeline.named_steps["regressor"]
    
    # Get feature names after one-hot encoding
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    
    importances = []
    if hasattr(regressor, "feature_importances_"):
        raw_vals = regressor.feature_importances_
        for name, val in zip(all_feature_names, raw_vals):
            importances.append({"feature": name, "importance": round(float(val), 4)})
    elif hasattr(regressor, "coef_"):
        raw_vals = np.abs(regressor.coef_)
        total = np.sum(raw_vals) if np.sum(raw_vals) > 0 else 1.0
        normalized = raw_vals / total
        for name, val in zip(all_feature_names, normalized):
            importances.append({"feature": name, "importance": round(float(val), 4)})
            
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances

def train_and_export():
    """
    Full workflow execution for training, hyperparameter tuning, and export.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = get_default_data_path()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Check or generate dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating benchmark data...")
        df = generate_student_dataset(n_samples=1000, random_seed=42)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = load_data(data_path)
        
    print(f"Loaded dataset containing {len(df)} records.")
    
    # 2. Extract features and target
    X, y = get_feature_and_target_matrices(df)
    
    # Train-test split (80/20) with seed for full reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 3. Model benchmarking
    preprocessor = build_preprocessor()
    comparison_df, trained_pipelines, best_name = evaluate_candidate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # 4. Fine-tuning the best model with GridSearchCV
    print(f"\nOptimizing hyperparameters for {best_name}...")
    champion_pipe = trained_pipelines[best_name]
    
    if "Gradient Boosting" in best_name:
        param_grid = {
            "regressor__n_estimators": [100, 150],
            "regressor__learning_rate": [0.03, 0.06, 0.1],
            "regressor__max_depth": [3, 4]
        }
    elif "Random Forest" in best_name:
        param_grid = {
            "regressor__n_estimators": [100, 150],
            "regressor__max_depth": [6, 8],
            "regressor__min_samples_split": [2, 5]
        }
    elif "Ridge" in best_name:
        param_grid = {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
        }
    elif "Linear Regression" in best_name:
        param_grid = {
            "regressor__fit_intercept": [True, False]
        }
    else:
        param_grid = {}
        
    grid_search = GridSearchCV(
        champion_pipe,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    final_pipeline = grid_search.best_estimator_
    print(f"Optimal parameters: {grid_search.best_params_}")
    
    # Final evaluation
    y_test_pred = final_pipeline.predict(X_test)
    final_test_r2 = r2_score(y_test, y_test_pred)
    final_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    final_test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\nFinal Tuned Model Performance on Hold-out Test Set:")
    print(f"  R-Squared (R2)  : {final_test_r2:.4f}")
    print(f"  Root Mean Sq Err: {final_test_rmse:.3f}")
    print(f"  Mean Abs Error  : {final_test_mae:.3f}")
    
    # Feature importances
    feature_importances = extract_feature_importances(final_pipeline, preprocessor)
    
    # 5. Export model artifact
    model_save_path = os.path.join(models_dir, "student_performance_model.pkl")
    joblib.dump(final_pipeline, model_save_path)
    print(f"\nModel pipeline saved to: {model_save_path}")
    
    # Export metadata JSON
    metadata = {
        "model_name": best_name,
        "best_params": grid_search.best_params_,
        "metrics": {
            "test_r2": round(final_test_r2, 4),
            "test_rmse": round(final_test_rmse, 3),
            "test_mae": round(final_test_mae, 3)
        },
        "model_comparison": comparison_df.to_dict(orient="records"),
        "top_features": feature_importances[:12],
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_schema": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        }
    }
    
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata exported to: {meta_path}")

if __name__ == "__main__":
    train_and_export()

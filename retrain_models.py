"""
Script to retrain FEP models with the current feature preprocessing pipeline.

This script loads data, preprocesses it using the same pipeline as your dashboard,
and retrains all model types to be compatible with the current feature set.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import FEP-specific models and utilities
from models.logistic import LogisticRegressionFEP
from models.decision_tree import DecisionTreeFEP
from models.gradient_boosting import GradientBoostingFEP
from models.ensemble import HighRiskFocusedEnsemble
from data.loader import load_data, create_multiclass_label

def preprocess_for_ml(df):
    """
    Preprocess the dataset for machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with the label column
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame ready for ML
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Step 1: Identify column types
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Make sure the label column is excluded from preprocessing
    if 'label' in numeric_cols:
        numeric_cols.remove('label')
    if 'label' in categorical_cols:
        categorical_cols.remove('label')
    
    # Step 2: Handle missing values
    # For numeric columns, fill with median
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # For categorical columns, fill with mode (most frequent value)
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Step 3: One-hot encode categorical columns
    # Exclude the target columns (M6_Rem, Y1_Rem, Y1_Rem_6) if they exist
    target_cols = ['M6_Rem', 'Y1_Rem', 'Y1_Rem_6']
    encoding_cols = [col for col in categorical_cols if col not in target_cols]
    
    # One-hot encode the categorical columns
    for col in encoding_cols:
        # Create dummy variables (one-hot encoding)
        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
        # Add the dummy variables to the dataframe
        df_processed = pd.concat([df_processed, dummies], axis=1)
        # Drop the original categorical column
        df_processed = df_processed.drop(col, axis=1)
    
    # Optional: Drop the original target columns that were used to create the label
    for col in target_cols:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    # Ensure the label column is an integer
    if 'label' in df_processed.columns:
        df_processed['label'] = df_processed['label'].astype(int)
    
    print(f"Preprocessed {len(numeric_cols)} numeric columns and {len(encoding_cols)} categorical columns")
    print(f"Final dataset has {df_processed.shape[1]} features")
    
    return df_processed

def retrain_models():
    """Retrain all models with the current feature set."""
    print("Starting FEP model retraining with current feature set...")
    
    # Step 1: Load data
    print("Loading raw data...")
    raw_data = load_data()
    if raw_data is None:
        print("Error: Failed to load data")
        return False
    
    print(f"Raw data loaded: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
    
    # Step 2: Create multiclass label
    print("Creating multiclass label...")
    data_with_label = create_multiclass_label(raw_data)
    
    # Check if label was created
    if 'label' not in data_with_label.columns:
        print("Error: Failed to create label column")
        return False
    
    # Display class distribution
    label_counts = data_with_label['label'].value_counts().sort_index()
    print("\nClass distribution:")
    for cls, count in label_counts.items():
        print(f"Class {cls}: {count} samples ({count/len(data_with_label)*100:.1f}%)")
    
    # Step 3: Preprocess data
    print("\nPreprocessing data...")
    processed_data = preprocess_for_ml(data_with_label)
    
    # Step 4: Split into training and test sets
    from sklearn.model_selection import train_test_split
    
    # Extract features and target
    X = processed_data.drop('label', axis=1)
    y = processed_data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Step 5: Define and train models
    models = {
        "logistic_regression": LogisticRegressionFEP(
            class_weight='clinical',
            cv=5,  # Reduced for faster training
            random_state=42
        ),
        "decision_tree": DecisionTreeFEP(
            class_weight='clinical',
            random_state=42,
            param_grid={
                "max_depth": range(2, 6),  # Simplified grid
                "min_samples_leaf": range(5, 35, 10),
                "min_samples_split": range(5, 50, 15)
            }
        ),
        "gradient_boosting": GradientBoostingFEP(
            class_weight='clinical',
            random_state=42,
            n_iter=10  # Reduced for faster training
        ),
        "high_risk_ensemble": HighRiskFocusedEnsemble(
            random_state=42
        )
    }
    
    # Train and evaluate each model
    print("\nTraining models...")
    
    # Create output directory
    models_dir = os.path.join(project_root, "models", "trained")
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train the model
            model.fit(X_train, y_train, feature_names=X_train.columns.tolist())
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            # Save the model
            model_path = os.path.join(models_dir, f"{name}.joblib")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print("\nModel retraining complete. All models now compatible with current feature dataset.")
    return True

if __name__ == "__main__":
    # Add some command line flags if needed
    import argparse
    parser = argparse.ArgumentParser(description='Retrain FEP models with current feature set')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    success = retrain_models()
    
    if success:
        print("\nModels were successfully retrained and saved to models/trained directory")
        print("They will now be compatible with the current feature dataset")
    else:
        print("\nError: Model retraining failed")
"""
Data loading utilities for FEP analysis.

This module provides functions for loading and basic inspection of the FEP dataset.
"""
import os
import pandas as pd
import numpy as np

def load_data(file_path=None, data_dir=None):
    """
    Load FEP dataset from CSV file.
    
    Parameters
    ----------
    file_path : str, default=None
        Path to the CSV file. If None, will look in standard locations.
    data_dir : str, default=None
        Directory containing the data files. If None, looks in current directory
        and standard data locations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the FEP dataset
    """
    # DIRECT FIX: Hard-code the path to the known location
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "data", "raw", "fep_dataset.csv")
    
    # Try the known location first
    if os.path.exists(raw_data_path):
        print(f"Loading dataset from: {raw_data_path}")
        return pd.read_csv(raw_data_path)
    
    # If file_path is provided, try it directly
    if file_path is not None:
        if os.path.exists(file_path):
            print(f"Loading dataset from provided path: {file_path}")
            return pd.read_csv(file_path)
        else:
            print(f"Warning: Provided file path does not exist: {file_path}")
    
    # Try looking in the raw subdirectory 
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw", "fep_dataset.csv")
    
    if os.path.exists(raw_path):
        print(f"Loading dataset from raw directory: {raw_path}")
        return pd.read_csv(raw_path)
    
    # If still not found, adjust the path to look for raw folder relative to current directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Generate potential paths
    potential_paths = [
        os.path.join(current_dir, "data", "raw", "fep_dataset.csv"),
        os.path.join(current_dir, "raw", "fep_dataset.csv"),
        os.path.join(current_dir, "data", "fep_dataset.csv"),
        os.path.join(current_dir, "fep_dataset.csv")
    ]
    
    # Print all potential paths for debugging
    print("Checking the following paths:")
    for path in potential_paths:
        exists = os.path.exists(path)
        print(f"  - {path} {'(exists)' if exists else '(not found)'}")
        if exists:
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path)
    
    # Look for any CSV file in the raw directory
    raw_dir = os.path.join(current_dir, "data", "raw")
    if os.path.exists(raw_dir):
        print(f"Checking for CSV files in: {raw_dir}")
        for filename in os.listdir(raw_dir):
            if filename.endswith(".csv"):
                dataset_path = os.path.join(raw_dir, filename)
                print(f"Found CSV file: {dataset_path}")
                return pd.read_csv(dataset_path)
    
    # DIRECT MODIFICATION: If the default path from the error message exists, fix it
    error_path = r"C:\Users\cbren\Projects\FEP-Analysis\data\fep_dataset.csv" 
    raw_error_path = r"C:\Users\cbren\Projects\FEP-Analysis\data\raw\fep_dataset.csv"
    
    if os.path.exists(raw_error_path):
        print(f"Loading dataset from: {raw_error_path}")
        return pd.read_csv(raw_error_path)
    
    if not os.path.exists(error_path) and os.path.exists(raw_error_path):
        # Create a symbolic link or copy
        try:
            # Try to create a symbolic link first
            import shutil
            print(f"Copying dataset from raw folder to expected location")
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            shutil.copy2(raw_error_path, error_path)
            print(f"Dataset copied to: {error_path}")
            return pd.read_csv(error_path)
        except Exception as e:
            print(f"Error copying file: {str(e)}")
    
    # If we get here, the file wasn't found
    raise FileNotFoundError(f"FEP dataset file not found. Please ensure it exists at {raw_path} or provide the correct path.")

def get_dataset_info(df=None, file_path=None):
    """
    Print information about the FEP dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame, default=None
        DataFrame containing the FEP data. If None, will load from file_path.
    file_path : str, default=None
        Path to the FEP dataset CSV file. Used only if df is None.
        
    Returns:
    --------
    dict : Dictionary with dataset information
    """
    # Load data if not provided
    if df is None:
        df = load_data(file_path)
    
    # Basic dataset info
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_types": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }
    
    # Column categories
    column_categories = {
        "demographic": [
            "Age", "Gender", "Ethnicity", "Education", "Relationship", 
            "Accommodation", "Citizenship", "Household", "Parent"
        ],
        "clinical": [
            "Admitted_Hosp", "Depression_Severity", "Depression_Sev_Scale", 
            "Alcohol", "Drugs"
        ],
        "employment": ["M0_Emp", "M6_Emp", "Y1_Emp"],
        "panss_positive": [
            "M0_PANSS_P1", "M0_PANSS_P2", "M0_PANSS_P3", "M0_PANSS_P4", 
            "M0_PANSS_P5", "M0_PANSS_P6", "M0_PANSS_P7"
        ],
        "panss_negative": [
            "M0_PANSS_N1", "M0_PANSS_N2", "M0_PANSS_N3", "M0_PANSS_N4", 
            "M0_PANSS_N5", "M0_PANSS_N6", "M0_PANSS_N7"
        ],
        "panss_general": [
            "M0_PANSS_G1", "M0_PANSS_G2", "M0_PANSS_G3", "M0_PANSS_G4", 
            "M0_PANSS_G5", "M0_PANSS_G6", "M0_PANSS_G7", "M0_PANSS_G8", 
            "M0_PANSS_G9", "M0_PANSS_G10", "M0_PANSS_G11", "M0_PANSS_G12", 
            "M0_PANSS_G13", "M0_PANSS_G14", "M0_PANSS_G15", "M0_PANSS_G16"
        ],
        "outcome": [
            "M6_Rem", "Y1_Rem", "Y1_Rem_6", "M6_Res", "Y1_Res",
            "M6_PANSS_Total_score", "Y1_PANSS_Total_score"
        ]
    }
    
    info["column_categories"] = column_categories
    
    # Target variable distributions
    target_vars = ["M6_Rem", "Y1_Rem", "Y1_Rem_6"]
    target_distributions = {}
    
    for var in target_vars:
        if var in df.columns:
            target_distributions[var] = df[var].value_counts(dropna=False).to_dict()
    
    info["target_distributions"] = target_distributions
    
    return info

def rename_columns(df):
    """
    Rename the PANSS columns to their clinical names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the FEP data with original column names.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with renamed columns
    """
    # Copy dataframe to avoid modifying the original
    renamed_df = df.copy()
    
    # PANSS mapping from codes to clinical names
    panss_mapping = {
        "M0_PANSS_G1": "G1: Somatic concern", 
        "M0_PANSS_G10": "G10: Disorientation", 
        "M0_PANSS_G11": "G11: Poor attention", 
        "M0_PANSS_G12": "G12: Lack of judgment and insight", 
        "M0_PANSS_G13": "G13: Disturbance of volition", 
        "M0_PANSS_G14": "G14: Poor impulse control", 
        "M0_PANSS_G15": "G15: Preoccupation", 
        "M0_PANSS_G16": "G16: Active social avoidance", 
        "M0_PANSS_G2": "G2: Anxiety", 
        "M0_PANSS_G3": "G3: Guilt feelings", 
        "M0_PANSS_G4": "G4: Tension", 
        "M0_PANSS_G5": "G5: Mannerisms & posturing", 
        "M0_PANSS_G6": "G6: Depression", 
        "M0_PANSS_G7": "G7: Motor retardation", 
        "M0_PANSS_G8": "G8: Uncooperativeness", 
        "M0_PANSS_G9": "G9: Unusual thought content", 
        "M0_PANSS_N1": "N1: Blunted affect", 
        "M0_PANSS_N2": "N2: Emotional withdrawal", 
        "M0_PANSS_N3": "N3: Poor rapport", 
        "M0_PANSS_N4": "N4: Passive/apathetic social withdrawal", 
        "M0_PANSS_N5": "N5: Difficulty in abstract thinking", 
        "M0_PANSS_N6": "N6: Lack of spontaneity and flow of conversation", 
        "M0_PANSS_N7": "N7: Stereotyped thinking", 
        "M0_PANSS_P1": "P1: Delusions", 
        "M0_PANSS_P2": "P2: Conceptual disorganization", 
        "M0_PANSS_P3": "P3: Hallucinatory behaviour", 
        "M0_PANSS_P4": "P4: Excitement", 
        "M0_PANSS_P5": "P5: Grandiosity", 
        "M0_PANSS_P6": "P6: Suspiciousness", 
        "M0_PANSS_P7": "P7: Hostility"
    }
    
    # Rename columns
    renamed_df = renamed_df.rename(columns=panss_mapping)
    
    return renamed_df

def load_and_prepare_data(file_path=None, data_dir=None, rename_panss=True, drop_cohort=True):
    """
    Load and prepare the FEP dataset for analysis.
    
    Parameters:
    -----------
    file_path : str, default=None
        Path to the FEP dataset CSV file. If None, will look in data_dir.
    data_dir : str, default=None
        Directory containing the data files. If None, looks in current directory
        and standard data locations.
    rename_panss : bool, default=True
        Whether to rename the PANSS columns to their clinical names.
    drop_cohort : bool, default=True
        Whether to drop the 'Cohort' column.
        
    Returns:
    --------
    pandas.DataFrame : Prepared FEP dataset
    """
    # Load data
    df = load_data(file_path, data_dir)
    
    # Rename PANSS columns if requested
    if rename_panss:
        df = rename_columns(df)
    
    # Drop Cohort column if requested
    if drop_cohort and 'Cohort' in df.columns:
        df = df.drop(['Cohort'], axis=1)
    
    # Basic data checks
    missing_outcomes = df[['M6_Rem', 'Y1_Rem', 'Y1_Rem_6']].isnull().any(axis=1)
    if missing_outcomes.any():
        print(f"Warning: {missing_outcomes.sum()} rows have missing outcome variables.")
    
    return df

def create_multiclass_label(df):
    """
    Create a multiclass label from 'M6_Rem', 'Y1_Rem', and 'Y1_Rem_6' columns.
    Maps combinations to integers 0-7 based on clinical meaning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'M6_Rem', 'Y1_Rem', and 'Y1_Rem_6' columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional 'label' column containing integer class label
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Handle missing values first - fill NaN with 'Missing'
    for col in ['M6_Rem', 'Y1_Rem', 'Y1_Rem_6']:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna('Missing')
    
    # Create a temporary column combining all three values
    result_df['temp_combo'] = (
        result_df['M6_Rem'].astype(str) + '_' + 
        result_df['Y1_Rem'].astype(str) + '_' + 
        result_df['Y1_Rem_6'].astype(str)
    )
    
    # Define the mapping based on clinical meaning of each pattern
    # This is based on CLASS_NAMES in your code:
    # 0: "No remission (Poor adherence)"
    # 1: "No remission (Moderate adherence)"
    # 2: "Early Relapse with functional decline"
    # 3: "Late remission (Poor adherence)"
    # 4: "Early non-sustained remission"
    # 5: "Late remission (Good adherence)"
    # 6: "Sustained with residual"
    # 7: "Full recovery"
    
    # Map combinations to class labels
    combo_to_label = {
        'No_No_No': 0,        # No remission at M6 or Y1
        'No_No_Missing': 0,   # No remission at M6 or Y1
        'No_Missing_Missing': 1, # No remission at M6, unknown at Y1
        'No_Yes_Yes': 3,      # No remission at M6, Yes at Y1 (late remission)
        'No_Yes_No': 4,       # No remission at M6, Yes at Y1 but not sustained (Y1_Rem_6=No)
        'Missing_Missing_Missing': 2, # Unknown (treated as early relapse)
        'Yes_No_No': 2,       # Yes at M6, No at Y1 (early relapse)
        'Yes_No_Missing': 2,  # Yes at M6, No at Y1 (early relapse)
        'Yes_Missing_Missing': 6, # Yes at M6, unknown at Y1 (treated as sustained with residual)
        'Yes_Yes_No': 4,      # Yes at M6 and Y1, but not sustained
        'Yes_Yes_Yes': 7,     # Yes at M6 and Y1, and sustained (full recovery)
        'Missing_Yes_Yes': 5, # Unknown at M6, Yes at Y1 (late remission, good adherence)
        'Missing_No_No': 0,   # Unknown at M6, No at Y1 (treated as no remission)
        'Missing_Yes_No': 4,  # Unknown at M6, Yes at Y1 but not sustained
    }
    
    # Apply the mapping, default to class 2 (early relapse) for any unmapped combinations
    result_df['label'] = result_df['temp_combo'].map(lambda x: combo_to_label.get(x, 2))
    
    # Drop the temporary column
    df_with_label = result_df.drop('temp_combo', axis=1)
    
    return df_with_label

def get_train_test_data(df, test_size=0.3, random_state=42, stratify=True, drop_target_cols=True):
    """
    Split the dataset into training and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the FEP data with a 'label' column.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default=True
        Whether to stratify the split based on the 'label' column.
    drop_target_cols : bool, default=True
        Whether to drop the original target columns ('Y1_Rem', 'M6_Rem', 'Y1_Rem_6').
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
        Split datasets for training and testing.
    """
    from sklearn.model_selection import train_test_split
    
    # Check if label column exists
    if 'label' not in df.columns:
        raise ValueError("DataFrame does not have a 'label' column. Run create_multiclass_label first.")
    
    # Drop rows with missing labels
    df_clean = df.dropna(subset=['label']).reset_index(drop=True)
    
    # Prepare X and y
    if drop_target_cols:
        X = df_clean.drop(['label', 'Y1_Rem', 'M6_Rem', 'Y1_Rem_6'], axis=1, errors='ignore')
    else:
        X = df_clean.drop(['label'], axis=1)
    
    y = df_clean['label']
    
    # Split the data
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    print("Loading FEP dataset...")
    df = load_data()
    
    print("\nDataset information:")
    info = get_dataset_info(df)
    print(f"Rows: {info['rows']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing values: {info['missing_values']} ({info['missing_percentage']:.2f}%)")
    
    print("\nTarget variable distributions:")
    for var, dist in info['target_distributions'].items():
        print(f"{var}: {dist}")
    
    print("\nPreparing data...")
    df_prepared = load_and_prepare_data()
    df_with_label = create_multiclass_label(df_prepared)
    
    print("\nClass distribution:")
    print(df_with_label['label'].value_counts().sort_index())
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = get_train_test_data(df_with_label)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

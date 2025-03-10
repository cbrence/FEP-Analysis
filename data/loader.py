"""
Data loading utilities for FEP analysis.

This module provides functions for loading and basic inspection of the FEP dataset.
"""
import os
import pandas as pd
import numpy as np

def load_data(file_path="./data/raw/fep_dataset.csv"):
    """
    Load FEP dataset from CSV file.
    
    Parameters
    ----------
    file_path : str, default="./data/raw/fep_dataset.csv"
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the FEP dataset
    
    Notes:
    ------
    The FEP dataset contains 162 rows and 56 columns including:
    - Demographic information (Age, Gender, Ethnicity, etc.)
    - Clinical information (Admitted_Hosp, Depression_Severity, etc.)
    - PANSS scores (P1-P7, N1-N7, G1-G16)
    - Outcome measures (M6_Rem, Y1_Rem, Y1_Rem_6, etc.)


    """
    import pandas as pd
    return pd.read_csv(file_path)

          
    # If file_path is provided, use it directly
    if file_path is not None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FEP dataset file not found at: {file_path}")
        return pd.read_csv(file_path)
    
    # Otherwise, search in data_dir or standard locations
    search_paths = []
    
    # Add data_dir if provided
    if data_dir is not None:
        search_paths.append(os.path.join(data_dir, "fep_dataset.csv"))
    
    # Add current directory
    search_paths.append("fep_dataset.csv")
    
    # Add standard data locations
    search_paths.extend([
        os.path.join("data", "fep_dataset.csv"),
        os.path.join("fep_analysis", "data", "fep_dataset.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fep_dataset.csv")
    ])
    
    # Try each path
    for path in search_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # If we get here, the file wasn't found
    raise FileNotFoundError(f"FEP dataset file not found. Searched in: {search_paths}")

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
    Create a multiclass label based on remission patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'Y1_Rem', 'M6_Rem', 'Y1_Rem_6'.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with an additional 'label' column
    """
    # Create a copy to avoid modifying the original
    df_with_label = df.copy()
    
    # Ensure the remission columns are binary (0/1)
    for col in ['Y1_Rem', 'M6_Rem', 'Y1_Rem_6']:
        if col in df.columns:
            if df[col].dtype == object:
                # Convert Yes/No to 1/0
                df_with_label[col] = df_with_label[col].map({'Y': 1, 'N': 0})
    
    # Create the multiclass label
    labels = []
    
    for idx, row in df_with_label.iterrows():
        if pd.isna(row['Y1_Rem']) or pd.isna(row['M6_Rem']) or pd.isna(row['Y1_Rem_6']):
            labels.append(np.nan)
        elif row['Y1_Rem'] == 0 and row['M6_Rem'] == 0 and row['Y1_Rem_6'] == 0:
            labels.append(0)  # No remission at any point
        elif row['Y1_Rem'] == 1 and row['M6_Rem'] == 1 and row['Y1_Rem_6'] == 1:
            labels.append(1)  # Sustained remission
        elif row['Y1_Rem'] == 1 and row['M6_Rem'] == 0 and row['Y1_Rem_6'] == 0:
            labels.append(2)  # Late remission only
        elif row['Y1_Rem'] == 1 and row['M6_Rem'] == 1 and row['Y1_Rem_6'] == 0:
            labels.append(3)  # Early remission, not sustained
        elif row['Y1_Rem'] == 1 and row['M6_Rem'] == 0 and row['Y1_Rem_6'] == 1:
            labels.append(4)  # Late sustained remission
        elif row['Y1_Rem'] == 0 and row['M6_Rem'] == 1 and row['Y1_Rem_6'] == 0:
            labels.append(5)  # Early remission only
        elif row['Y1_Rem'] == 0 and row['M6_Rem'] == 1 and row['Y1_Rem_6'] == 1:
            labels.append(6)  # Early sustained, late relapse
        elif row['Y1_Rem'] == 0 and row['M6_Rem'] == 0 and row['Y1_Rem_6'] == 1:
            labels.append(7)  # Sustained remission after Y1
        else:
            labels.append(np.nan)  # Should not happen if data is clean
    
    df_with_label['label'] = labels
    
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

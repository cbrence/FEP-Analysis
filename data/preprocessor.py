"""
Data preprocessing utilities for FEP analysis.

This module provides functions for cleaning, transforming, and preparing
the FEP dataset for analysis.
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

def clean_data(df):
    """
    Remove unnecessary columns and rows with missing outcome data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw FEP dataset.
        
    Returns:
    --------
    pandas.DataFrame : Cleaned dataset.
    """
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Drop 'Cohort' column if it exists
    if 'Cohort' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(['Cohort'], axis=1)
    
    # Drop rows with missing outcome values
    df_cleaned = df_cleaned.dropna(subset=['M6_Rem', 'Y1_Rem', 'Y1_Rem_6'])
    
    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def impute_missing_values(df):
    """
    Impute missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with missing values.
        
    Returns:
    --------
    pandas.DataFrame : Dataset with imputed values.
    """
    # Make a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Identify numerical and categorical variables
    numerical_cols = [col for col in df_imputed.columns if df_imputed[col].dtype == float]
    categorical_cols = [col for col in df_imputed.columns if col not in numerical_cols]
    
    # Scale numerical features for KNN imputation
    df_num = df_imputed[numerical_cols].copy()
    
    # Only standardize if there are non-null values
    for col in numerical_cols:
        if df_num[col].notna().sum() > 0:
            mean_val = df_num[col].mean()
            std_val = df_num[col].std()
            if std_val > 0:  # Avoid division by zero
                df_num[col] = (df_num[col] - mean_val) / std_val
    
    # Impute numerical features using KNN
    imputer = KNNImputer(n_neighbors=5)
    df_num_imputed = pd.DataFrame(
        imputer.fit_transform(df_num),
        columns=numerical_cols,
        index=df_num.index
    )
    
    # Impute categorical features using mode
    df_cat = df_imputed[categorical_cols].copy()
    df_cat_imputed = df_cat.apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
    
    # Combine imputed data
    df_imputed = pd.concat([df_num_imputed, df_cat_imputed], axis=1)
    
    return df_imputed

def encode_target_variables(df):
    """
    Encode target variables and create multiclass label.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical target variables.
        
    Returns:
    --------
    pandas.DataFrame : Dataset with encoded target variables.
    """
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # List of target columns
    target_cols = ['M6_Rem', 'Y1_Rem', 'Y1_Rem_6']
    
    # Check if target columns exist and need encoding
    if all(col in df_encoded.columns for col in target_cols):
        if df_encoded[target_cols[0]].dtype == object:  # If string values
            # Convert Yes/No to 1/0
            for col in target_cols:
                df_encoded[col] = df_encoded[col].map({'Y': 1, 'N': 0})
        else:
            # Use ordinal encoder for numerical encoding
            encoder = OrdinalEncoder()
            df_encoded[target_cols] = encoder.fit_transform(df_encoded[target_cols])
    
    # Create multiclass label
    label = []
    for i in df_encoded.index:
        if df_encoded.loc[i, 'Y1_Rem'] == 0 and df_encoded.loc[i, 'M6_Rem'] == 0 and df_encoded.loc[i, 'Y1_Rem_6'] == 0:
            label.append(0)  # No remission at any point
        elif df_encoded.loc[i, 'Y1_Rem'] == 1 and df_encoded.loc[i, 'M6_Rem'] == 1 and df_encoded.loc[i, 'Y1_Rem_6'] == 1:
            label.append(1)  # Sustained remission
        elif df_encoded.loc[i, 'Y1_Rem'] == 1 and df_encoded.loc[i, 'M6_Rem'] == 0 and df_encoded.loc[i, 'Y1_Rem_6'] == 0:
            label.append(2)  # Late remission only
        elif df_encoded.loc[i, 'Y1_Rem'] == 1 and df_encoded.loc[i, 'M6_Rem'] == 1 and df_encoded.loc[i, 'Y1_Rem_6'] == 0:
            label.append(3)  # Early remission, not sustained
        elif df_encoded.loc[i, 'Y1_Rem'] == 1 and df_encoded.loc[i, 'M6_Rem'] == 0 and df_encoded.loc[i, 'Y1_Rem_6'] == 1:
            label.append(4)  # Late sustained remission
        elif df_encoded.loc[i, 'Y1_Rem'] == 0 and df_encoded.loc[i, 'M6_Rem'] == 1 and df_encoded.loc[i, 'Y1_Rem_6'] == 0:
            label.append(5)  # Early remission only
        elif df_encoded.loc[i, 'Y1_Rem'] == 0 and df_encoded.loc[i, 'M6_Rem'] == 1 and df_encoded.loc[i, 'Y1_Rem_6'] == 1:
            label.append(6)  # Early sustained, late relapse
        elif df_encoded.loc[i, 'Y1_Rem'] == 0 and df_encoded.loc[i, 'M6_Rem'] == 0 and df_encoded.loc[i, 'Y1_Rem_6'] == 1:
            label.append(7)  # Sustained remission after Y1
    
    # Add label to dataframe
    df_encoded['label'] = label
    
    return df_encoded

def preprocess_pipeline(df, drop_original_targets=False):
    """
    Run complete preprocessing pipeline on FEP dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw FEP dataset.
    drop_original_targets : bool, default=False
        Whether to drop the original target columns after creating the label.
        
    Returns:
    --------
    pandas.DataFrame : Fully preprocessed dataset.
    """
    # Clean data (remove unnecessary columns and rows with missing outcomes)
    df_cleaned = clean_data(df)
    
    # Impute missing values
    df_imputed = impute_missing_values(df_cleaned)
    
    # Encode target variables and create multiclass label
    df_encoded = encode_target_variables(df_imputed)
    
    # Optionally drop original target columns
    if drop_original_targets:
        df_encoded = df_encoded.drop(['M6_Rem', 'Y1_Rem', 'Y1_Rem_6'], axis=1)
    
    return df_encoded
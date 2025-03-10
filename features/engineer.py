"""
Feature engineering utilities for FEP analysis.

This module provides functions for transforming and engineering features
from the FEP dataset to improve model performance.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Setup logging
logger = logging.getLogger(__name__)

class PANSSFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform PANSS features to address skewness and create derived features.
    
    This transformer applies appropriate transformations to PANSS features
    based on their distributions and creates clinically relevant derived features.
    """
    
    def __init__(self, transform_method='cbrt', create_derived_features=True):
        """
        Initialize the PANSS feature transformer.
        
        Parameters:
        -----------
        transform_method : str, default='cbrt'
            Method to transform skewed features:
            - 'cbrt': Cube root transformation (less aggressive than log)
            - 'log': Natural logarithm (after adding 1)
            - 'sqrt': Square root transformation
            - None: No transformation
        create_derived_features : bool, default=True
            Whether to create derived features from PANSS items.
        """
        self.transform_method = transform_method
        self.create_derived_features = create_derived_features
        self.skewed_features_ = None
    
    def fit(self, X, y=None):
        """
        Identify skewed features based on data distribution.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with PANSS features.
        y : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Verify that input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Identify PANSS features
        panss_cols = []
        for col in X.columns:
            if any(col.startswith(prefix) for prefix in ['P', 'N', 'G']) and ':' in col:
                panss_cols.append(col)
        
        if not panss_cols:
            panss_cols = []
            for col in X.columns:
                if any(col.startswith(prefix) for prefix in ['M0_PANSS_P', 'M0_PANSS_N', 'M0_PANSS_G']):
                    panss_cols.append(col)
        
        logger.info(f"Identified {len(panss_cols)} PANSS features")
        
        # Identify skewed features (skewness > 0.5)
        self.skewed_features_ = []
        for col in panss_cols:
            if col in X.columns:
                skewness = X[col].skew()
                if skewness > 0.5:
                    self.skewed_features_.append(col)
        
        logger.info(f"Identified {len(self.skewed_features_)} skewed PANSS features")
        
        return self
    
    def transform(self, X):
        """
        Transform PANSS features and create derived features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with PANSS features.
            
        Returns:
        --------
        pandas.DataFrame : Transformed data.
        """
        # Verify that input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Make a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Apply transformations to skewed features
        if self.transform_method and self.skewed_features_:
            for col in self.skewed_features_:
                if col in X_transformed.columns:
                    if self.transform_method == 'cbrt':
                        X_transformed[col] = np.cbrt(X_transformed[col])
                    elif self.transform_method == 'log':
                        X_transformed[col] = np.log1p(X_transformed[col])
                    elif self.transform_method == 'sqrt':
                        X_transformed[col] = np.sqrt(X_transformed[col])
        
        # Create derived features if requested
        if self.create_derived_features:
            # 1. PANSS Subscale Scores
            panss_p_cols = [col for col in X_transformed.columns if (col.startswith('P') and ':' in col) or col.startswith('M0_PANSS_P')]
            panss_n_cols = [col for col in X_transformed.columns if (col.startswith('N') and ':' in col) or col.startswith('M0_PANSS_N')]
            panss_g_cols = [col for col in X_transformed.columns if (col.startswith('G') and ':' in col) or col.startswith('M0_PANSS_G')]
            
            if panss_p_cols:
                X_transformed['PANSS_P_Total'] = X_transformed[panss_p_cols].sum(axis=1)
            
            if panss_n_cols:
                X_transformed['PANSS_N_Total'] = X_transformed[panss_n_cols].sum(axis=1)
            
            if panss_g_cols:
                X_transformed['PANSS_G_Total'] = X_transformed[panss_g_cols].sum(axis=1)
            
            # 2. PANSS Factor Scores (based on clinical groupings)
            # Positive factor
            pos_items = [col for col in X_transformed.columns if
                         any(item in col for item in ['P1:', 'P3:', 'P5:', 'P6:', 'G9:']) or
                         any(item in col for item in ['M0_PANSS_P1', 'M0_PANSS_P3', 'M0_PANSS_P5', 'M0_PANSS_P6', 'M0_PANSS_G9'])]
            if pos_items:
                X_transformed['PANSS_Factor_Positive'] = X_transformed[pos_items].mean(axis=1)
            
            # Negative factor
            neg_items = [col for col in X_transformed.columns if
                         any(item in col for item in ['N1:', 'N2:', 'N3:', 'N4:', 'N6:', 'G7:']) or
                         any(item in col for item in ['M0_PANSS_N1', 'M0_PANSS_N2', 'M0_PANSS_N3', 'M0_PANSS_N4', 'M0_PANSS_N6', 'M0_PANSS_G7'])]
            if neg_items:
                X_transformed['PANSS_Factor_Negative'] = X_transformed[neg_items].mean(axis=1)
            
            # Disorganized factor
            dis_items = [col for col in X_transformed.columns if
                         any(item in col for item in ['P2:', 'N5:', 'G5:', 'G10:', 'G11:']) or
                         any(item in col for item in ['M0_PANSS_P2', 'M0_PANSS_N5', 'M0_PANSS_G5', 'M0_PANSS_G10', 'M0_PANSS_G11'])]
            if dis_items:
                X_transformed['PANSS_Factor_Disorganized'] = X_transformed[dis_items].mean(axis=1)
            
            # Excitement factor
            exc_items = [col for col in X_transformed.columns if
                         any(item in col for item in ['P4:', 'P7:', 'G8:', 'G14:']) or
                         any(item in col for item in ['M0_PANSS_P4', 'M0_PANSS_P7', 'M0_PANSS_G8', 'M0_PANSS_G14'])]
            if exc_items:
                X_transformed['PANSS_Factor_Excitement'] = X_transformed[exc_items].mean(axis=1)
            
            # Anxiety/depression factor
            anx_items = [col for col in X_transformed.columns if
                         any(item in col for item in ['G1:', 'G2:', 'G3:', 'G4:', 'G6:']) or
                         any(item in col for item in ['M0_PANSS_G1', 'M0_PANSS_G2', 'M0_PANSS_G3', 'M0_PANSS_G4', 'M0_PANSS_G6'])]
            if anx_items:
                X_transformed['PANSS_Factor_Anxiety_Depression'] = X_transformed[anx_items].mean(axis=1)
            
            # 3. Positive-to-Negative Ratio
            if 'PANSS_P_Total' in X_transformed.columns and 'PANSS_N_Total' in X_transformed.columns:
                # Add small constant to avoid division by zero
                X_transformed['PANSS_P_N_Ratio'] = X_transformed['PANSS_P_Total'] / (X_transformed['PANSS_N_Total'] + 0.001)
        
        logger.info(f"Transformed PANSS features: original={X.shape[1]}, new={X_transformed.shape[1]}")
        
        return X_transformed

def transform_skewed_features(df, skewness_threshold=0.5, transform_method='cbrt'):
    """
    Apply appropriate transformations to skewed numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing numerical features.
    skewness_threshold : float, default=0.5
        Threshold for considering a feature as skewed.
    transform_method : str, default='cbrt'
        Transformation method:
        - 'cbrt': Cube root transformation
        - 'log': Natural logarithm (adding 1 first)
        - 'sqrt': Square root transformation
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with transformed features.
    """
    # Make a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Apply transformations to skewed numerical features
    for col in numerical_cols:
        skewness = df[col].skew()
        if abs(skewness) > skewness_threshold:
            logger.debug(f"Transforming skewed feature {col} (skewness={skewness:.2f})")
            
            if transform_method == 'cbrt':
                df_transformed[col] = np.cbrt(df[col])
            elif transform_method == 'log':
                # Add 1 to handle zeros
                df_transformed[col] = np.log1p(df[col])
            elif transform_method == 'sqrt':
                df_transformed[col] = np.sqrt(df[col])
    
    return df_transformed

def one_hot_encode_features(df, categorical_features=None, drop_first=True):
    """
    One-hot encode categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing categorical features.
    categorical_features : list, default=None
        List of categorical features to encode. If None, encodes all object and category columns.
    drop_first : bool, default=True
        Whether to drop the first category for each feature to avoid multicollinearity.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with one-hot encoded features.
    """
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Identify categorical columns if not specified
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    # Filter to only include columns that exist
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    if not categorical_features:
        logger.warning("No categorical features found for one-hot encoding")
        return df_encoded
    
    logger.info(f"One-hot encoding {len(categorical_features)} categorical features")
    
    # Create one-hot encoder
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse=False)
    
    # Apply one-hot encoding
    encoded_data = encoder.fit_transform(df_encoded[categorical_features])
    
    # Create DataFrame with encoded data
    feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
    
    # Concat with original data and drop original categorical columns
    df_encoded = pd.concat([df_encoded.drop(categorical_features, axis=1), encoded_df], axis=1)
    
    logger.info(f"One-hot encoding created {len(feature_names)} new features")
    
    return df_encoded

def create_interaction_features(df, feature_pairs=None, numerical_only=False):
    """
    Create interaction features between pairs of features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing features.
    feature_pairs : list of tuples, default=None
        List of (feature1, feature2) pairs to create interactions for.
        If None, creates interactions for a subset of important features.
    numerical_only : bool, default=False
        Whether to only include numerical features in interactions.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with added interaction features.
    """
    # Make a copy to avoid modifying the original
    df_with_interactions = df.copy()
    
    # Identify numerical columns if needed
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # If no feature pairs specified, use default pairs
    if feature_pairs is None:
        # Use predefined pairs if available in settings
        if hasattr(settings, 'FEATURE_INTERACTION_PAIRS') and settings.FEATURE_INTERACTION_PAIRS:
            feature_pairs = settings.FEATURE_INTERACTION_PAIRS
        else:
            # Default to some PANSS interactions
            panss_p_cols = [col for col in df.columns if (col.startswith('P') and ':' in col) or col.startswith('M0_PANSS_P')]
            panss_n_cols = [col for col in df.columns if (col.startswith('N') and ':' in col) or col.startswith('M0_PANSS_N')]
            
            # Create a subset of 2-3 features from each for interactions
            if len(panss_p_cols) >= 2 and len(panss_n_cols) >= 2:
                key_p_cols = panss_p_cols[:3]  # First 3 positive symptoms
                key_n_cols = panss_n_cols[:3]  # First 3 negative symptoms
                
                # Create pairs across positive and negative symptoms
                feature_pairs = []
                for p_col in key_p_cols:
                    for n_col in key_n_cols:
                        feature_pairs.append((p_col, n_col))
    
    if not feature_pairs:
        logger.warning("No feature pairs specified for interaction features")
        return df_with_interactions
    
    # Filter to only include valid columns
    valid_feature_pairs = []
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Check if they're numerical if required
            if numerical_only:
                if feat1 in numerical_cols and feat2 in numerical_cols:
                    valid_feature_pairs.append((feat1, feat2))
            else:
                valid_feature_pairs.append((feat1, feat2))
    
    # Create interaction features
    for feat1, feat2 in valid_feature_pairs:
        interaction_name = f"Interaction_{feat1}_{feat2}"
        
        # Create product for numerical features
        if feat1 in numerical_cols and feat2 in numerical_cols:
            df_with_interactions[interaction_name] = df[feat1] * df[feat2]
        else:
            # For categorical, create a new combined feature
            df_with_interactions[interaction_name] = df[feat1].astype(str) + "_" + df[feat2].astype(str)
    
    logger.info(f"Created {len(valid_feature_pairs)} interaction features")
    
    return df_with_interactions

def scale_features(df, method='standard', robust_for_outliers=True):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing numerical features.
    method : str, default='standard'
        Scaling method:
        - 'standard': StandardScaler (zero mean, unit variance)
        - 'robust': RobustScaler (median and IQR)
        - 'minmax': MinMaxScaler (0 to 1 range)
    robust_for_outliers : bool, default=True
        Whether to use RobustScaler for features with outliers.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with scaled features.
    """
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        logger.warning("No numerical features found for scaling")
        return df_scaled
    
    # Determine which features have outliers
    if robust_for_outliers:
        features_with_outliers = []
        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            
            # Check if there are values outside the bounds
            if (df[col] > upper_bound).any() or (df[col] < lower_bound).any():
                features_with_outliers.append(col)
        
        # Remove outlier features from standard scaling list
        standard_scale_cols = [col for col in numerical_cols if col not in features_with_outliers]
        robust_scale_cols = features_with_outliers
    else:
        standard_scale_cols = numerical_cols
        robust_scale_cols = []
    
    # Apply standard scaling
    if standard_scale_cols:
        if method == 'standard':
            scaler = StandardScaler()
            df_scaled[standard_scale_cols] = scaler.fit_transform(df[standard_scale_cols])
        elif method == 'robust':
            scaler = RobustScaler()
            df_scaled[standard_scale_cols] = scaler.fit_transform(df[standard_scale_cols])
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df_scaled[standard_scale_cols] = scaler.fit_transform(df[standard_scale_cols])
    
    # Apply robust scaling to features with outliers
    if robust_scale_cols:
        robust_scaler = RobustScaler()
        df_scaled[robust_scale_cols] = robust_scaler.fit_transform(df[robust_scale_cols])
    
    logger.info(f"Scaled {len(numerical_cols)} numerical features")
    
    return df_scaled

def engineer_features(df, apply_transformations=True, create_interactions=True, 
                     apply_scaling=True, one_hot_encode=True):
    """
    Apply complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw input data.
    apply_transformations : bool, default=True
        Whether to apply transformations to skewed features.
    create_interactions : bool, default=True
        Whether to create interaction features.
    apply_scaling : bool, default=True
        Whether to scale numerical features.
    one_hot_encode : bool, default=True
        Whether to one-hot encode categorical features.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with engineered features.
    """
    logger.info("Starting feature engineering pipeline")
    
    # Make a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # 1. Transform PANSS features
    panss_transformer = PANSSFeatureTransformer()
    df_engineered = panss_transformer.fit_transform(df_engineered)
    
    # 2. Transform skewed features
    if apply_transformations:
        df_engineered = transform_skewed_features(df_engineered)
    
    # 3. Create interaction features
    if create_interactions:
        df_engineered = create_interaction_features(df_engineered)
    
    # 4. One-hot encode categorical features
    if one_hot_encode:
        df_engineered = one_hot_encode_features(df_engineered)
    
    # 5. Scale features
    if apply_scaling:
        df_engineered = scale_features(df_engineered)
    
    logger.info(f"Feature engineering complete: {df.shape[1]} original features, {df_engineered.shape[1]} engineered features")
    
    return df_engineered

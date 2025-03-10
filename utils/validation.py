"""
Input validation utilities for FEP analysis.

This module provides functions for validating inputs to ensure data integrity
and proper function of models and analysis pipelines.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import re
import warnings
import logging


def validate_dataframe(df: pd.DataFrame,
                      required_columns: Optional[List[str]] = None,
                      column_types: Optional[Dict[str, type]] = None,
                      no_missing: Optional[List[str]] = None,
                      raise_error: bool = True,
                      logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate a pandas DataFrame against a set of requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : Optional[List[str]], default=None
        List of columns that must be present
    column_types : Optional[Dict[str, type]], default=None
        Dictionary mapping column names to expected types
    no_missing : Optional[List[str]], default=None
        List of columns that should not have missing values
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        error_messages.append("DataFrame is empty or None")
        if logger:
            logger.error("DataFrame is empty or None")
        if raise_error:
            raise ValueError("DataFrame is empty or None")
        return False, error_messages
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            is_valid = False
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check column types
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                # For numeric types, allow conversion
                if expected_type in (int, float) and pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # For string types, check if it's string-like
                if expected_type == str and pd.api.types.is_string_dtype(df[col]):
                    continue
                
                # For datetime, check if it's datetime-like
                if expected_type == pd.Timestamp and pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                
                # For categorical types
                if expected_type == 'category' and pd.api.types.is_categorical_dtype(df[col]):
                    continue
                
                # Check if types match
                if not isinstance(df[col].dtype, expected_type) and df[col].dtype != expected_type:
                    is_valid = False
                    error_msg = f"Column '{col}' has type {df[col].dtype}, expected {expected_type}"
                    error_messages.append(error_msg)
                    if logger:
                        logger.error(error_msg)
    
    # Check for missing values
    if no_missing:
        for col in no_missing:
            if col in df.columns and df[col].isna().any():
                is_valid = False
                missing_count = df[col].isna().sum()
                error_msg = f"Column '{col}' has {missing_count} missing values"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_feature_names(feature_names: List[str],
                         pattern: Optional[str] = None,
                         forbidden_names: Optional[List[str]] = None,
                         required_names: Optional[List[str]] = None,
                         raise_error: bool = True,
                         logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate feature names against a set of requirements.
    
    Parameters
    ----------
    feature_names : List[str]
        List of feature names to validate
    pattern : Optional[str], default=None
        Regular expression pattern that feature names should match
    forbidden_names : Optional[List[str]], default=None
        List of names that should not be in feature_names
    required_names : Optional[List[str]], default=None
        List of names that must be in feature_names
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if feature_names is empty
    if not feature_names:
        error_messages.append("Feature names list is empty")
        if logger:
            logger.error("Feature names list is empty")
        if raise_error:
            raise ValueError("Feature names list is empty")
        return False, error_messages
    
    # Check for duplicates
    duplicates = set([name for name in feature_names if feature_names.count(name) > 1])
    if duplicates:
        is_valid = False
        error_msg = f"Duplicate feature names found: {', '.join(duplicates)}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check against pattern
    if pattern:
        invalid_names = [name for name in feature_names if not re.match(pattern, name)]
        if invalid_names:
            is_valid = False
            error_msg = f"Features don't match pattern '{pattern}': {', '.join(invalid_names)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check for forbidden names
    if forbidden_names:
        forbidden_found = [name for name in feature_names if name in forbidden_names]
        if forbidden_found:
            is_valid = False
            error_msg = f"Forbidden feature names found: {', '.join(forbidden_found)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check for required names
    if required_names:
        missing_required = [name for name in required_names if name not in feature_names]
        if missing_required:
            is_valid = False
            error_msg = f"Missing required feature names: {', '.join(missing_required)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
        missing_names = [name for name in required_names if name not in feature_names]
        if missing_names:
            is_valid = False
            error_msg = f"Required feature names missing: {', '.join(missing_names)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
        missing_names = [name for name in required_names if name not in feature_names]
        if missing_names:
            is_valid = False
            error_msg = f"Required feature names missing: {', '.join(missing_names)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_numeric_array(array: Union[np.ndarray, List],
                         min_length: Optional[int] = None,
                         max_length: Optional[int] = None,
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None,
                         no_nan: bool = True,
                         raise_error: bool = True,
                         logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate a numeric array against a set of requirements.
    
    Parameters
    ----------
    array : Union[np.ndarray, List]
        Array to validate
    min_length : Optional[int], default=None
        Minimum required length of the array
    max_length : Optional[int], default=None
        Maximum allowed length of the array
    min_value : Optional[float], default=None
        Minimum allowed value in the array
    max_value : Optional[float], default=None
        Maximum allowed value in the array
    no_nan : bool, default=True
        Whether to check for NaN values
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Convert to numpy array if needed
    if not isinstance(array, np.ndarray):
        try:
            array = np.array(array)
        except:
            error_messages.append("Input cannot be converted to numpy array")
            if logger:
                logger.error("Input cannot be converted to numpy array")
            if raise_error:
                raise ValueError("Input cannot be converted to numpy array")
            return False, error_messages
    
    # Check if array is empty
    if array.size == 0:
        error_messages.append("Array is empty")
        if logger:
            logger.error("Array is empty")
        if raise_error:
            raise ValueError("Array is empty")
        return False, error_messages
    
    # Check if array is numeric
    if not np.issubdtype(array.dtype, np.number):
        is_valid = False
        error_msg = f"Array is not numeric, dtype: {array.dtype}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check length constraints
    if min_length is not None and len(array) < min_length:
        is_valid = False
        error_msg = f"Array length {len(array)} is less than minimum {min_length}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    if max_length is not None and len(array) > max_length:
        is_valid = False
        error_msg = f"Array length {len(array)} is greater than maximum {max_length}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check value constraints
    if min_value is not None and np.any(array < min_value):
        is_valid = False
        count = np.sum(array < min_value)
        error_msg = f"Array contains {count} values less than minimum {min_value}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    if max_value is not None and np.any(array > max_value):
        is_valid = False
        count = np.sum(array > max_value)
        error_msg = f"Array contains {count} values greater than maximum {max_value}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check for NaN values
    if no_nan and np.any(np.isnan(array)):
        is_valid = False
        count = np.sum(np.isnan(array))
        error_msg = f"Array contains {count} NaN values"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_probability_array(array: Union[np.ndarray, List],
                             binary: bool = False,
                             sum_to_one: bool = False,
                             raise_error: bool = True,
                             logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate that an array contains valid probabilities.
    
    Parameters
    ----------
    array : Union[np.ndarray, List]
        Array to validate
    binary : bool, default=False
        Whether to check that values are 0 or 1 only
    sum_to_one : bool, default=False
        Whether to check that probabilities sum to 1
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # First use the numeric array validator
    valid, errors = validate_numeric_array(
        array, 
        min_value=0.0, 
        max_value=1.0,
        no_nan=True,
        raise_error=False
    )
    
    if not valid:
        is_valid = False
        error_messages.extend(errors)
        
        if logger:
            for error in errors:
                logger.error(error)
    
    # Convert to numpy array if needed
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    # Check if binary (0 or 1 only)
    if binary and not np.all(np.isin(array, [0, 1])):
        is_valid = False
        error_msg = "Array contains values other than 0 and 1"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check if probabilities sum to 1
    if sum_to_one:
        # Handle multi-dimensional arrays
        if array.ndim > 1:
            # Check if each row sums to 1
            row_sums = np.sum(array, axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5):
                is_valid = False
                error_msg = "Not all rows sum to 1.0"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
        else:
            # Single dimension array
            if not np.isclose(np.sum(array), 1.0, rtol=1e-5, atol=1e-5):
                is_valid = False
                error_msg = f"Array sums to {np.sum(array)}, not 1.0"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_model_params(params: Dict[str, Any],
                        allowed_params: Optional[Dict[str, type]] = None,
                        required_params: Optional[List[str]] = None,
                        param_constraints: Optional[Dict[str, Callable[[Any], bool]]] = None,
                        raise_error: bool = True,
                        logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate model parameters against a set of requirements.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of model parameters
    allowed_params : Optional[Dict[str, type]], default=None
        Dictionary mapping parameter names to their expected types
    required_params : Optional[List[str]], default=None
        List of required parameter names
    param_constraints : Optional[Dict[str, Callable[[Any], bool]]], default=None
        Dictionary mapping parameter names to validation functions
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if params is None or empty
    if params is None or not params:
        error_messages.append("Parameters dictionary is empty or None")
        if logger:
            logger.error("Parameters dictionary is empty or None")
        if raise_error:
            raise ValueError("Parameters dictionary is empty or None")
        return False, error_messages
    
    # Check allowed parameters
    if allowed_params:
        for param_name, param_value in params.items():
            if param_name not in allowed_params:
                is_valid = False
                error_msg = f"Unknown parameter: {param_name}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
            else:
                expected_type = allowed_params[param_name]
                if not isinstance(param_value, expected_type):
                    is_valid = False
                    error_msg = f"Parameter '{param_name}' has type {type(param_value)}, expected {expected_type}"
                    error_messages.append(error_msg)
                    if logger:
                        logger.error(error_msg)
    
    # Check required parameters
    if required_params:
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            is_valid = False
            error_msg = f"Missing required parameters: {', '.join(missing_params)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check parameter constraints
    if param_constraints:
        for param_name, constraint_func in param_constraints.items():
            if param_name in params:
                try:
                    if not constraint_func(params[param_name]):
                        is_valid = False
                        error_msg = f"Parameter '{param_name}' with value {params[param_name]} does not satisfy constraints"
                        error_messages.append(error_msg)
                        if logger:
                            logger.error(error_msg)
                except Exception as e:
                    is_valid = False
                    error_msg = f"Error validating parameter '{param_name}': {str(e)}"
                    error_messages.append(error_msg)
                    if logger:
                        logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_prediction_input(X: Union[pd.DataFrame, np.ndarray],
                            expected_shape: Optional[Tuple[Optional[int], int]] = None,
                            expected_columns: Optional[List[str]] = None,
                            raise_error: bool = True,
                            logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate input data for model prediction.
    
    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input data for prediction
    expected_shape : Optional[Tuple[Optional[int], int]], default=None
        Expected shape of input data (n_samples, n_features)
        First dimension can be None to allow any number of samples
    expected_columns : Optional[List[str]], default=None
        Expected column names if X is a DataFrame
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if X is None or empty
    if X is None:
        error_messages.append("Input is None")
        if logger:
            logger.error("Input is None")
        if raise_error:
            raise ValueError("Input is None")
        return False, error_messages
    
    # Handle different input types
    if isinstance(X, pd.DataFrame):
        if X.empty:
            error_messages.append("Input DataFrame is empty")
            if logger:
                logger.error("Input DataFrame is empty")
            if raise_error:
                raise ValueError("Input DataFrame is empty")
            return False, error_messages
        
        # Check columns if specified
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in X.columns]
            if missing_columns:
                is_valid = False
                error_msg = f"Missing required columns: {', '.join(missing_columns)}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
            
            extra_columns = [col for col in X.columns if col not in expected_columns]
            if extra_columns:
                # This is just a warning, not an error
                warning_msg = f"Input contains extra columns: {', '.join(extra_columns)}"
                warnings.warn(warning_msg)
                if logger:
                    logger.warning(warning_msg)
        
        # Check shape if specified
        if expected_shape:
            expected_n_samples, expected_n_features = expected_shape
            actual_n_samples, actual_n_features = X.shape
            
            if expected_n_samples is not None and actual_n_samples != expected_n_samples:
                is_valid = False
                error_msg = f"Input has {actual_n_samples} samples, expected {expected_n_samples}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
            
            if actual_n_features != expected_n_features:
                is_valid = False
                error_msg = f"Input has {actual_n_features} features, expected {expected_n_features}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
    
    elif isinstance(X, np.ndarray):
        if X.size == 0:
            error_messages.append("Input array is empty")
            if logger:
                logger.error("Input array is empty")
            if raise_error:
                raise ValueError("Input array is empty")
            return False, error_messages
        
        # Check shape if specified
        if expected_shape:
            # Handle 1D arrays
            if X.ndim == 1:
                actual_shape = (1, X.shape[0])
            else:
                actual_shape = X.shape
            
            expected_n_samples, expected_n_features = expected_shape
            actual_n_samples, actual_n_features = actual_shape
            
            if expected_n_samples is not None and actual_n_samples != expected_n_samples:
                is_valid = False
                error_msg = f"Input has {actual_n_samples} samples, expected {expected_n_samples}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
            
            if actual_n_features != expected_n_features:
                is_valid = False
                error_msg = f"Input has {actual_n_features} features, expected {expected_n_features}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
    
    else:
        is_valid = False
        error_msg = f"Unsupported input type: {type(X)}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_threshold(threshold: float,
                     min_value: float = 0.0,
                     max_value: float = 1.0,
                     raise_error: bool = True,
                     logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate a probability threshold value.
    
    Parameters
    ----------
    threshold : float
        Threshold value to validate
    min_value : float, default=0.0
        Minimum allowed value
    max_value : float, default=1.0
        Maximum allowed value
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if threshold is a number
    if not isinstance(threshold, (int, float)):
        is_valid = False
        error_msg = f"Threshold must be a number, got {type(threshold)}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    else:
        # Check if threshold is within valid range
        if threshold < min_value or threshold > max_value:
            is_valid = False
            error_msg = f"Threshold {threshold} is outside valid range [{min_value}, {max_value}]"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_clinical_weights(weights: Dict[str, float],
                            required_keys: Optional[List[str]] = None,
                            min_weight: Optional[float] = None,
                            max_weight: Optional[float] = None,
                            raise_error: bool = True,
                            logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate clinical weights dictionary.
    
    Parameters
    ----------
    weights : Dict[str, float]
        Dictionary of clinical weights
    required_keys : Optional[List[str]], default=None
        List of required keys in the weights dictionary
    min_weight : Optional[float], default=None
        Minimum allowed weight value
    max_weight : Optional[float], default=None
        Maximum allowed weight value
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if weights is None or empty
    if weights is None or not weights:
        error_messages.append("Weights dictionary is empty or None")
        if logger:
            logger.error("Weights dictionary is empty or None")
        if raise_error:
            raise ValueError("Weights dictionary is empty or None")
        return False, error_messages
    
    # Check required keys
    if required_keys:
        missing_keys = [key for key in required_keys if key not in weights]
        if missing_keys:
            is_valid = False
            error_msg = f"Missing required weights: {', '.join(missing_keys)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check weight values
    for key, value in weights.items():
        # Check if value is a number
        if not isinstance(value, (int, float)):
            is_valid = False
            error_msg = f"Weight '{key}' has non-numeric value: {value}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
            continue
        
        # Check minimum value
        if min_weight is not None and value < min_weight:
            is_valid = False
            error_msg = f"Weight '{key}' ({value}) is less than minimum {min_weight}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
        
        # Check maximum value
        if max_weight is not None and value > max_weight:
            is_valid = False
            error_msg = f"Weight '{key}' ({value}) is greater than maximum {max_weight}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messagescol]):
                    continue
                
                # Check if types match
                if not isinstance(df[col].dtype, expected_type) and df[col].dtype != expected_type:
                    is_valid = False
                    error_msg = f"Column '{col}' has type {df[col].dtype}, expected {expected_type}"
                    error_messages.append(error_msg)
                    if logger:
                        logger.error(error_msg)
    
    # Check for missing values
    if no_missing:
        for col in no_missing:
            if col in df.columns and df[col].isna().any():
                is_valid = False
                missing_count = df[col].isna().sum()
                error_msg = f"Column '{col}' has {missing_count} missing values"
                error_messages.append(error_msg)
                if logger:
                    logger.error(error_msg)
    
    # Raise error if validation failed and raise_error is True
    if not is_valid and raise_error:
        raise ValueError("\n".join(error_messages))
    
    return is_valid, error_messages


def validate_feature_names(feature_names: List[str],
                         pattern: Optional[str] = None,
                         forbidden_names: Optional[List[str]] = None,
                         required_names: Optional[List[str]] = None,
                         raise_error: bool = True,
                         logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Validate feature names against a set of requirements.
    
    Parameters
    ----------
    feature_names : List[str]
        List of feature names to validate
    pattern : Optional[str], default=None
        Regular expression pattern that feature names should match
    forbidden_names : Optional[List[str]], default=None
        List of names that should not be in feature_names
    required_names : Optional[List[str]], default=None
        List of names that must be in feature_names
    raise_error : bool, default=True
        Whether to raise an error if validation fails
    logger : Optional[logging.Logger], default=None
        Logger to log validation errors
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []
    
    # Check if feature_names is empty
    if not feature_names:
        error_messages.append("Feature names list is empty")
        if logger:
            logger.error("Feature names list is empty")
        if raise_error:
            raise ValueError("Feature names list is empty")
        return False, error_messages
    
    # Check for duplicates
    duplicates = set([name for name in feature_names if feature_names.count(name) > 1])
    if duplicates:
        is_valid = False
        error_msg = f"Duplicate feature names found: {', '.join(duplicates)}"
        error_messages.append(error_msg)
        if logger:
            logger.error(error_msg)
    
    # Check against pattern
    if pattern:
        invalid_names = [name for name in feature_names if not re.match(pattern, name)]
        if invalid_names:
            is_valid = False
            error_msg = f"Features don't match pattern '{pattern}': {', '.join(invalid_names)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check for forbidden names
    if forbidden_names:
        forbidden_found = [name for name in feature_names if name in forbidden_names]
        if forbidden_found:
            is_valid = False
            error_msg = f"Forbidden feature names found: {', '.join(forbidden_found)}"
            error_messages.append(error_msg)
            if logger:
                logger.error(error_msg)
    
    # Check for required names
    if required_names:
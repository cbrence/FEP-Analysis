"""
Configuration settings utility functions for FEP analysis.

This module provides utility functions for accessing specific 
configuration settings for the FEP analysis package.
"""

import os
from typing import Dict, Any, List, Tuple

from .loader import load_config


def get_data_path(data_type: str = "raw") -> str:
    """
    Get the path to data files.
    
    Parameters
    ----------
    data_type : str, default="raw"
        Type of data path to retrieve ("raw" or "processed")
        
    Returns
    -------
    str
        Path to the data directory or file
    """
    config = load_config()
    
    if data_type.lower() == "raw":
        return config["data"]["raw_data_path"]
    elif data_type.lower() == "processed":
        return config["data"]["processed_data_path"]
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'raw' or 'processed'.")


def get_visualization_settings() -> Dict[str, Any]:
    """
    Get visualization settings.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of visualization settings
    """
    config = load_config()
    return config["visualization"]


def get_feature_columns() -> Dict[str, List[str]]:
    """
    Get categorized feature columns.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with categorical_columns, continuous_columns, and target_columns
    """
    config = load_config()
    return {
        "categorical_columns": config["features"]["categorical_columns"],
        "continuous_columns": config["features"]["continuous_columns"],
        "target_columns": config["features"]["target_columns"]
    }


def get_model_parameters(model_name: str) -> Dict[str, Any]:
    """
    Get parameters for a specific model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of model parameters
    """
    config = load_config()
    
    if model_name in config["models"]:
        return config["models"][model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in configuration")


def get_evaluation_metrics() -> List[str]:
    """
    Get list of evaluation metrics.
    
    Returns
    -------
    List[str]
        List of evaluation metric names
    """
    config = load_config()
    return config["evaluation"]["metrics"]


def get_figure_size() -> Tuple[int, int]:
    """
    Get default figure size for visualizations.
    
    Returns
    -------
    Tuple[int, int]
        Figure size as (width, height)
    """
    config = load_config()
    return tuple(config["visualization"]["figure_size"])


def get_default_model() -> str:
    """
    Get the name of the default model.
    
    Returns
    -------
    str
        Name of the default model
    """
    config = load_config()
    return config["models"]["default_model"]


def get_webapp_settings() -> Dict[str, Any]:
    """
    Get web application settings.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of web application settings
    """
    config = load_config()
    return config["webapp"]

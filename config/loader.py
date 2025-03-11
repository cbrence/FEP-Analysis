"""
Configuration loader for FEP analysis.

This module provides functions for loading and managing 
configuration settings for the FEP analysis package.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Parameters
    ----------
    config_path : Optional[str], default=None
        Path to the configuration file. If None, loads the default config.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    # If no path provided, use default
    if config_path is None:
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'default_config.json')
    
    # Determine file type from extension
    file_ext = os.path.splitext(config_path)[1].lower()
    
    # Load based on file type
    try:
        if file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif file_ext in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
        
        return config
    except FileNotFoundError:
        # If file not found, return default configuration
        return get_default_config()
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration settings.
    
    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary
    """
    return {
        "data": {
            "raw_data_path": "./data/raw/fep_dataset.csv",
            "processed_data_path": "./data/processed/",
            "test_size": 0.2,
            "random_state": 42
        },
        "features": {
            "categorical_columns": [
                "Cohort", "Accommodation", "Admitted_Hosp", "Alcohol", "Citizenship",
                "Depression_Severity", "Drugs", "Education", "Ethnicity", "Gender",
                "Household", "M0_Emp", "Parent", "Relationship", "M6_Emp", "M6_Rem",
                "M6_Res", "Y1_Emp", "Y1_Rem", "Y1_Res", "Y1_Rem_6"
            ],
            "continuous_columns": [
                "Age", "Depression_Sev_Scale", "Education_Num", 
                "M0_PANSS_G1", "M0_PANSS_G2", "M0_PANSS_G3", "M0_PANSS_G4", 
                "M0_PANSS_G5", "M0_PANSS_G6", "M0_PANSS_G7", "M0_PANSS_G8", 
                "M0_PANSS_G9", "M0_PANSS_G10", "M0_PANSS_G11", "M0_PANSS_G12", 
                "M0_PANSS_G13", "M0_PANSS_G14", "M0_PANSS_G15", "M0_PANSS_G16",
                "M0_PANSS_N1", "M0_PANSS_N2", "M0_PANSS_N3", "M0_PANSS_N4", 
                "M0_PANSS_N5", "M0_PANSS_N6", "M0_PANSS_N7",
                "M0_PANSS_P1", "M0_PANSS_P2", "M0_PANSS_P3", "M0_PANSS_P4", 
                "M0_PANSS_P5", "M0_PANSS_P6", "M0_PANSS_P7",
                "M6_PANSS_Total_score", "Y1_PANSS_Total_score"
            ],
            "target_columns": ["Y1_Rem", "Y1_Res"]
        },
        "models": {
            "default_model": "ensemble",
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "neural_network": {
                "hidden_layer_sizes": [50, 25],
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "max_iter": 200,
                "random_state": 42
            },
            "ensemble": {
                "models": ["logistic_regression", "gradient_boosting"],
                "weights": [0.4, 0.6]
            }
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "cv_folds": 5,
            "threshold": 0.5
        },
        "visualization": {
            "color_palette": "viridis",
            "figure_size": [10, 6],
            "dpi": 100,
            "save_figures": False,
            "figures_path": "./figures/"
        },
        "webapp": {
            "title": "FEP Outcome Prediction",
            "description": "Predicting remission and response in First Episode Psychosis",
            "theme": "light",
            "show_code": False
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : str
        Path where to save the configuration
    """
    # Determine file type from extension
    file_ext = os.path.splitext(config_path)[1].lower()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save based on file type
    try:
        if file_ext == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        elif file_ext in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {str(e)}")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
        
    Returns
    -------
    Dict[str, Any]
        Model configuration dictionary
    """
    config = load_config()
    if model_name in config["models"]:
        return config["models"][model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in configuration")


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Parameters
    ----------
    updates : Dict[str, Any]
        Dictionary with updated values
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    config = load_config()
    
    # Helper function for recursive dictionary update
    def update_dict(original, updates):
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                update_dict(original[key], value)
            else:
                original[key] = value
    
    update_dict(config, updates)
    return config

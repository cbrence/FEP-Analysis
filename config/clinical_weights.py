"""
Clinical weights configuration for FEP analysis.

This module defines the clinical risk weights and thresholds based on the asymmetric
costs of different types of errors in FEP prediction. Clinicians can customize
these weights based on their expertise and specific clinical priorities.
"""
import os
import json
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default clinical configuration
DEFAULT_CONFIG = {
    # Class definitions and descriptions
    "class_definitions": {
        "0": "No remission at 6 months, No remission at 12 months, Poor treatment adherence (Highest risk)",
        "1": "No remission at 6 months, No remission at 12 months, Moderate treatment adherence (Very high risk)",
        "2": "Remission at 6 months, No remission at 12 months - Early Relapse with significant functional decline (High risk)",
        "3": "No remission at 6 months, Remission at 12 months, Poor treatment adherence (Moderate-high risk)",
        "4": "Remission at 6 months, No remission at 12 months, Maintained social functioning (Moderate risk)",
        "5": "No remission at 6 months, Remission at 12 months, Good treatment adherence (Moderate-low risk)",
        "6": "Remission at 6 months, Remission at 12 months with some residual symptoms (Low risk)",
        "7": "Remission at 6 months, Remission at 12 months, Full symptomatic and functional recovery (Lowest risk)"
    },
    
    # Clinical risk categorization (which classes are high/moderate/low risk)
    "risk_levels": {
        "high_risk": [0, 1, 2],      # Highest and high risk classes
        "moderate_risk": [3, 4, 5],   # Moderate risk classes
        "low_risk": [6, 7]            # Low risk classes
    },
    
    # Class weights for model training (higher = more important)
    "class_weights": {
        "0": 10.0,  # Highest risk - Poor adherence, no remission at either time point
        "1": 8.0,   # Very high risk - Moderate adherence, no remission at either time point
        "2": 7.0,   # High risk - Early relapse with functional decline
        "3": 5.0,   # Moderate-high risk - Poor adherence but late remission
        "4": 4.0,   # Moderate risk - Early remission not sustained but maintained functioning
        "5": 3.0,   # Moderate-low risk - Good adherence with late remission
        "6": 2.0,   # Low risk - Sustained remission but residual symptoms
        "7": 1.0    # Lowest risk - Full recovery
    },
    
    # Prediction thresholds (lower threshold = higher sensitivity)
    "prediction_thresholds": {
        "0": 0.2,  # Very low threshold for highest risk
        "1": 0.25, # Low threshold for very high risk
        "2": 0.3,  # Low threshold for high risk
        "3": 0.35, # Moderately low threshold for moderate-high risk
        "4": 0.4,  # Moderate threshold for moderate risk
        "5": 0.45, # Moderate threshold for moderate-low risk
        "6": 0.5,  # Standard threshold for low risk
        "7": 0.55  # Higher threshold for lowest risk
    },
    
    # Costs for different types of errors (used in evaluation)
    "error_costs": {
        "false_negative": {  # Missing a case that needs intervention
            "high_risk": 10.0,     # Missing high-risk cases (classes 0, 1, 2)
            "moderate_risk": 5.0,  # Missing moderate-risk cases (classes 3, 4, 5)
            "low_risk": 1.0        # Missing low-risk cases (classes 6, 7)
        },
        "false_positive": {  # Unnecessary intervention
            "high_risk": 1.0,      # Acceptable cost for high-risk
            "moderate_risk": 2.0,  # Moderate cost for moderate-risk
            "low_risk": 3.0        # Higher cost for low-risk (avoid overtreatment)
        }
    },
    
    # Risk stratification thresholds for the dashboard
    "stratification_thresholds": {
        "high_risk": 0.3,     # Threshold for high-risk category
        "moderate_risk": 0.2  # Threshold for moderate-risk category
    },
    
    # Clinical recommendations based on risk level
    "clinical_recommendations": {
        "high_risk": [
            "Consider more frequent monitoring (weekly)",
            "Implement strategies to improve medication adherence",
            "Intensive psychosocial interventions",
            "Family/caregiver education and involvement",
            "Regular monitoring for early warning signs of relapse",
            "Consider case management or assertive community treatment"
        ],
        "moderate_risk": [
            "Increase monitoring frequency (bi-weekly)",
            "Address specific modifiable risk factors",
            "Targeted psychosocial support",
            "Education about early warning signs",
            "Regular medication review",
            "Employment/educational support as appropriate"
        ],
        "low_risk": [
            "Maintain standard monitoring schedule",
            "Continue current management plan",
            "Routine monitoring for changes in status",
            "Focus on functional recovery and social integration",
            "Gradual transition to less intensive services"
        ]
    },
    
    # Time weights for early detection
    "time_discount_factor": 0.9  # Weight factor for time-weighted errors
}

# Current active configuration (starts with default)
_ACTIVE_CONFIG = DEFAULT_CONFIG.copy()

# File paths for configuration
DEFAULT_CONFIG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "clinical_configs"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "default_config.yaml"

def ensure_config_dir():
    """Ensure the configuration directory exists."""
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
def save_default_config():
    """Save the default configuration to file."""
    ensure_config_dir()
    with open(DEFAULT_CONFIG_PATH, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
    logger.info(f"Default config saved to {DEFAULT_CONFIG_PATH}")

def load_config(config_path=None):
    """
    Load a configuration from file.
    
    Parameters:
    -----------
    config_path : str or Path, default=None
        Path to configuration file. If None, uses default config.
        
    Returns:
    --------
    dict : Loaded configuration
    """
    global _ACTIVE_CONFIG
    
    if config_path is None:
        # Use default if exists, otherwise save it first
        if not DEFAULT_CONFIG_PATH.exists():
            save_default_config()
        config_path = DEFAULT_CONFIG_PATH
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            if str(config_path).endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        
        # Update active configuration
        _ACTIVE_CONFIG = config
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration instead")
        return DEFAULT_CONFIG.copy()

def save_config(config, config_path):
    """
    Save a configuration to file.
    
    Parameters:
    -----------
    config : dict
        Configuration to save
    config_path : str or Path
        Path to save configuration to
    """
    try:
        with open(config_path, 'w') as f:
            if str(config_path).endswith('.json'):
                json.dump(config, f, indent=2)
            else:
                yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def get_config():
    """Get the currently active configuration."""
    return _ACTIVE_CONFIG

def update_config(new_config):
    """
    Update the active configuration.
    
    Parameters:
    -----------
    new_config : dict
        New configuration values
        
    Returns:
    --------
    dict : Updated configuration
    """
    global _ACTIVE_CONFIG
    
    # Deep update to handle nested dictionaries
    def deep_update(source, updates):
        for key, value in updates.items():
            if key in source and isinstance(source[key], dict) and isinstance(value, dict):
                deep_update(source[key], value)
            else:
                source[key] = value
    
    # Update configuration
    deep_update(_ACTIVE_CONFIG, new_config)
    logger.info("Configuration updated")
    return _ACTIVE_CONFIG

def reset_to_default():
    """Reset to default configuration."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = DEFAULT_CONFIG.copy()
    logger.info("Configuration reset to default")
    return _ACTIVE_CONFIG

def validate_config(config):
    """
    Validate that a configuration has all required fields and valid values.
    
    Parameters:
    -----------
    config : dict
        Configuration to validate
        
    Returns:
    --------
    tuple : (is_valid, error_message)
    """
    required_sections = [
        "class_definitions", "risk_levels", "class_weights", 
        "prediction_thresholds", "error_costs", "stratification_thresholds"
    ]
    
    # Check that all required sections exist
    for section in required_sections:
        if section not in config:
            return False, f"Missing required section: {section}"
    
    # Check that risk levels are valid
    all_classes = set(map(int, config["class_definitions"].keys()))
    defined_risk_classes = set()
    for risk_level, classes in config["risk_levels"].items():
        defined_risk_classes.update(classes)
    
    # Check that all classes are assigned a risk level
    if all_classes != defined_risk_classes:
        missing = all_classes - defined_risk_classes
        extra = defined_risk_classes - all_classes
        error_msg = []
        if missing:
            error_msg.append(f"Classes without risk level: {missing}")
        if extra:
            error_msg.append(f"Risk levels with undefined classes: {extra}")
        return False, "; ".join(error_msg)
    
    # Validate that all classes have weights
    weighted_classes = set(map(int, config["class_weights"].keys()))
    if all_classes != weighted_classes:
        return False, "Not all classes have weights assigned"
    
    # Validate threshold values
    for cls, threshold in config["prediction_thresholds"].items():
        if not 0 < threshold < 1:
            return False, f"Invalid threshold ({threshold}) for class {cls}, must be between 0 and 1"
    
    return True, "Configuration is valid"

# On module import, try to load the default configuration
try:
    load_config()
except Exception as e:
    logger.warning(f"Could not load configuration: {e}")
    logger.info("Using default configuration values")

# Make variables from config available at module level
# This allows imports like: from config.clinical_weights import CLASS_WEIGHTS

# Class definitions and descriptions
CLASS_DEFINITIONS = {int(k): v for k, v in _ACTIVE_CONFIG["class_definitions"].items()}

# Clinical risk categorization
HIGH_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["high_risk"]
MODERATE_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["moderate_risk"]
LOW_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["low_risk"]

# Class weights for model training
CLASS_WEIGHTS = {int(k): v for k, v in _ACTIVE_CONFIG["class_weights"].items()}

# Prediction thresholds
PREDICTION_THRESHOLDS = {int(k): v for k, v in _ACTIVE_CONFIG["prediction_thresholds"].items()}

# Costs for different types of errors
ERROR_COSTS = _ACTIVE_CONFIG["error_costs"]

# Risk stratification thresholds
RISK_STRATIFICATION = _ACTIVE_CONFIG["stratification_thresholds"]

# Clinical recommendations
CLINICAL_RECOMMENDATIONS = _ACTIVE_CONFIG["clinical_recommendations"]

# Time discount factor
TIME_DISCOUNT_FACTOR = _ACTIVE_CONFIG["time_discount_factor"]

# Utility function to get risk level for a class
def get_risk_level(cls):
    """Get risk level for a class."""
    if cls in HIGH_RISK_CLASSES:
        return "high_risk"
    elif cls in MODERATE_RISK_CLASSES:
        return "moderate_risk"
    else:
        return "low_risk"

# Create a dictionary mapping classes to risk levels
CLASS_TO_RISK_LEVEL = {cls: get_risk_level(cls) for cls in CLASS_DEFINITIONS.keys()}

# For backward compatibility
def refresh_from_config():
    """Refresh module-level variables from the active configuration."""
    global CLASS_DEFINITIONS, HIGH_RISK_CLASSES, MODERATE_RISK_CLASSES, LOW_RISK_CLASSES
    global CLASS_WEIGHTS, PREDICTION_THRESHOLDS, ERROR_COSTS, RISK_STRATIFICATION
    global CLINICAL_RECOMMENDATIONS, TIME_DISCOUNT_FACTOR, CLASS_TO_RISK_LEVEL
    
    # Ensure class definitions are integers
    CLASS_DEFINITIONS = {int(k): v for k, v in _ACTIVE_CONFIG["class_definitions"].items()}

    # Ensure risk level classes are integers
    HIGH_RISK_CLASSES = [int(cls) if isinstance(cls, str) else cls 
                         for cls in _ACTIVE_CONFIG["risk_levels"]["high_risk"]]
    MODERATE_RISK_CLASSES = [int(cls) if isinstance(cls, str) else cls 
                             for cls in _ACTIVE_CONFIG["risk_levels"]["moderate_risk"]]
    LOW_RISK_CLASSES = [int(cls) if isinstance(cls, str) else cls 
                         for cls in _ACTIVE_CONFIG["risk_levels"]["low_risk"]]

    # Update all module-level variables
    CLASS_DEFINITIONS = {int(k): v for k, v in _ACTIVE_CONFIG["class_definitions"].items()}
    HIGH_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["high_risk"]
    MODERATE_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["moderate_risk"]
    LOW_RISK_CLASSES = _ACTIVE_CONFIG["risk_levels"]["low_risk"]
    CLASS_WEIGHTS = {int(k): v for k, v in _ACTIVE_CONFIG["class_weights"].items()}
    PREDICTION_THRESHOLDS = {int(k): v for k, v in _ACTIVE_CONFIG["prediction_thresholds"].items()}
    ERROR_COSTS = _ACTIVE_CONFIG["error_costs"]
    RISK_STRATIFICATION = _ACTIVE_CONFIG["stratification_thresholds"]
    CLINICAL_RECOMMENDATIONS = _ACTIVE_CONFIG["clinical_recommendations"]
    TIME_DISCOUNT_FACTOR = _ACTIVE_CONFIG["time_discount_factor"]
    CLASS_TO_RISK_LEVEL = {cls: get_risk_level(cls) for cls in CLASS_DEFINITIONS.keys()}

    # Rebuild CLASS_TO_RISK_LEVEL with consistent integer keys
    CLASS_TO_RISK_LEVEL = {int(cls): get_risk_level(int(cls)) for cls in CLASS_DEFINITIONS.keys()}
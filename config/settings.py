"""
General configuration settings for the FEP analysis application.

This module defines application-wide settings that are not directly related
to clinical weights but control various aspects of the system's behavior.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
DEFAULT_DATASET = "fep_dataset.csv"
DATA_ENCODING = "utf-8"

# Model training settings
RANDOM_SEED = 42
TEST_SIZE = 0.3
CROSS_VALIDATION_FOLDS = 5
USE_STRATIFIED_SAMPLING = True

# Feature engineering settings
APPLY_FEATURE_SCALING = True
FEATURE_IMPORTANCE_THRESHOLD = 0.015  # Minimum importance threshold for feature selection
USE_PANSS_CLINICAL_NAMES = True  # Whether to rename PANSS columns to clinical names

# Model parameters
MODEL_PARAMS = {
    "logistic_regression": {
        "cv": 10,
        "solver": "saga",
        "penalty": "l1",
        "multi_class": "ovr"
    },
    "decision_tree": {
        "param_grid": {
            "max_depth": range(2, 8),
            "min_samples_leaf": range(5, 55, 5),
            "min_samples_split": range(5, 110, 5)
        }
    },
    "gradient_boosting": {
        "param_dist": {
            "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "max_depth": range(3, 8),
            "n_estimators": range(100, 150, 10),
            "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "min_samples_leaf": range(1, 10)
        },
        "n_iter": 20
    },
    "neural_network": {
        "hidden_layer_sizes": (6, 6),
        "activation": "relu",
        "learning_rate": 0.05,
        "max_iter": 250
    },
    "high_risk_ensemble": {
        "high_risk_threshold": 0.7
    }
}

# Evaluation settings
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "clinical_utility"
]

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-whitegrid"
DEFAULT_FIGURE_SIZE = (10, 6)
COLOR_PALETTE = {
    "high_risk": "#ff6b6b",
    "moderate_risk": "#feca57",
    "low_risk": "#1dd1a1",
    "logistic_regression": "#ff9999",
    "decision_tree": "#66b3ff",
    "gradient_boosting": "#99ff99",
    "high_risk_ensemble": "#ffcc99"
}

# Dashboard settings
DASHBOARD_TITLE = "FEP Outcome Prediction"
DASHBOARD_ICON = "🧠"
DASHBOARD_LAYOUT = "wide"
CACHE_EXPIRY_HOURS = 24

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# Feature groupings for visualization and analysis
FEATURE_GROUPS = {
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
        "P1: Delusions", "P2: Conceptual disorganization", "P3: Hallucinatory behaviour",
        "P4: Excitement", "P5: Grandiosity", "P6: Suspiciousness", "P7: Hostility"
    ],
    "panss_negative": [
        "N1: Blunted affect", "N2: Emotional withdrawal", "N3: Poor rapport",
        "N4: Passive/apathetic social withdrawal", "N5: Difficulty in abstract thinking",
        "N6: Lack of spontaneity and flow of conversation", "N7: Stereotyped thinking"
    ],
    "panss_general": [
        "G1: Somatic concern", "G2: Anxiety", "G3: Guilt feelings", "G4: Tension",
        "G5: Mannerisms & posturing", "G6: Depression", "G7: Motor retardation",
        "G8: Uncooperativeness", "G9: Unusual thought content", "G10: Disorientation",
        "G11: Poor attention", "G12: Lack of judgment and insight", "G13: Disturbance of volition",
        "G14: Poor impulse control", "G15: Preoccupation", "G16: Active social avoidance"
    ]
}

# Early warning signs configuration
EARLY_WARNING_SIGNS = {
    "sleep_disturbance": ["G3: Guilt feelings", "G2: Anxiety"],
    "social_withdrawal": ["N4: Passive/apathetic social withdrawal", "G16: Active social avoidance"],
    "suspiciousness_increase": ["P6: Suspiciousness", "G14: Poor impulse control"],
    "thought_disorder": ["P2: Conceptual disorganization", "P5: Grandiosity"],
    "medication_adherence": ["G12: Lack of judgment and insight"]
}

# Environment-specific settings (development, testing, production)
ENVIRONMENT = os.environ.get("FEP_ENVIRONMENT", "development")

# Development settings
if ENVIRONMENT == "development":
    LOG_LEVEL = "DEBUG"
    USE_SAMPLE_DATA = True
    ENABLE_PROFILING = True

# Testing settings
elif ENVIRONMENT == "testing":
    LOG_LEVEL = "DEBUG"
    RANDOM_SEED = 42
    USE_DUMMY_MODELS = True

# Production settings
elif ENVIRONMENT == "production":
    LOG_LEVEL = "WARNING"
    LOG_TO_FILE = True
    DASHBOARD_DEBUG_MODE = False
    ENABLE_TELEMETRY = True

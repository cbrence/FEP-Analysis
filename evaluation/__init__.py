"""
Evaluation tools for FEP prediction models.

This package provides tools for evaluating FEP prediction models with a focus
on clinical utility and prioritizing the correct identification of high-risk cases.
"""

# Import from metrics.py
from .metrics import (
    clinical_utility_score,
    weighted_f_score,
    time_weighted_error,
    risk_adjusted_auc,
    precision_at_risk_level,
    high_risk_recall_score,
    clinical_classification_report,
    print_clinical_classification_report,
    calibration_curve_data
)

# Import from cross_validation.py
from .cross_validation import (
    stratified_risk_split,
    bootstrap_evaluation,
    HighRiskStratifiedKFold,
    BalancedGroupKFold,
    cross_validate_with_metrics,
    temporal_cross_validation
)

__all__ = [
    # From metrics.py
    'clinical_utility_score',
    'weighted_f_score',
    'time_weighted_error',
    'risk_adjusted_auc',
    'precision_at_risk_level',
    'high_risk_recall_score',
    'clinical_classification_report',
    'print_clinical_classification_report',
    'calibration_curve_data',
    
    # From cross_validation.py
    'stratified_risk_split',
    'bootstrap_evaluation',
    'HighRiskStratifiedKFold',
    'BalancedGroupKFold',
    'cross_validate_with_metrics',
    'temporal_cross_validation'
]

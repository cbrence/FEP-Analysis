# fep_analysis/__init__.py
"""
FEP Outcome Prediction with Cost-Sensitive Metrics.

This package provides machine learning models and tools for predicting 
First Episode Psychosis (FEP) outcomes with a focus on preventing relapse
through cost-sensitive approaches.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'


# fep_analysis/config/__init__.py
"""
Configuration settings for the FEP analysis package.
"""


# fep_analysis/data/__init__.py
"""
Data loading and preprocessing tools for FEP analysis.
"""

from .loader import load_data
from .preprocessor import clean_data, impute_missing_values, encode_target_variables

__all__ = ['load_data', 'clean_data', 'impute_missing_values', 'encode_target_variables']


# fep_analysis/features/__init__.py
"""
Feature engineering tools for FEP analysis.
"""

from .engineer import transform_skewed_features, one_hot_encode_features
from .selector import select_features_by_importance
from .temporal import create_trend_features, SymptomTrendExtractor, EarlyWarningSigns

__all__ = [
    'transform_skewed_features',
    'one_hot_encode_features',
    'select_features_by_importance',
    'create_trend_features',
    'SymptomTrendExtractor',
    'EarlyWarningSigns'
]


# fep_analysis/models/__init__.py
"""
Machine learning models for FEP outcome prediction.
"""

from .base import BaseFEPModel
from .logistic import LogisticRegressionFEP
from .decision_tree import DecisionTreeFEP
from .gradient_boosting import GradientBoostingFEP
from .ensemble import HighRiskFocusedEnsemble, TimeDecayEnsemble, stacked_prediction

__all__ = [
    'BaseFEPModel',
    'LogisticRegressionFEP',
    'DecisionTreeFEP',
    'GradientBoostingFEP',
    'HighRiskFocusedEnsemble',
    'TimeDecayEnsemble',
    'stacked_prediction'
]


# fep_analysis/evaluation/__init__.py
"""
Evaluation tools for FEP prediction models.
"""

from .metrics import (
    clinical_utility_score,
    weighted_f_score,
    time_weighted_error,
    risk_adjusted_auc,
    precision_at_risk_level
)
from .threshold_optimization import (
    find_optimal_threshold,
    find_optimal_thresholds_multiclass,
    apply_thresholds,
    plot_threshold_analysis
)
from .clinical_utility import (
    calculate_error_costs,
    compare_model_utilities,
    plot_utility_comparison,
    plot_cost_breakdown,
    compare_thresholds
)

__all__ = [
    'clinical_utility_score',
    'weighted_f_score',
    'time_weighted_error',
    'risk_adjusted_auc',
    'precision_at_risk_level',
    'find_optimal_threshold',
    'find_optimal_thresholds_multiclass',
    'apply_thresholds',
    'plot_threshold_analysis',
    'calculate_error_costs',
    'compare_model_utilities',
    'plot_utility_comparison',
    'plot_cost_breakdown',
    'compare_thresholds'
]


# fep_analysis/visualization/__init__.py
"""
Visualization tools for FEP analysis results.
"""

from .plots import plot_feature_importance, plot_confusion_matrix, plot_roc_curves
from .model_comparison import plot_model_comparison, plot_threshold_effects
from .feature_importance import plot_top_features, plot_feature_groups
from .risk_stratification import plot_risk_distribution, plot_risk_timeline

__all__ = [
    'plot_feature_importance',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_model_comparison',
    'plot_threshold_effects',
    'plot_top_features',
    'plot_feature_groups',
    'plot_risk_distribution',
    'plot_risk_timeline'
]


# fep_analysis/webapp/__init__.py
"""
Interactive dashboard for FEP prediction.
"""

from .app import main

__all__ = ['main']

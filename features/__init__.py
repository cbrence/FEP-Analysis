"""
Feature engineering tools for FEP analysis.

This package provides functions and classes for creating, selecting, and
transforming features for FEP outcome prediction, with a special focus
on early warning signs detection.
"""

# Import from engineer.py
from .engineer import (
    transform_skewed_features,
    one_hot_encode_features,
    scale_features,
    create_interaction_features,
    engineer_features,
    PANSSFeatureTransformer
)

# Import from selector.py
from .selector import (
    select_features_by_importance,
    select_features_by_correlation,
    select_features_for_multicollinearity,
    select_best_features,
    FeatureSelector
)

# Import from temporal.py
from .temporal import (
    create_trend_features,
    create_early_warning_features,
    filter_longitudinal_data,
    SymptomTrendExtractor,
    EarlyWarningSigns,
    SymptomStabilityExtractor,
    TemporalPatternDetector
)

__all__ = [
    # From engineer.py
    'transform_skewed_features',
    'one_hot_encode_features',
    'scale_features',
    'create_interaction_features',
    'engineer_features',
    'PANSSFeatureTransformer',
    
    # From selector.py
    'select_features_by_importance',
    'select_features_by_correlation',
    'select_features_for_multicollinearity',
    'select_best_features',
    'FeatureSelector',
    
    # From temporal.py
    'create_trend_features',
    'create_early_warning_features',
    'filter_longitudinal_data',
    'SymptomTrendExtractor',
    'EarlyWarningSigns',
    'SymptomStabilityExtractor',
    'TemporalPatternDetector'
]

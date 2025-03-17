# fep_analysis/webapp/pages/__init__.py
"""
Page modules for the FEP prediction dashboard.
"""

from .components.home import render_home_page
from .components.prediction import show_prediction_tool
from .components.model_comparison import show_model_comparison
from .components.feature_importance import display_feature_importance

__all__ = [
    'show_home',
    'show_prediction_tool',
    'show_model_comparison',
    'display_feature_importance'
]


# fep_analysis/webapp/components/__init__.py
"""
UI components for the FEP prediction dashboard.
"""

from .components.risk_display import (
    display_risk_stratification,
    display_probability_bars,
    display_risk_factors,
   # display_risk_meter
)

from .components.inputs import (
    demographic_inputs,
    clinical_inputs,
    panss_positive_inputs,
    panss_negative_inputs,
    panss_general_inputs
)

from .components.results import (
    display_prediction_results,
    display_confusion_matrix,
    display_feature_importance
)

__all__ = [
    'display_risk_stratification',
    'display_probability_bars',
    'display_risk_factors',
    #'display_risk_meter',
    'demographic_inputs',
    'clinical_inputs',
    'panss_positive_inputs',
    'panss_negative_inputs',
    'panss_general_inputs',
    'display_prediction_results',
    'display_confusion_matrix',
    'display_feature_importance'
]

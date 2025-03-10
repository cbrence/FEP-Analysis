"""
Web application pages for FEP analysis.

This module provides the pages for the FEP analysis web application.
"""

from .home import render_home_page
from .feature_importance import render_feature_importance_page

__all__ = [
    'render_home_page',
    'render_feature_importance_page'
]

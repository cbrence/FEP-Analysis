"""
Main Streamlit application for FEP prediction.

This module defines the main dashboard structure and navigation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights
from webapp.pages import home, prediction, model_comparison, feature_importance, config_interface

def load_models():
    """
    Load pretrained models from disk.
    
    Returns:
    --------
    dict : Dictionary of models
    """
    models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
    
    # Check if models directory exists
    if not os.path.exists(models_path):
        # For demonstration purposes, return dummy models
        return {
            "Logistic Regression": None,
            "Decision Tree": None,
            "Gradient Boosting": None,
            "High-Risk Ensemble": None
        }
    
    models = {}
    
    # Load each model if it exists
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "High-Risk Ensemble": "high_risk_ensemble.joblib"
    }
    
    for name, filename in model_files.items():
        file_path = os.path.join(models_path, filename)
        if os.path.exists(file_path):
            models[name] = joblib.load(file_path)
        else:
            models[name] = None
    
    return models

def main():
    """
    Main function to run the Streamlit app.
    """
    state = {
        'patients_df': None,  # Will hold patient data once loaded
        'models': {},         # Will hold loaded models
        'recent_predictions': []  # History of recent predictions
    }

    # Set page config
    st.set_page_config(
        page_title="FEP Prediction Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state if needed
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    # Sidebar for navigation
    st.sidebar.title("FEP Prediction Dashboard")
    
    # Navigation options
    nav_options = [
        "Home",
        "Prediction Tool",
        "Model Comparison",
        "Feature Importance",
        "Clinical Settings"  # New option for configuration
    ]
    
    page = st.sidebar.radio("Navigate", nav_options)
    
    # Display selected page
    if page == "Home":
        home.render_home_page(state)
    elif page == "Prediction Tool":
        prediction.show_prediction_tool(st.session_state.models)
    elif page == "Model Comparison":
        model_comparison.show_model_comparison()
    elif page == "Feature Importance":
        feature_importance.render_feature_importance_page(state)
    elif page == "Clinical Settings":
        config_interface.show_config_editor()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This dashboard implements cost-sensitive outcome metrics 
    to prioritize preventing FEP relapses.
    
    ### Risk Levels
    - 🔴 High Risk: Classes {0}
    - 🟠 Moderate Risk: Classes {1}
    - 🟢 Low Risk: Classes {2}
    """.format(
        clinical_weights.HIGH_RISK_CLASSES,
        clinical_weights.MODERATE_RISK_CLASSES,
        clinical_weights.LOW_RISK_CLASSES
    ))
    
    # Additional information about clinical settings
    if page == "Clinical Settings":
        st.sidebar.info("""
        ### Clinical Configuration
        
        The settings on this page allow clinicians to customize:
        
        - Which outcome patterns are considered high risk
        - How much emphasis to place on each class
        - Thresholds for making predictions
        - Relative costs of different error types
        
        These settings directly affect how the models make predictions
        and how risks are communicated.
        """)

if __name__ == "__main__":
    main()

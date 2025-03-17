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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights
from webapp.components import home, prediction, model_comparison, feature_importance, config_interface
from webapp.components.model_comparison import load_fep_dataset

def load_models():
    """
    Load pretrained models from disk.
    
    Returns:
    --------
    dict : Dictionary of models
    """
    # First try the dashboard-specific trained models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'dashboard_trained')
    
    # If that doesn't exist, try other directories
    if not os.path.exists(models_dir):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'trained')
        
    if not os.path.exists(models_dir):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        st.warning(f"Models directory not found at {models_dir}")
        return {}
    
    models = {}
    
    # Load each model if it exists
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    for model_file in model_files:
        try:
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            
            # Format the model name for display
            model_name = model_file.replace('.joblib', '').replace('_', ' ').title()
            # Handle special case for the ensemble model
            if 'high_risk_ensemble' in model_file.lower():
                model_name = "High-Risk Ensemble"
                
            models[model_name] = model
            st.success(f"Loaded model: {model_name}")
        except Exception as e:
            st.error(f"Error loading model {model_file}: {str(e)}")
    
    return models

   # for name, filename in model_files.items():
   #     file_path = os.path.join(models_path, filename)
   #     if os.path.exists(file_path):
   #         models[name] = joblib.load(file_path)
   #     else:
   #         models[name] = None
    
   # return models

def ensure_dataset_available():
    """
    Make sure the dataset can be loaded from the expected location.
    If not, try to copy it from the raw subdirectory.
    """
    import os
    import shutil
    
    # Expected locations
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    expected_path = os.path.join(data_dir, 'fep_dataset.csv')
    raw_path = os.path.join(data_dir, 'raw', 'fep_dataset.csv')
    
    # Check if the expected path exists
    if os.path.exists(expected_path):
        return True
    
    # If not, check if the raw path exists
    if os.path.exists(raw_path):
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            # Copy the file
            shutil.copy2(raw_path, expected_path)
            return True
        except Exception as e:
            st.error(f"Error copying dataset: {str(e)}")
            return False
    
    return False


def update_sidebar_risk_levels():
    """
    Update the sidebar to display correct risk level assignments.
    This function should be called at the beginning of your app's main function.
    """
    # Ensure correct risk levels are displayed in the sidebar
    import streamlit as st
    
    # Add a section to the sidebar showing risk level assignments
    st.sidebar.subheader("Risk Level Classification")
    
    # High Risk Classes - correct assignment [0, 1, 2]
    st.sidebar.markdown("**🔴 High Risk: Classes [0, 1, 2]**")
    
    # Moderate Risk Classes - correct assignment [3, 4, 5]  
    st.sidebar.markdown("**🟠 Moderate Risk: Classes [3, 4, 5]**")
    
    # Low Risk Classes - correct assignment [6, 7]
    st.sidebar.markdown("**🟢 Low Risk: Classes [6 7]**")
    
    # Add a divider for visual separation
    st.sidebar.markdown("---")


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
    - 🔴 High Risk: Classes [0, 1, 2]
    - 🟠 Moderate Risk: Classes [3, 4, 5]
    - 🟢 Low Risk: Classes [6, 7]
                        
     ### Class Descriptions
    - Class 0: No remission at 6 months, No remission at 12 months, Poor treatment adherence (Highest risk)
    - Class 1: No remission at 6 months, No remission at 12 months, Moderate treatment adherence (Very high risk) 
    - Class 2: Remission at 6 months, No remission at 12 months - Early Relapse with significant functional decline (High risk)
    - Class 3: No remission at 6 months, Remission at 12 months, Poor treatment adherence (Moderate-high risk)
    - Class 4: Remission at 6 months, No remission at 12 months, Maintained social functioning (Moderate risk)
    - Class 5: No remission at 6 months, Remission at 12 months, Good treatment adherence (Moderate-low risk)
    - Class 6: Remission at 6 months, Remission at 12 months with some residual symptoms (Low risk)
    - Class 7: Remission at 6 months, Remission at 12 months, Full symptomatic and functional recovery (Lowest risk)
    """)
    
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

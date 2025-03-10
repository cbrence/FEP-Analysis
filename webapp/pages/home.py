"""
Home page for the FEP analysis web application.

This module provides the main landing page for the FEP analysis web application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..components.inputs import patient_selector, model_selector
from ..components.results import display_prediction_results, display_metrics_table
from ..components.risk_display import display_risk_timeline


def render_home_page(state: Dict[str, Any]) -> None:
    """
    Render the home page of the FEP analysis web application.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.title("First Episode Psychosis (FEP) Outcomes Prediction")
    st.markdown("""
    Welcome to the FEP Analysis Dashboard. This application helps predict and analyze 
    outcomes for patients with First Episode Psychosis using advanced machine learning models.
    """)
    
    # Check if data is loaded
    if 'patients_df' not in state or state['patients_df'] is None:
        st.warning("No data loaded. Please load patient data to continue.")
        
        # Add sample data option for demonstration
        if st.button("Load Sample Data"):
            load_sample_data(state)
        return
    
    # Create dashboard sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Quick Prediction")
        render_quick_prediction_section(state)
    
    with col2:
        st.subheader("Overall Statistics")
        render_statistics_section(state)
    
    # Recent predictions section
    st.subheader("Recent Predictions")
    render_recent_predictions_section(state)
    
    # Risk timeline section
    st.subheader("Patient Risk Timeline")
    render_risk_timeline_section(state)


def render_quick_prediction_section(state: Dict[str, Any]) -> None:
    """
    Render the quick prediction section of the home page.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Patient selector
    selected_patient = patient_selector(
        state['patients_df'],
        key="home_patient_selector",
        label="Select Patient for Prediction",
        patient_info_cols=state.get('patient_info_cols', ['age', 'gender', 'admission_date'])
    )
    
    # Model selector
    selected_model = model_selector(
        state.get('available_models', ["Default Model"]),
        default_model=state.get('default_model'),
        key="home_model_selector",
        label="Select Model",
        model_descriptions=state.get('model_descriptions')
    )
    
    # Threshold slider
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=state.get('default_threshold', 0.5),
        step=0.01,
        key="home_threshold_slider"
    )
    
    # Prediction button
    if st.button("Run Prediction", key="home_predict_button"):
        if selected_patient:
            # Get patient data
            patient_data = state['patients_df'][state['patients_df']['patient_id'] == selected_patient]
            
            # Run prediction
            prediction = run_prediction(patient_data, selected_model, threshold, state)
            
            # Store prediction in state
            if 'recent_predictions' not in state:
                state['recent_predictions'] = []
            
            # Add to recent predictions
            state['recent_predictions'].insert(0, {
                'patient_id': selected_patient,
                'model': selected_model,
                'threshold': threshold,
                'prediction': prediction,
                'timestamp': pd.Timestamp.now()
            })
            
            # Limit to 10 recent predictions
            state['recent_predictions'] = state['recent_predictions'][:10]
            
            # Display prediction
            display_prediction_results(
                prediction,
                threshold=threshold,
                patient_id=selected_patient,
                show_gauge=True,
                show_features=True
            )
        else:
            st.warning("Please select a patient for prediction.")


def render_statistics_section(state: Dict[str, Any]) -> None:
    """
    Render the statistics section of the home page.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Get statistics from state or generate if not present
    if 'model_metrics' not in state:
        # Generate sample metrics if not available
        generate_sample_metrics(state)
    
    # Display metrics
    display_metrics_table(
        state['model_metrics'],
        title="Current Model Performance",
        comparison_metrics=state.get('baseline_metrics')
    )
    
    # Display additional statistics
    if 'patients_df' in state:
        total_patients = len(state['patients_df']['patient_id'].unique())
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", total_patients)
        
        if 'outcomes_df' in state:
            with col2:
                positive_outcomes = state['outcomes_df']['outcome'].sum()
                st.metric("Positive Outcomes", positive_outcomes)
            
            with col3:
                negative_outcomes = len(state['outcomes_df']) - positive_outcomes
                st.metric("Negative Outcomes", negative_outcomes)


def render_recent_predictions_section(state: Dict[str, Any]) -> None:
    """
    Render the recent predictions section of the home page.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    if 'recent_predictions' not in state or not state['recent_predictions']:
        st.info("No recent predictions available.")
        return
    
    # Create tabs for recent predictions
    recent_preds = state['recent_predictions']
    tab_labels = [f"Patient {pred['patient_id']} ({pred['timestamp'].strftime('%H:%M:%S')})" 
                for pred in recent_preds[:5]]
    
    # Only show up to 5 recent predictions as tabs
    tabs = st.tabs(tab_labels)
    
    for i, tab in enumerate(tabs):
        if i < len(recent_preds):
            pred = recent_preds[i]
            with tab:
                display_prediction_results(
                    pred['prediction'],
                    threshold=pred['threshold'],
                    patient_id=pred['patient_id'],
                    show_gauge=True,
                    show_features=True
                )


def render_risk_timeline_section(state: Dict[str, Any]) -> None:
    """
    Render the risk timeline section of the home page.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Check if timeline data is available
    if 'risk_timeline_df' not in state:
        # Generate sample timeline data if not available
        generate_sample_risk_timeline(state)
    
    # Get data from state
    timeline_df = state['risk_timeline_df']
    
    # Display risk timeline
    display_risk_timeline(
        timestamps=timeline_df['timestamp'].values,
        risk_scores=timeline_df['risk_score'].values,
        patient_ids=timeline_df['patient_id'].values if 'patient_id' in timeline_df.columns else None,
        threshold=state.get('default_threshold', 0.5),
        title="Patient Risk Timeline"
    )


def run_prediction(patient_data: pd.DataFrame, 
                  model_name: str, 
                  threshold: float, 
                  state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a prediction for a patient using the selected model.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        DataFrame containing patient data
    model_name : str
        Name of the model to use
    threshold : float
        Prediction threshold
    state : Dict[str, Any]
        Application state dictionary
        
    Returns
    -------
    Dict[str, Any]
        Prediction results
    """
    # Check if we have a prediction function
    if 'prediction_function' in state and callable(state['prediction_function']):
        # Use the provided prediction function
        return state['prediction_function'](patient_data, model_name, threshold)
    
    # Otherwise, generate a sample prediction
    return generate_sample_prediction(patient_data, model_name, threshold)


def generate_sample_prediction(patient_data: pd.DataFrame, 
                              model_name: str, 
                              threshold: float) -> Dict[str, Any]:
    """
    Generate a sample prediction for demonstration purposes.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        DataFrame containing patient data
    model_name : str
        Name of the model to use
    threshold : float
        Prediction threshold
        
    Returns
    -------
    Dict[str, Any]
        Prediction results
    """
    # Generate a random score
    np.random.seed(hash(str(patient_data.iloc[0]['patient_id'])) % 2**32)
    risk_score = np.random.beta(2, 5)  # Beta distribution for realistic scores
    
    # Generate feature contributions
    feature_names = ['age', 'gender', 'previous_hospitalization', 'medication_adherence', 
                    'symptom_severity', 'substance_use', 'family_support', 
                    'employment_status', 'duration_untreated']
    
    contributions = np.random.normal(0, 0.1, len(feature_names))
    contributions[0] = 0.15  # Make age more important
    contributions[3] = -0.2  # Make medication_adherence protective
    contributions[4] = 0.25  # Make symptom_severity a risk factor
    
    # Scale contributions to sum to risk_score
    contributions = contributions / np.sum(np.abs(contributions)) * risk_score
    
    # Create a prediction result
    prediction = {
        'score': risk_score,
        'prediction': 1 if risk_score >= threshold else 0,
        'feature_contributions': dict(zip(feature_names, contributions)),
        'model': model_name,
        'timestamp': pd.Timestamp.now()
    }
    
    # Add top risk factors and protective factors
    top_risk_factors = [feature_names[i] for i in np.argsort(contributions)[-3:] if contributions[i] > 0]
    protective_factors = [feature_names[i] for i in np.argsort(contributions)[:3] if contributions[i] < 0]
    
    prediction['top_risk_factors'] = top_risk_factors
    prediction['protective_factors'] = protective_factors
    
    return prediction


def generate_sample_metrics(state: Dict[str, Any]) -> None:
    """
    Generate sample model metrics for demonstration purposes.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Current model metrics
    state['model_metrics'] = {
        'accuracy': 0.82,
        'precision': 0.76,
        'recall': 0.79,
        'f1': 0.775,
        'auc': 0.85,
        'clinical_utility': 0.72
    }
    
    # Baseline metrics for comparison
    state['baseline_metrics'] = {
        'accuracy': 0.78,
        'precision': 0.71,
        'recall': 0.74,
        'f1': 0.725,
        'auc': 0.81,
        'clinical_utility': 0.65
    }


def generate_sample_risk_timeline(state: Dict[str, Any]) -> None:
    """
    Generate sample risk timeline data for demonstration purposes.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Create sample timeline data
    np.random.seed(42)
    
    # Get patient IDs
    if 'patients_df' in state:
        patient_ids = state['patients_df']['patient_id'].unique()[:5]  # Use first 5 patients
    else:
        patient_ids = [f"P{i:03d}" for i in range(1, 6)]
    
    # Create timeline for each patient
    timeline_data = []
    
    for patient_id in patient_ids:
        # Generate timestamps (weekly for 3 months)
        start_date = pd.Timestamp.now() - pd.Timedelta(days=90)
        dates = [start_date + pd.Timedelta(days=7*i) for i in range(13)]
        
        # Generate risk scores with trend
        base_risk = np.random.uniform(0.3, 0.7)
        trend = np.random.choice([-0.01, 0, 0.01])  # Trend direction
        
        for i, date in enumerate(dates):
            # Calculate risk score with trend and noise
            risk = base_risk + trend * i + np.random.normal(0, 0.05)
            risk = max(0, min(1, risk))  # Clip to [0, 1]
            
            timeline_data.append({
                'patient_id': patient_id,
                'timestamp': date,
                'risk_score': risk
            })
    
    # Create DataFrame
    state['risk_timeline_df'] = pd.DataFrame(timeline_data)


def load_sample_data(state: Dict[str, Any]) -> None:
    """
    Load sample data for demonstration purposes.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    # Create sample patient data
    np.random.seed(42)
    n_patients = 100
    
    patient_data = []
    
    for i in range(1, n_patients + 1):
        # Generate patient ID
        patient_id = f"P{i:03d}"
        
        # Generate demographics
        age = np.random.randint(16, 65)
        gender = np.random.choice(['Male', 'Female'])
        
        # Generate clinical features
        previous_hospitalization = np.random.choice([0, 1], p=[0.7, 0.3])
        medication_adherence = np.random.uniform(0, 1)
        symptom_severity = np.random.uniform(0, 1)
        substance_use = np.random.choice([0, 1], p=[0.6, 0.4])
        family_support = np.random.uniform(0, 1)
        employment_status = np.random.choice(['Employed', 'Unemployed', 'Student'])
        duration_untreated = np.random.randint(0, 104)  # weeks
        
        # Generate admission date
        admission_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 365))
        
        # Add to patient data
        patient_data.append({
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'previous_hospitalization': previous_hospitalization,
            'medication_adherence': medication_adherence,
            'symptom_severity': symptom_severity,
            'substance_use': substance_use,
            'family_support': family_support,
            'employment_status': employment_status,
            'duration_untreated': duration_untreated,
            'admission_date': admission_date
        })
    
    # Create DataFrame
    state['patients_df'] = pd.DataFrame(patient_data)
    
    # Generate sample outcomes
    outcome_data = []
    
    for patient in patient_data:
        # Calculate outcome probability based on features
        p_relapse = (0.2 + 
                     0.1 * patient['previous_hospitalization'] + 
                     0.3 * patient['symptom_severity'] - 
                     0.2 * patient['medication_adherence'] -
                     0.1 * patient['family_support'] +
                     0.1 * patient['substance_use'])
        
        # Clip probability
        p_relapse = max(0.1, min(0.9, p_relapse))
        
        # Generate outcome
        outcome = np.random.choice([0, 1], p=[1-p_relapse, p_relapse])
        
        # Add to outcome data
        outcome_data.append({
            'patient_id': patient['patient_id'],
            'outcome': outcome,
            'outcome_date': patient['admission_date'] + pd.Timedelta(days=np.random.randint(30, 180))
        })
    
    # Create DataFrame
    state['outcomes_df'] = pd.DataFrame(outcome_data)
    
    # Set available models
    state['available_models'] = [
        "Ensemble Model (Primary)",
        "Logistic Regression",
        "Gradient Boosting",
        "Neural Network"
    ]
    
    state['default_model'] = "Ensemble Model (Primary)"
    
    # Set model descriptions
    state['model_descriptions'] = {
        "Ensemble Model (Primary)": "Combined model focusing on high-risk cases with time-decay weighting.",
        "Logistic Regression": "Simple interpretable model using primary clinical features.",
        "Gradient Boosting": "Advanced model with higher accuracy but lower interpretability.",
        "Neural Network": "Experimental model with temporal feature modeling."
    }
    
    # Set default threshold
    state['default_threshold'] = 0.6
    
    # Set patient info columns for display
    state['patient_info_cols'] = ['age', 'gender', 'admission_date']
    
    # Generate sample metrics
    generate_sample_metrics(state)
    
    # Generate sample risk timeline
    generate_sample_risk_timeline(state)
    
    # Add success message
    st.success("Sample data loaded successfully.")

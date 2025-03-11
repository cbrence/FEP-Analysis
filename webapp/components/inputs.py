"""
Input components for the FEP analysis web application.

This module provides reusable Streamlit components for user input
such as selectors, sliders, and form inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import json


def demographic_inputs(default_values: Optional[Dict[str, Any]] = None,
                       key: str = "demographics") -> Dict[str, Any]:
    """
    Create input widgets for patient demographic information.
    
    Parameters
    ----------
    default_values : Optional[Dict[str, Any]], default=None
        Dictionary mapping field names to default values
    key : str, default="demographics"
        Base key for the Streamlit components
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping field names to input values
    """
    if default_values is None:
        default_values = {}
    
    st.subheader("Demographic Information")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # First column
    with col1:
        # Age
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=90,
            value=default_values.get("Age", 25),
            step=1,
            key=f"{key}_age"
        )
        
        # Gender
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(default_values.get("Gender", "Male")),
            key=f"{key}_gender"
        )
        
        # Ethnicity
        ethnicity = st.selectbox(
            "Ethnicity",
            options=["White", "Black", "Asian", "Hispanic", "Mixed", "Other"],
            index=["White", "Black", "Asian", "Hispanic", "Mixed", "Other"].index(
                default_values.get("Ethnicity", "White")),
            key=f"{key}_ethnicity"
        )
    
    # Second column
    with col2:
        # Education
        education = st.selectbox(
            "Education",
            options=["Primary", "Secondary", "College", "University", "Postgraduate"],
            index=["Primary", "Secondary", "College", "University", "Postgraduate"].index(
                default_values.get("Education", "Secondary")),
            key=f"{key}_education"
        )
        
        # Education years
        education_years = st.number_input(
            "Education Years",
            min_value=0,
            max_value=25,
            value=default_values.get("Education_Num", 12),
            step=1,
            key=f"{key}_education_years"
        )
        
        # Accommodation
        accommodation = st.selectbox(
            "Accommodation",
            options=["Independent", "Family Home", "Supported", "Homeless", "Other"],
            index=["Independent", "Family Home", "Supported", "Homeless", "Other"].index(
                default_values.get("Accommodation", "Family Home")),
            key=f"{key}_accommodation"
        )
    
    # Additional demographics in expander
    with st.expander("Additional Demographics"):
        # Citizenship
        citizenship = st.selectbox(
            "Citizenship",
            options=["Citizen", "Permanent Resident", "Temporary Resident", "Other"],
            index=["Citizen", "Permanent Resident", "Temporary Resident", "Other"].index(
                default_values.get("Citizenship", "Citizen")),
            key=f"{key}_citizenship"
        )
        
        # Relationship status
        relationship = st.selectbox(
            "Relationship Status",
            options=["Single", "Married", "Partnered", "Separated", "Divorced", "Widowed"],
            index=["Single", "Married", "Partnered", "Separated", "Divorced", "Widowed"].index(
                default_values.get("Relationship", "Single")),
            key=f"{key}_relationship"
        )
        
        # Household
        household = st.selectbox(
            "Household",
            options=["Lives Alone", "With Partner", "With Family", "Shared Housing", "Other"],
            index=["Lives Alone", "With Partner", "With Family", "Shared Housing", "Other"].index(
                default_values.get("Household", "With Family")),
            key=f"{key}_household"
        )
        
        # Employment
        employment = st.selectbox(
            "Employment",
            options=["Employed", "Unemployed", "Student", "Retired", "Unable to Work"],
            index=["Employed", "Unemployed", "Student", "Retired", "Unable to Work"].index(
                default_values.get("M0_Emp", "Unemployed")),
            key=f"{key}_employment"
        )
    
    # Combine all inputs into a single dictionary
    values = {
        "Age": age,
        "Gender": gender,
        "Ethnicity": ethnicity,
        "Education": education,
        "Education_Num": education_years,
        "Accommodation": accommodation,
        "Citizenship": citizenship,
        "Relationship": relationship,
        "Household": household,
        "M0_Emp": employment
    }
    
    return values


def clinical_inputs(default_values: Optional[Dict[str, Any]] = None,
                  key: str = "clinical") -> Dict[str, Any]:
    """
    Create input widgets for patient clinical information.
    
    Parameters
    ----------
    default_values : Optional[Dict[str, Any]], default=None
        Dictionary mapping field names to default values
    key : str, default="clinical"
        Base key for the Streamlit components
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping field names to input values
    """
    if default_values is None:
        default_values = {}
    
    st.subheader("Clinical Information")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # First column - Substance use and hospitalization
    with col1:
        # Hospital admission
        admitted_hosp = st.radio(
            "Previous Hospital Admission",
            options=["Yes", "No"],
            index=["Yes", "No"].index(default_values.get("Admitted_Hosp", "No")),
            key=f"{key}_admitted_hosp",
            horizontal=True
        )
        
        # Substance use
        drugs = st.radio(
            "Drug Use",
            options=["Yes", "No"],
            index=["Yes", "No"].index(default_values.get("Drugs", "No")),
            key=f"{key}_drugs",
            horizontal=True
        )
        
        # Alcohol use
        alcohol = st.radio(
            "Problematic Alcohol Use",
            options=["Yes", "No"],
            index=["Yes", "No"].index(default_values.get("Alcohol", "No")),
            key=f"{key}_alcohol",
            horizontal=True
        )
    
    # Second column - Depression
    with col2:
        # Depression severity
        depression_severity = st.select_slider(
            "Depression Severity",
            options=["None", "Mild", "Moderate", "Severe"],
            value=default_values.get("Depression_Severity", "None"),
            key=f"{key}_depression_severity"
        )
        
        # Depression scale
        depression_scale = st.slider(
            "Depression Scale Score",
            min_value=0,
            max_value=10,
            value=default_values.get("Depression_Sev_Scale", 0),
            step=1,
            key=f"{key}_depression_scale"
        )
    
    # Cohort information
    cohort = st.selectbox(
        "Cohort",
        options=["First Episode", "Early Intervention", "Control", "Other"],
        index=["First Episode", "Early Intervention", "Control", "Other"].index(
            default_values.get("Cohort", "First Episode")),
        key=f"{key}_cohort"
    )
    
    # Additional clinical information
    with st.expander("Additional Clinical Information"):
        # Parental presence
        parent = st.selectbox(
            "Parental Presence",
            options=["Both", "One", "None"],
            index=["Both", "One", "None"].index(default_values.get("Parent", "Both")),
            key=f"{key}_parent"
        )
        
        # Duration of untreated psychosis (in weeks)
        dup_weeks = st.number_input(
            "Duration of Untreated Psychosis (weeks)",
            min_value=0,
            max_value=260,  # 5 years in weeks
            value=default_values.get("DUP_Weeks", 8),
            step=1,
            key=f"{key}_dup_weeks"
        )
        
        # Age of onset
        age_onset = st.number_input(
            "Age of Onset",
            min_value=10,
            max_value=90,
            value=default_values.get("Age_Onset", 22),
            step=1,
            key=f"{key}_age_onset"
        )
    
    # Convert depression severity to scale if needed
    if depression_severity == "None":
        depression_sev_value = 0
    elif depression_severity == "Mild":
        depression_sev_value = 1
    elif depression_severity == "Moderate":
        depression_sev_value = 2
    else:  # Severe
        depression_sev_value = 3
    
    # Combine all inputs into a single dictionary
    values = {
        "Admitted_Hosp": admitted_hosp,
        "Drugs": drugs,
        "Alcohol": alcohol,
        "Depression_Severity": depression_severity,
        "Depression_Sev_Scale": depression_scale,
        "Cohort": cohort,
        "Parent": parent,
        "DUP_Weeks": dup_weeks,
        "Age_Onset": age_onset,
        "Depression_Sev_Value": depression_sev_value
    }
    
    return values


def panss_positive_inputs(default_values: Optional[Dict[str, int]] = None,
                        key: str = "panss_positive") -> Dict[str, int]:
    """
    Create input widgets for PANSS Positive Scale items.
    
    Parameters
    ----------
    default_values : Optional[Dict[str, int]], default=None
        Dictionary mapping PANSS item names to default values
    key : str, default="panss_positive"
        Base key for the Streamlit components
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping PANSS item names to values
    """
    if default_values is None:
        default_values = {}
    
    st.subheader("PANSS Positive Scale")
    
    # Define PANSS positive items
    panss_positive_items = {
        "P1": "Delusions",
        "P2": "Conceptual disorganization",
        "P3": "Hallucinatory behavior",
        "P4": "Excitement",
        "P5": "Grandiosity",
        "P6": "Suspiciousness/persecution",
        "P7": "Hostility"
    }
    
    # Create a dictionary to store values
    values = {}
    
    # Create inputs for each item
    for item_code, item_name in panss_positive_items.items():
        field_key = f"M0_PANSS_{item_code}"
        
        # Create a slider for each item
        values[field_key] = st.slider(
            f"{item_code}. {item_name}",
            min_value=1,
            max_value=7,
            value=default_values.get(field_key, 1),
            step=1,
            help="1=Absent, 2=Minimal, 3=Mild, 4=Moderate, 5=Moderate-Severe, 6=Severe, 7=Extreme",
            key=f"{key}_{item_code.lower()}"
        )
    
    # Display total score
    total_score = sum(values.values())
    st.metric("Positive Scale Total", total_score)
    st.caption("Range: 7-49, Higher scores indicate more severe symptoms")
    
    return values


def panss_negative_inputs(default_values: Optional[Dict[str, int]] = None,
                        key: str = "panss_negative") -> Dict[str, int]:
    """
    Create input widgets for PANSS Negative Scale items.
    
    Parameters
    ----------
    default_values : Optional[Dict[str, int]], default=None
        Dictionary mapping PANSS item names to default values
    key : str, default="panss_negative"
        Base key for the Streamlit components
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping PANSS item names to values
    """
    if default_values is None:
        default_values = {}
    
    st.subheader("PANSS Negative Scale")
    
    # Define PANSS negative items
    panss_negative_items = {
        "N1": "Blunted affect",
        "N2": "Emotional withdrawal",
        "N3": "Poor rapport",
        "N4": "Passive/apathetic social withdrawal",
        "N5": "Difficulty in abstract thinking",
        "N6": "Lack of spontaneity & flow of conversation",
        "N7": "Stereotyped thinking"
    }
    
    # Create a dictionary to store values
    values = {}
    
    # Create inputs for each item
    for item_code, item_name in panss_negative_items.items():
        field_key = f"M0_PANSS_{item_code}"
        
        # Create a slider for each item
        values[field_key] = st.slider(
            f"{item_code}. {item_name}",
            min_value=1,
            max_value=7,
            value=default_values.get(field_key, 1),
            step=1,
            help="1=Absent, 2=Minimal, 3=Mild, 4=Moderate, 5=Moderate-Severe, 6=Severe, 7=Extreme",
            key=f"{key}_{item_code.lower()}"
        )
    
    # Display total score
    total_score = sum(values.values())
    st.metric("Negative Scale Total", total_score)
    st.caption("Range: 7-49, Higher scores indicate more severe symptoms")
    
    return values


def panss_general_inputs(default_values: Optional[Dict[str, int]] = None,
                       key: str = "panss_general") -> Dict[str, int]:
    """
    Create input widgets for PANSS General Psychopathology Scale items.
    
    Parameters
    ----------
    default_values : Optional[Dict[str, int]], default=None
        Dictionary mapping PANSS item names to default values
    key : str, default="panss_general"
        Base key for the Streamlit components
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping PANSS item names to values
    """
    if default_values is None:
        default_values = {}
    
    st.subheader("PANSS General Psychopathology Scale")
    
    # Define PANSS general items
    panss_general_items = {
        "G1": "Somatic concern",
        "G2": "Anxiety",
        "G3": "Guilt feelings",
        "G4": "Tension",
        "G5": "Mannerisms & posturing",
        "G6": "Depression",
        "G7": "Motor retardation",
        "G8": "Uncooperativeness",
        "G9": "Unusual thought content",
        "G10": "Disorientation",
        "G11": "Poor attention",
        "G12": "Lack of judgment & insight",
        "G13": "Disturbance of volition",
        "G14": "Poor impulse control",
        "G15": "Preoccupation",
        "G16": "Active social avoidance"
    }
    
    # Create a dictionary to store values
    values = {}
    
    # Create tabs for better organization due to many items
    tab1, tab2 = st.tabs(["Items G1-G8", "Items G9-G16"])
    
    with tab1:
        # First half of general items
        for item_code in ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]:
            item_name = panss_general_items[item_code]
            field_key = f"M0_PANSS_{item_code}"
            
            # Create a slider for each item
            values[field_key] = st.slider(
                f"{item_code}. {item_name}",
                min_value=1,
                max_value=7,
                value=default_values.get(field_key, 1),
                step=1,
                help="1=Absent, 2=Minimal, 3=Mild, 4=Moderate, 5=Moderate-Severe, 6=Severe, 7=Extreme",
                key=f"{key}_{item_code.lower()}"
            )
    
    with tab2:
        # Second half of general items
        for item_code in ["G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]:
            item_name = panss_general_items[item_code]
            field_key = f"M0_PANSS_{item_code}"
            
            # Create a slider for each item
            values[field_key] = st.slider(
                f"{item_code}. {item_name}",
                min_value=1,
                max_value=7,
                value=default_values.get(field_key, 1),
                step=1,
                help="1=Absent, 2=Minimal, 3=Mild, 4=Moderate, 5=Moderate-Severe, 6=Severe, 7=Extreme",
                key=f"{key}_{item_code.lower()}"
            )
    
    # Display total score
    total_score = sum(values.values())
    st.metric("General Psychopathology Scale Total", total_score)
    st.caption("Range: 16-112, Higher scores indicate more severe symptoms")
    
    return values


def patient_selector(patients_df: pd.DataFrame,
                    key: str = "patient_selector",
                    label: str = "Select Patient",
                    allow_multiple: bool = False,
                    patient_id_col: str = "patient_id",
                    patient_info_cols: Optional[List[str]] = None) -> Union[str, List[str]]:
    """
    Create a patient selection dropdown with optional patient information display.
    
    Parameters
    ----------
    patients_df : pd.DataFrame
        DataFrame containing patient information
    key : str, default="patient_selector"
        Unique key for the Streamlit component
    label : str, default="Select Patient"
        Label for the selection dropdown
    allow_multiple : bool, default=False
        Whether to allow selection of multiple patients
    patient_id_col : str, default="patient_id"
        Column name for patient identifiers
    patient_info_cols : Optional[List[str]], default=None
        Additional columns to display as patient information
        
    Returns
    -------
    Union[str, List[str]]
        Selected patient ID(s)
    """
    # Get unique patient IDs
    patient_ids = sorted(patients_df[patient_id_col].unique())
    
    # Create selector component
    if allow_multiple:
        selected_patients = st.multiselect(
            label=label,
            options=patient_ids,
            key=key
        )
    else:
        selected_patients = st.selectbox(
            label=label,
            options=patient_ids,
            key=key
        )
    
    # Display patient information if requested
    if patient_info_cols and not allow_multiple and selected_patients:
        st.subheader("Patient Information")
        
        # Get patient data
        patient_data = patients_df[patients_df[patient_id_col] == selected_patients].iloc[0]
        
        # Create columns for layout
        cols = st.columns(min(3, len(patient_info_cols)))
        
        # Display patient information
        for i, col_name in enumerate(patient_info_cols):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Format value based on type
                val = patient_data[col_name]
                if pd.api.types.is_datetime64_any_dtype(patient_data[col_name]):
                    val = val.strftime('%Y-%m-%d')
                elif isinstance(val, float):
                    val = f"{val:.2f}"
                
                st.metric(label=col_name.replace('_', ' ').title(), value=val)
    
    return selected_patients


def date_range_selector(default_days_back: int = 30,
                       max_days_back: int = 365,
                       key: str = "date_range") -> Tuple[datetime, datetime]:
    """
    Create a date range selector for time-based filtering.
    
    Parameters
    ----------
    default_days_back : int, default=30
        Default number of days to look back
    max_days_back : int, default=365
        Maximum number of days to allow looking back
    key : str, default="date_range"
        Unique key for the Streamlit component
        
    Returns
    -------
    Tuple[datetime, datetime]
        (start_date, end_date)
    """
    # Calculate default dates
    today = datetime.now().date()
    default_start = today - timedelta(days=default_days_back)
    min_date = today - timedelta(days=max_days_back)
    
    # Create date selection containers
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=today,
            key=f"{key}_start"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=today,
            min_value=min_date,
            max_value=today,
            key=f"{key}_end"
        )
    
    # Ensure start date is before end date
    if start_date > end_date:
        st.warning("Start date must be before end date. Adjusting automatically.")
        start_date = end_date - timedelta(days=1)
    
    # Convert to datetime for consistent handling
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    return start_datetime, end_datetime


def feature_selector(available_features: List[str],
                    default_selected: Optional[List[str]] = None,
                    key: str = "feature_selector",
                    label: str = "Select Features",
                    feature_groups: Optional[Dict[str, List[str]]] = None,
                    max_features: Optional[int] = None) -> List[str]:
    """
    Create a feature selection component with optional grouping.
    
    Parameters
    ----------
    available_features : List[str]
        List of all available features
    default_selected : Optional[List[str]], default=None
        List of features to select by default
    key : str, default="feature_selector"
        Unique key for the Streamlit component
    label : str, default="Select Features"
        Label for the component
    feature_groups : Optional[Dict[str, List[str]]], default=None
        Dictionary mapping group names to lists of feature names
    max_features : Optional[int], default=None
        Maximum number of features that can be selected
        
    Returns
    -------
    List[str]
        List of selected feature names
    """
    # Create expander for feature selection
    with st.expander(label):
        # Determine if we should use groups
        if feature_groups:
            # Create tabs for each group
            tabs = st.tabs(list(feature_groups.keys()) + ["All Features"])
            
            selected_features = []
            
            # Create feature checkboxes for each group
            for i, (group_name, group_features) in enumerate(feature_groups.items()):
                with tabs[i]:
                    # Select all button for this group
                    if st.button(f"Select All in {group_name}", key=f"{key}_select_all_{i}"):
                        group_selected = group_features
                    else:
                        # Create checkboxes for features in this group
                        group_selected = []
                        for feature in group_features:
                            is_default = default_selected and feature in default_selected
                            if st.checkbox(feature, value=is_default, key=f"{key}_{feature}"):
                                group_selected.append(feature)
                    
                    selected_features.extend(group_selected)
            
            # All features tab
            with tabs[-1]:
                # Create a multiselect for all features
                all_selected = st.multiselect(
                    "Select Features",
                    options=available_features,
                    default=default_selected or [],
                    key=f"{key}_all"
                )
                
                # If user selects from All Features tab, override previous selections
                if all_selected:
                    selected_features = all_selected
        else:
            # Simple multiselect without groups
            selected_features = st.multiselect(
                "Select Features",
                options=available_features,
                default=default_selected or [],
                key=key
            )
        
        # Show warning if too many features selected
        if max_features and len(selected_features) > max_features:
            st.warning(f"Too many features selected. Maximum is {max_features}.")
            selected_features = selected_features[:max_features]
        
        # Show count of selected features
        st.caption(f"Selected {len(selected_features)} features")
    
    return selected_features


def model_selector(available_models: List[str],
                  default_model: Optional[str] = None,
                  key: str = "model_selector",
                  label: str = "Select Model",
                  model_descriptions: Optional[Dict[str, str]] = None,
                  allow_multiple: bool = False) -> Union[str, List[str]]:
    """
    Create a model selection dropdown with optional descriptions.
    
    Parameters
    ----------
    available_models : List[str]
        List of available model names
    default_model : Optional[str], default=None
        Default model to select
    key : str, default="model_selector"
        Unique key for the Streamlit component
    label : str, default="Select Model"
        Label for the selection dropdown
    model_descriptions : Optional[Dict[str, str]], default=None
        Dictionary mapping model names to descriptions
    allow_multiple : bool, default=False
        Whether to allow selection of multiple models
        
    Returns
    -------
    Union[str, List[str]]
        Selected model name(s)
    """
    # Create selector component
    if allow_multiple:
        selected_models = st.multiselect(
            label=label,
            options=available_models,
            default=[default_model] if default_model else [],
            key=key
        )
    else:
        selected_models = st.selectbox(
            label=label,
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key=key
        )
    
    # Display model description if available
    if model_descriptions and not allow_multiple and selected_models:
        if selected_models in model_descriptions:
            st.caption(model_descriptions[selected_models])
    
    return selected_models


def threshold_slider(default_value: float = 0.5,
                   min_value: float = 0.0,
                   max_value: float = 1.0,
                   step: float = 0.01,
                   key: str = "threshold_slider",
                   label: str = "Prediction Threshold",
                   format: str = "%.2f",
                   help_text: Optional[str] = None) -> float:
    """
    Create a slider for selecting a prediction threshold.
    
    Parameters
    ----------
    default_value : float, default=0.5
        Default threshold value
    min_value : float, default=0.0
        Minimum allowed threshold value
    max_value : float, default=1.0
        Maximum allowed threshold value
    step : float, default=0.01
        Step size for the slider
    key : str, default="threshold_slider"
        Unique key for the Streamlit component
    label : str, default="Prediction Threshold"
        Label for the slider
    format : str, default="%.2f"
        Format string for the displayed value
    help_text : Optional[str], default=None
        Help text to display with the slider
        
    Returns
    -------
    float
        Selected threshold value
    """
    # Create threshold slider
    threshold = st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        format=format,
        key=key,
        help=help_text
    )
    
    return threshold


def parameter_inputs(param_config: Dict[str, Dict[str, Any]],
                   default_values: Optional[Dict[str, Any]] = None,
                   key: str = "params",
                   title: str = "Model Parameters") -> Dict[str, Any]:
    """
    Create input widgets for model parameters based on a configuration.
    
    Parameters
    ----------
    param_config : Dict[str, Dict[str, Any]]
        Configuration for parameter inputs. Each key is a parameter name,
        and each value is a dictionary with keys:
        - 'type': input type ('slider', 'number', 'select', 'checkbox', etc.)
        - 'label': display label
        - 'min_value', 'max_value', 'step': for numeric inputs
        - 'options': for select inputs
        - other type-specific options
    default_values : Optional[Dict[str, Any]], default=None
        Dictionary mapping parameter names to default values
    key : str, default="params"
        Base key for the Streamlit components
    title : str, default="Model Parameters"
        Title for the parameter inputs section
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to input values
    """
    # Create container for parameters
    st.subheader(title)
    param_values = {}
    
    # Set default values
    if default_values is None:
        default_values = {}
    
    # Create input for each parameter
    for param_name, config in param_config.items():
        # Get parameter type and label
        param_type = config.get('type', 'number')
        param_label = config.get('label', param_name)
        
        # Get default value
        default = default_values.get(param_name, config.get('default'))
        
        # Create appropriate input widget based on type
        if param_type == 'slider':
            param_values[param_name] = st.slider(
                label=param_label,
                min_value=config.get('min_value', 0.0),
                max_value=config.get('max_value', 1.0),
                value=default,
                step=config.get('step', 0.01),
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'number':
            param_values[param_name] = st.number_input(
                label=param_label,
                min_value=config.get('min_value'),
                max_value=config.get('max_value'),
                value=default,
                step=config.get('step', 1),
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'select':
            param_values[param_name] = st.selectbox(
                label=param_label,
                options=config.get('options', []),
                index=config.get('options', []).index(default) if default in config.get('options', []) else 0,
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'multiselect':
            param_values[param_name] = st.multiselect(
                label=param_label,
                options=config.get('options', []),
                default=default if isinstance(default, list) else [],
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'checkbox':
            param_values[param_name] = st.checkbox(
                label=param_label,
                value=default if default is not None else False,
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'radio':
            param_values[param_name] = st.radio(
                label=param_label,
                options=config.get('options', []),
                index=config.get('options', []).index(default) if default in config.get('options', []) else 0,
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'text':
            param_values[param_name] = st.text_input(
                label=param_label,
                value=default if default is not None else "",
                key=f"{key}_{param_name}"
            )
        
        elif param_type == 'textarea':
            param_values[param_name] = st.text_area(
                label=param_label,
                value=default if default is not None else "",
                key=f"{key}_{param_name}"
            )
        
        # Add help text if provided
        if 'help' in config:
            st.caption(config['help'])
    
    return param_values

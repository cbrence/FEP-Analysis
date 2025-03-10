"""
Feature importance page for the FEP analysis web application.

This module provides visualizations and analysis of feature importance
in the FEP outcome prediction models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

from ..components.inputs import model_selector, feature_selector
from ..components.results import display_feature_importance
from ..components.risk_display import display_risk_composition


def render_feature_importance_page(state: Dict[str, Any]) -> None:
    """
    Render the feature importance page of the FEP analysis web application.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.title("Feature Importance Analysis")
    st.markdown("""
    This page provides insights into which factors contribute most to the predicted 
    outcomes in First Episode Psychosis patients.
    """)
    
    # Check if data is loaded
    if 'patients_df' not in state or state['patients_df'] is None:
        st.warning("No data loaded. Please load patient data to continue.")
        return
    
    # Create tabs for different views
    tabs = st.tabs([
        "Global Feature Importance", 
        "Feature Groups", 
        "Feature Correlations",
        "Risk Composition"
    ])
    
    # Global Feature Importance tab
    with tabs[0]:
        render_global_importance_section(state)
    
    # Feature Groups tab
    with tabs[1]:
        render_feature_groups_section(state)
    
    # Feature Correlations tab
    with tabs[2]:
        render_feature_correlations_section(state)
    
    # Risk Composition tab
    with tabs[3]:
        render_risk_composition_section(state)


def render_global_importance_section(state: Dict[str, Any]) -> None:
    """
    Render the global feature importance section.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.subheader("Global Feature Importance")
    st.markdown("""
    This visualization shows the overall importance of each feature across all predictions,
    helping to identify which factors are most influential in the model.
    """)
    
    # Model selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = model_selector(
            state.get('available_models', ["Default Model"]),
            default_model=state.get('default_model'),
            key="fi_model_selector",
            label="Select Model",
            model_descriptions=state.get('model_descriptions')
        )
    
    with col2:
        importance_method = st.selectbox(
            "Importance Method",
            options=["Model", "SHAP", "Permutation"],
            index=0,
            key="fi_method_selector"
        )
    
    # Get feature importance data
    feature_importance = get_feature_importance(state, selected_model, importance_method)
    
    if feature_importance is not None:
        # Display feature importance
        display_feature_importance(
            feature_names=feature_importance['features'],
            importance_values=feature_importance['importance'],
            title=f"Feature Importance ({importance_method})",
            top_n=15,
            method=importance_method.lower()
        )
        
        # Add description
        if importance_method == "Model":
            st.info("""
            **Model Feature Importance** shows the importance of features based on the model's internal 
            feature weighting. This is most accurate for tree-based models like Gradient Boosting.
            """)
        elif importance_method == "SHAP":
            st.info("""
            **SHAP Values** (SHapley Additive exPlanations) show the contribution of each feature to 
            predictions, accounting for feature interactions and providing consistent interpretations.
            """)
        elif importance_method == "Permutation":
            st.info("""
            **Permutation Importance** measures the decrease in model performance when a feature is 
            randomly shuffled, indicating how much the model depends on that feature.
            """)
    else:
        st.warning("Feature importance data not available for the selected model and method.")


def render_feature_groups_section(state: Dict[str, Any]) -> None:
    """
    Render the feature groups section.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.subheader("Feature Group Analysis")
    st.markdown("""
    This visualization groups related features to show the importance of different 
    categories of factors in predicting outcomes.
    """)
    
    # Get feature groups
    feature_groups = get_feature_groups(state)
    
    if feature_groups is None:
        st.warning("Feature group information not available.")
        return
    
    # Create group selector
    selected_groups = st.multiselect(
        "Select Feature Groups to Display",
        options=list(feature_groups.keys()),
        default=list(feature_groups.keys()),
        key="fi_group_selector"
    )
    
    # Filter groups
    filtered_groups = {k: v for k, v in feature_groups.items() if k in selected_groups}
    
    # Get group importance
    group_importance = get_group_importance(state, filtered_groups)
    
    if group_importance is not None:
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_indices = np.argsort(group_importance['importance'])
        sorted_groups = [group_importance['groups'][i] for i in sorted_indices]
        sorted_importance = [group_importance['importance'][i] for i in sorted_indices]
        sorted_counts = [group_importance['counts'][i] for i in sorted_indices]
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_groups)))
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_groups, sorted_importance, color=colors)
        
        # Add feature counts
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"({sorted_counts[i]} features)",
                   va='center', fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Group Importance')
        
        # Display chart
        st.pyplot(fig)
        
        # Display group descriptions
        st.subheader("Feature Group Descriptions")
        
        # Create columns for descriptions
        cols = st.columns(2)
        
        for i, group in enumerate(sorted_groups):
            with cols[i % 2]:
                features = feature_groups[group]
                st.markdown(f"**{group}** ({len(features)} features)")
                st.markdown(", ".join(features[:5]) + 
                         ("..." if len(features) > 5 else ""))
                st.markdown("---")
    else:
        st.warning("Group importance data not available.")


def render_feature_correlations_section(state: Dict[str, Any]) -> None:
    """
    Render the feature correlations section.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.subheader("Feature Correlations")
    st.markdown("""
    This visualization shows correlations between different features, helping to identify 
    redundant information and potential feature interactions.
    """)
    
    # Get available features
    if 'patients_df' in state:
        available_features = state['patients_df'].columns.tolist()
        # Remove non-numeric features
        available_features = [col for col in available_features 
                             if pd.api.types.is_numeric_dtype(state['patients_df'][col])]
    else:
        available_features = []
    
    # Feature selector
    selected_features = feature_selector(
        available_features=available_features,
        default_selected=available_features[:10] if len(available_features) > 10 else available_features,
        key="fi_correlation_selector",
        label="Select Features for Correlation Analysis",
        max_features=15
    )
    
    if not selected_features:
        st.warning("Please select at least two features for correlation analysis.")
        return
    
    if len(selected_features) < 2:
        st.warning("Please select at least two features for correlation analysis.")
        return
    
    # Calculate correlations
    if 'patients_df' in state:
        corr = state['patients_df'][selected_features].corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
        
        # Add ticks and labels
        ax.set_xticks(np.arange(len(selected_features)))
        ax.set_yticks(np.arange(len(selected_features)))
        ax.set_xticklabels(selected_features, rotation=45, ha="right")
        ax.set_yticklabels(selected_features)
        
        # Add correlation values
        for i in range(len(selected_features)):
            for j in range(len(selected_features)):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                              ha="center", va="center", 
                              color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
        
        # Set title
        ax.set_title("Feature Correlation Matrix")
        
        # Display chart
        st.pyplot(fig)
        
        # Add interpretation guidance
        st.info("""
        **Interpreting Correlations:**
        - **Values close to 1.0**: Strong positive correlation (features increase together)
        - **Values close to -1.0**: Strong negative correlation (one increases as the other decreases)
        - **Values close to 0.0**: Little to no correlation (features are independent)
        
        Strong correlations may indicate redundant features that could be simplified in the model.
        """)
    else:
        st.warning("Patient data not available for correlation analysis.")


def render_risk_composition_section(state: Dict[str, Any]) -> None:
    """
    Render the risk composition section.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    """
    st.subheader("Risk Score Composition")
    st.markdown("""
    This visualization shows how different features contribute to individual risk scores,
    helping to understand which factors drive predictions for specific patients.
    """)
    
    # Model selector
    selected_model = model_selector(
        state.get('available_models', ["Default Model"]),
        default_model=state.get('default_model'),
        key="fi_composition_model_selector",
        label="Select Model",
        model_descriptions=state.get('model_descriptions')
    )
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        n_features = st.slider(
            "Number of Top Features",
            min_value=3,
            max_value=10,
            value=5,
            key="fi_composition_n_features"
        )
    
    with col2:
        normalize = st.checkbox(
            "Normalize Contributions",
            value=False,
            key="fi_composition_normalize"
        )
    
    # Get feature contributions and risk scores
    contributions, risk_scores = get_feature_contributions(state, selected_model)
    
    if contributions is not None and risk_scores is not None:
        # Display risk composition
        display_risk_composition(
            feature_contributions=contributions,
            risk_scores=risk_scores,
            title=f"Risk Score Composition ({selected_model})",
            n_features=n_features,
            normalize=normalize
        )
        
        # Add explanation
        st.info("""
        **Understanding Risk Composition:**
        - **Positive values (blue)** indicate factors that increase the predicted risk.
        - **Negative values (orange)** indicate protective factors that decrease the risk.
        - The longer the bar, the stronger the influence of that feature on the prediction.
        
        Different patients may have different risk factors, reflecting the personalized 
        nature of the prediction model.
        """)
    else:
        st.warning("Feature contribution data not available for the selected model.")
    
    # Add patient-specific analysis option
    st.subheader("Individual Patient Analysis")
    
    # Patient selector
    if 'patients_df' in state:
        selected_patient = st.selectbox(
            "Select Patient for Individual Analysis",
            options=state['patients_df']['patient_id'].unique(),
            key="fi_individual_patient"
        )
        
        if st.button("Analyze Patient", key="fi_analyze_patient"):
            # Get patient-specific feature contributions
            patient_contributions = get_patient_feature_contributions(
                state, selected_model, selected_patient
            )
            
            if patient_contributions is not None:
                # Create visualization
                st.write(f"Feature Contributions for Patient {selected_patient}")
                
                # Convert to DataFrame for visualization
                contribs_df = pd.DataFrame({
                    'Feature': list(patient_contributions.keys()),
                    'Contribution': list(patient_contributions.values())
                }).sort_values('Contribution')
                
                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot bars with color based on contribution direction
                colors = ['#ff7f0e' if x < 0 else '#1f77b4' for x in contribs_df['Contribution']]
                bars = ax.barh(contribs_df['Feature'], contribs_df['Contribution'], color=colors)
                
                # Add zero line
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Set labels and title
                ax.set_xlabel('Contribution to Risk Score')
                ax.set_title(f'Feature Contributions for Patient {selected_patient}')
                
                # Display chart
                st.pyplot(fig)
                
                # Add recommendations based on contributions
                st.subheader("Clinical Insights")
                
                # Identify top risk factors and protective factors
                risk_factors = contribs_df[contribs_df['Contribution'] > 0].sort_values('Contribution', ascending=False)
                protective_factors = contribs_df[contribs_df['Contribution'] < 0].sort_values('Contribution')
                
                # Display risk factors
                if not risk_factors.empty:
                    st.write("**Top Risk Factors:**")
                    for _, row in risk_factors.head(3).iterrows():
                        st.markdown(f"- **{row['Feature']}** (Contribution: {row['Contribution']:.3f})")
                
                # Display protective factors
                if not protective_factors.empty:
                    st.write("**Protective Factors:**")
                    for _, row in protective_factors.head(3).iterrows():
                        st.markdown(f"- **{row['Feature']}** (Contribution: {row['Contribution']:.3f})")
                
                # Add clinical recommendations
                st.write("**Potential Interventions:**")
                for _, row in risk_factors.head(2).iterrows():
                    feature = row['Feature']
                    # Generate recommendation based on feature name
                    recommendation = generate_recommendation(feature)
                    st.markdown(f"- {recommendation}")
            else:
                st.warning("Feature contribution data not available for the selected patient and model.")


def get_feature_importance(state: Dict[str, Any], 
                          model_name: str, 
                          method: str) -> Optional[Dict[str, List]]:
    """
    Get feature importance data for the selected model and method.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    model_name : str
        Name of the model
    method : str
        Importance method ("Model", "SHAP", or "Permutation")
        
    Returns
    -------
    Optional[Dict[str, List]]
        Dictionary with 'features' and 'importance' lists, or None if not available
    """
    # Check if we have a feature importance function
    if 'get_feature_importance' in state and callable(state['get_feature_importance']):
        # Use the provided function
        return state['get_feature_importance'](model_name, method)
    
    # Otherwise, generate sample data
    return generate_sample_feature_importance(state, model_name, method)


def generate_sample_feature_importance(state: Dict[str, Any], 
                                      model_name: str, 
                                      method: str) -> Dict[str, List]:
    """
    Generate sample feature importance data for demonstration purposes.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    model_name : str
        Name of the model
    method : str
        Importance method ("Model", "SHAP", or "Permutation")
        
    Returns
    -------
    Dict[str, List]
        Dictionary with 'features' and 'importance' lists
    """
    # Set seed based on model and method for consistent results
    np.random.seed(hash(model_name + method) % 2**32)
    
    # Define features
    features = [
        'age', 'gender', 'previous_hospitalization', 'medication_adherence', 
        'symptom_severity', 'substance_use', 'family_support', 
        'employment_status', 'duration_untreated', 'age_onset',
        'education_years', 'marital_status', 'living_situation', 
        'social_support', 'trauma_history', 'physical_health',
        'cognitive_functioning', 'insight', 'treatment_response',
        'hospitalization_count'
    ]
    
    # Generate importance values based on method
    if method == "Model":
        # More deterministic for model-based importance
        importance = np.array([
            0.15,  # age
            0.05,  # gender
            0.12,  # previous_hospitalization
            0.18,  # medication_adherence
            0.20,  # symptom_severity
            0.10,  # substance_use
            0.08,  # family_support
            0.06,  # employment_status
            0.09,  # duration_untreated
            0.07,  # age_onset
            0.04,  # education_years
            0.03,  # marital_status
            0.04,  # living_situation
            0.07,  # social_support
            0.06,  # trauma_history
            0.05,  # physical_health
            0.08,  # cognitive_functioning
            0.12,  # insight
            0.15,  # treatment_response
            0.11   # hospitalization_count
        ])
        
        # Add some model-specific variation
        if model_name == "Logistic Regression":
            importance *= np.random.uniform(0.8, 1.2, size=len(importance))
        elif model_name == "Gradient Boosting":
            importance *= np.random.uniform(0.7, 1.3, size=len(importance))
        elif model_name == "Neural Network":
            importance *= np.random.uniform(0.6, 1.4, size=len(importance))
        
    elif method == "SHAP":
        # SHAP values can be positive or negative
        importance = np.random.normal(0, 0.1, size=len(features))
        importance += np.array([
            0.12,  # age
            0.03,  # gender
            0.10,  # previous_hospitalization
            -0.15,  # medication_adherence (negative = protective)
            0.18,  # symptom_severity
            0.08,  # substance_use
            -0.10,  # family_support (negative = protective)
            -0.05,  # employment_status
            0.07,  # duration_untreated
            0.05,  # age_onset
            -0.04,  # education_years
            0.02,  # marital_status
            0.03,  # living_situation
            -0.08,  # social_support
            0.09,  # trauma_history
            0.04,  # physical_health
            -0.07,  # cognitive_functioning
            0.11,  # insight
            -0.12,  # treatment_response
            0.09   # hospitalization_count
        ])
        
    else:  # Permutation
        # Permutation importance is always positive
        importance = np.abs(np.random.normal(0, 0.05, size=len(features)))
        importance += np.array([
            0.10,  # age
            0.04,  # gender
            0.09,  # previous_hospitalization
            0.14,  # medication_adherence
            0.16,  # symptom_severity
            0.08,  # substance_use
            0.07,  # family_support
            0.05,  # employment_status
            0.08,  # duration_untreated
            0.06,  # age_onset
            0.03,  # education_years
            0.02,  # marital_status
            0.03,  # living_situation
            0.06,  # social_support
            0.05,  # trauma_history
            0.04,  # physical_health
            0.07,  # cognitive_functioning
            0.10,  # insight
            0.13,  # treatment_response
            0.09   # hospitalization_count
        ])
    
    # Normalize to sum to 1 for Model and Permutation methods
    if method != "SHAP":
        importance = importance / np.sum(importance)
    
    return {
        'features': features,
        'importance': importance.tolist()
    }


def get_feature_groups(state: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    """
    Get feature group definitions.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
        
    Returns
    -------
    Optional[Dict[str, List[str]]]
        Dictionary mapping group names to lists of feature names, or None if not available
    """
    # Check if we have feature groups in state
    if 'feature_groups' in state:
        return state['feature_groups']
    
    # Otherwise, generate sample groups
    return {
        'Demographics': [
            'age', 'gender', 'education_years', 'marital_status', 
            'employment_status', 'living_situation'
        ],
        'Clinical History': [
            'previous_hospitalization', 'hospitalization_count', 'age_onset',
            'duration_untreated', 'trauma_history', 'physical_health'
        ],
        'Symptoms & Functioning': [
            'symptom_severity', 'cognitive_functioning', 'insight',
            'treatment_response'
        ],
        'Social Factors': [
            'family_support', 'social_support', 'substance_use'
        ],
        'Treatment': [
            'medication_adherence'
        ]
    }


def get_group_importance(state: Dict[str, Any], 
                        feature_groups: Dict[str, List[str]]) -> Optional[Dict[str, List]]:
    """
    Get importance values for feature groups.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to lists of feature names
        
    Returns
    -------
    Optional[Dict[str, List]]
        Dictionary with 'groups', 'importance', and 'counts' lists, or None if not available
    """
    # Check if we have a group importance function
    if 'get_group_importance' in state and callable(state['get_group_importance']):
        # Use the provided function
        return state['get_group_importance'](feature_groups)
    
    # Otherwise, generate sample data
    return generate_sample_group_importance(feature_groups)


def generate_sample_group_importance(feature_groups: Dict[str, List[str]]) -> Dict[str, List]:
    """
    Generate sample group importance data for demonstration purposes.
    
    Parameters
    ----------
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to lists of feature names
        
    Returns
    -------
    Dict[str, List]
        Dictionary with 'groups', 'importance', and 'counts' lists
    """
    # Set seed for consistent results
    np.random.seed(42)
    
    # Calculate group importance
    groups = []
    importance = []
    counts = []
    
    for group, features in feature_groups.items():
        groups.append(group)
        counts.append(len(features))
        
        # Generate importance value based on group
        if group == "Clinical History":
            base_importance = 0.30
        elif group == "Symptoms & Functioning":
            base_importance = 0.25
        elif group == "Treatment":
            base_importance = 0.20
        elif group == "Social Factors":
            base_importance = 0.15
        else:  # Demographics
            base_importance = 0.10
        
        # Add some random variation
        group_importance = base_importance + np.random.uniform(-0.05, 0.05)
        importance.append(group_importance)
    
    # Normalize importance to sum to 1
    importance = np.array(importance) / np.sum(importance)
    
    return {
        'groups': groups,
        'importance': importance.tolist(),
        'counts': counts
    }


def get_feature_contributions(state: Dict[str, Any], 
                            model_name: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Get feature contributions and risk scores.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    model_name : str
        Name of the model
        
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]
        Tuple of (feature_contributions_df, risk_scores), or (None, None) if not available
    """
    # Check if we have a feature contributions function
    if 'get_feature_contributions' in state and callable(state['get_feature_contributions']):
        # Use the provided function
        return state['get_feature_contributions'](model_name)
    
    # Otherwise, generate sample data
    return generate_sample_feature_contributions(state, model_name)


def generate_sample_feature_contributions(state: Dict[str, Any], 
                                       model_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate sample feature contributions data for demonstration purposes.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    model_name : str
        Name of the model
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Tuple of (feature_contributions_df, risk_scores)
    """
    # Set seed based on model for consistent results
    np.random.seed(hash(model_name) % 2**32)
    
    # Generate risk scores for patients
    n_patients = 30
    risk_scores = np.random.beta(2, 5, size=n_patients)  # Beta distribution for realistic scores
    
    # Sort patients by risk score for better visualization
    sorted_indices = np.argsort(risk_scores)[::-1]  # Descending order
    risk_scores = risk_scores[sorted_indices]
    
    # Define features
    features = [
        'age', 'gender', 'previous_hospitalization', 'medication_adherence', 
        'symptom_severity', 'substance_use', 'family_support', 
        'employment_status', 'duration_untreated'
    ]
    
    # Generate contributions for each patient and feature
    contributions_data = {}
    
    for feature in features:
        # Generate base contribution based on feature
        if feature == 'symptom_severity':
            base_contrib = 0.2
        elif feature == 'medication_adherence':
            base_contrib = -0.15  # Protective factor
        elif feature == 'previous_hospitalization':
            base_contrib = 0.12
        elif feature == 'substance_use':
            base_contrib = 0.1
        elif feature == 'family_support':
            base_contrib = -0.1  # Protective factor
        else:
            base_contrib = 0.05
        
        # Add patient-specific variation
        patient_contribs = base_contrib + np.random.normal(0, 0.05, size=n_patients)
        
        # Apply sorting to match risk scores
        patient_contribs = patient_contribs[sorted_indices]
        
        # Store in dictionary
        contributions_data[feature] = patient_contribs
    
    # Create DataFrame
    contributions_df = pd.DataFrame(contributions_data)
    
    # Ensure contributions approximately sum to risk scores
    row_sums = contributions_df.sum(axis=1).values
    scaling_factors = risk_scores / row_sums
    
    for feature in features:
        contributions_df[feature] *= scaling_factors
    
    return contributions_df, risk_scores


def get_patient_feature_contributions(state: Dict[str, Any], 
                                    model_name: str,
                                    patient_id: str) -> Optional[Dict[str, float]]:
    """
    Get feature contributions for a specific patient.
    
    Parameters
    ----------
    state : Dict[str, Any]
        Application state dictionary
    model_name : str
        Name of the model
    patient_id : str
        Patient identifier
        
    Returns
    -------
    Optional[Dict[str, float]]
        Dictionary mapping feature names to contribution values, or None if not available
    """
    # Check if we have a patient contributions function
    if 'get_patient_contributions' in state and callable(state['get_patient_contributions']):
        # Use the provided function
        return state['get_patient_contributions'](model_name, patient_id)
    
    # Otherwise, generate sample data
    return generate_sample_patient_contributions(patient_id)


def generate_sample_patient_contributions(patient_id: str) -> Dict[str, float]:
    """
    Generate sample feature contributions for a specific patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to contribution values
    """
    # Set seed based on patient ID for consistent results
    np.random.seed(hash(patient_id) % 2**32)
    
    # Define features
    features = [
        'age', 'gender', 'previous_hospitalization', 'medication_adherence', 
        'symptom_severity', 'substance_use', 'family_support', 
        'employment_status', 'duration_untreated'
    ]
    
    # Generate contributions
    contributions = {}
    
    for feature in features:
        # Generate contribution based on feature
        if feature == 'symptom_severity':
            contrib = 0.2 + np.random.normal(0, 0.05)
        elif feature == 'medication_adherence':
            contrib = -0.15 + np.random.normal(0, 0.05)  # Protective factor
        elif feature == 'previous_hospitalization':
            contrib = 0.12 + np.random.normal(0, 0.03)
        elif feature == 'substance_use':
            contrib = 0.1 + np.random.normal(0, 0.04)
        elif feature == 'family_support':
            contrib = -0.1 + np.random.normal(0, 0.03)  # Protective factor
        else:
            contrib = np.random.normal(0, 0.08)
        
        contributions[feature] = contrib
    
    return contributions


def generate_recommendation(feature: str) -> str:
    """
    Generate a clinical recommendation based on feature name.
    
    Parameters
    ----------
    feature : str
        Feature name
        
    Returns
    -------
    str
        Clinical recommendation
    """
    recommendations = {
        'symptom_severity': "Consider more intensive symptom monitoring and targeted medication adjustment.",
        'medication_adherence': "Implement medication adherence strategies such as reminders or family involvement.",
        'previous_hospitalization': "Develop a more detailed relapse prevention plan with early warning signs.",
        'substance_use': "Refer to substance use treatment or dual diagnosis program.",
        'family_support': "Engage family in psychoeducation and support groups to enhance support network.",
        'employment_status': "Connect with vocational rehabilitation services for employment support.",
        'duration_untreated': "Emphasize importance of continuous treatment and engagement with services.",
        'age': "Consider age-appropriate interventions and support groups.",
        'gender': "Explore gender-specific support services if available.",
        'social_support': "Help build broader social support network through community resources.",
        'cognitive_functioning': "Consider cognitive remediation therapy to address cognitive deficits.",
        'insight': "Focus on psychoeducation to improve illness awareness and treatment adherence."
    }
    
    return recommendations.get(feature, f"Focus on addressing {feature.replace('_', ' ')} through appropriate interventions.")

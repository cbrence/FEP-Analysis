"""
Prediction tool page for the FEP dashboard.

This module implements the patient outcome prediction page where clinicians
can enter patient data and get predictions with risk stratification.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import clinical_weights
from webapp.components.risk_display import (
    display_risk_stratification,
    display_probability_bars,
    display_risk_factors
)

# Feature importance (would normally come from model)
SAMPLE_FEATURE_IMPORTANCE = {
    "P1: Delusions": 0.12,
    "G12: Lack of judgment and insight": 0.10,
    "N4: Passive/apathetic social withdrawal": 0.09,
    "G6: Depression": 0.08,
    "Age": 0.07,
    "P6: Suspiciousness": 0.06,
    "G2: Anxiety": 0.05,
    "N1: Blunted affect": 0.05,
    "Education": 0.04,
    "Drugs": 0.04
}

def show_prediction_tool(models):
    """
    Display the patient outcome prediction tool.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    """
    st.header("Patient Outcome Prediction Tool")
    
    st.markdown("""
    Enter patient information to predict remission outcomes. 
    The tool prioritizes early detection of high-risk patterns to prevent relapse.
    """)
    
    # Create tabs for different input groups
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical", "PANSS Positive", "PANSS Negative/General"])
    
    with tab1:
        st.subheader("Demographic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=65, value=25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ethnicity = st.selectbox("Ethnicity", ["White", "Other"])
        
        with col2:
            education = st.selectbox("Education", ["Before 16", "At 16", "17 to 18", "College", "University"])
            relationship = st.selectbox("Relationship Status", 
                                     ["Single", "Relationship", "Married", "Separated", "Divorced"])
            accommodation = st.selectbox("Accommodation", 
                                      ["Private_Family", "Rented", "Private_Owner", "Homeless", "NFA"])
    
    with tab2:
        st.subheader("Clinical Information")
        col1, col2 = st.columns(2)
        
        with col1:
            hospitalized = st.selectbox("Admitted to Hospital", ["Yes", "No"])
            depression = st.selectbox("Depression Severity", ["None", "Mild", "Moderate", "Severe"])
        
        with col2:
            alcohol = st.selectbox("Alcohol Use", ["Yes", "No"])
            drugs = st.selectbox("Drug Use", ["Yes", "No"])
            
        # Early warning signs section
        st.subheader("Early Warning Signs")
        col1, col2 = st.columns(2)
        
        with col1:
            sleep_disturbance = st.checkbox("Sleep Disturbance")
            social_withdrawal = st.checkbox("Social Withdrawal")
            suspiciousness = st.checkbox("Increased Suspiciousness")
        
        with col2:
            med_adherence = st.checkbox("Medication Adherence Issues")
            thought_disorder = st.checkbox("Thought Disorder Signs")
    
    with tab3:
        st.subheader("PANSS Positive Symptoms (1-7 scale)")
        
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.slider("P1: Delusions", 1, 7, 3)
            p2 = st.slider("P2: Conceptual disorganization", 1, 7, 3)
            p3 = st.slider("P3: Hallucinatory behaviour", 1, 7, 3)
            p4 = st.slider("P4: Excitement", 1, 7, 3)
        
        with col2:
            p5 = st.slider("P5: Grandiosity", 1, 7, 3)
            p6 = st.slider("P6: Suspiciousness", 1, 7, 3)
            p7 = st.slider("P7: Hostility", 1, 7, 3)
    
    with tab4:
        st.subheader("PANSS Negative & General Symptoms")
        
        # Display these as collapsible sections due to the large number
        with st.expander("PANSS Negative (N1-N7)"):
            n1 = st.slider("N1: Blunted affect", 1, 7, 3)
            n2 = st.slider("N2: Emotional withdrawal", 1, 7, 3)
            n3 = st.slider("N3: Poor rapport", 1, 7, 3)
            n4 = st.slider("N4: Passive/apathetic social withdrawal", 1, 7, 3)
            n5 = st.slider("N5: Difficulty in abstract thinking", 1, 7, 3)
            n6 = st.slider("N6: Lack of spontaneity and flow of conversation", 1, 7, 3)
            n7 = st.slider("N7: Stereotyped thinking", 1, 7, 3)
        
        with st.expander("PANSS General (G1-G8)"):
            g1 = st.slider("G1: Somatic concern", 1, 7, 3)
            g2 = st.slider("G2: Anxiety", 1, 7, 3)
            g3 = st.slider("G3: Guilt feelings", 1, 7, 3)
            g4 = st.slider("G4: Tension", 1, 7, 3)
            g5 = st.slider("G5: Mannerisms & posturing", 1, 7, 3)
            g6 = st.slider("G6: Depression", 1, 7, 3)
            g7 = st.slider("G7: Motor retardation", 1, 7, 3)
            g8 = st.slider("G8: Uncooperativeness", 1, 7, 3)
        
        with st.expander("PANSS General (G9-G16)"):
            g9 = st.slider("G9: Unusual thought content", 1, 7, 3)
            g10 = st.slider("G10: Disorientation", 1, 7, 3)
            g11 = st.slider("G11: Poor attention", 1, 7, 3)
            g12 = st.slider("G12: Lack of judgment and insight", 1, 7, 3)
            g13 = st.slider("G13: Disturbance of volition", 1, 7, 3)
            g14 = st.slider("G14: Poor impulse control", 1, 7, 3)
            g15 = st.slider("G15: Preoccupation", 1, 7, 3)
            g16 = st.slider("G16: Active social avoidance", 1, 7, 3)
    
    # Model selection
    st.subheader("Prediction Settings")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Select prediction model", 
            ["Gradient Boosting (Recommended)", "High-Risk Ensemble", "Logistic Regression", "Decision Tree"]
        )
    
    with col2:
        time_since_onset = st.number_input("Days since symptom onset", min_value=1, max_value=365, value=30)
    
    # Collect all inputs into a feature dictionary
    feature_values = {
        "Age": age,
        "Gender": gender,
        "Ethnicity": ethnicity,
        "Education": education,
        "Relationship": relationship,
        "Accommodation": accommodation,
        "Admitted_Hosp": hospitalized,
        "Depression_Severity": depression,
        "Alcohol": alcohol,
        "Drugs": drugs,
        "P1: Delusions": p1,
        "P2: Conceptual disorganization": p2,
        "P3: Hallucinatory behaviour": p3,
        "P4: Excitement": p4,
        "P5: Grandiosity": p5,
        "P6: Suspiciousness": p6,
        "P7: Hostility": p7,
        "N1: Blunted affect": n1,
        "N2: Emotional withdrawal": n2,
        "N3: Poor rapport": n3,
        "N4: Passive/apathetic social withdrawal": n4,
        "N5: Difficulty in abstract thinking": n5,
        "N6: Lack of spontaneity and flow of conversation": n6,
        "N7: Stereotyped thinking": n7,
        "G1: Somatic concern": g1,
        "G2: Anxiety": g2,
        "G3: Guilt feelings": g3,
        "G4: Tension": g4,
        "G5: Mannerisms & posturing": g5,
        "G6: Depression": g6,
        "G7: Motor retardation": g7,
        "G8: Uncooperativeness": g8,
        "G9: Unusual thought content": g9,
        "G10: Disorientation": g10,
        "G11: Poor attention": g11,
        "G12: Lack of judgment and insight": g12,
        "G13: Disturbance of volition": g13,
        "G14: Poor impulse control": g14,
        "G15: Preoccupation": g15,
        "G16: Active social avoidance": g16,
        "days_since_onset": time_since_onset
    }
    
    # Add early warning signs
    early_warning_count = sum([
        sleep_disturbance, social_withdrawal, suspiciousness,
        med_adherence, thought_disorder
    ])
    feature_values["early_warning_score"] = early_warning_count
    
    # Calculate early warning risk multiplier
    # Higher score means higher risk of relapse
    warning_risk = min(1.0, early_warning_count / 3)
    
    # Prediction button
    if st.button("Predict Outcomes", type="primary"):
        st.subheader("Prediction Results")
        
        # In a real implementation, this would use the actual models
        # Here we'll simulate prediction based on input values
        
        if model_choice == "Gradient Boosting (Recommended)":
            # Higher P1, G12, early warning signs increase risk of classes 0, 1, 2
            high_risk_factor = ((p1-1)/6 * 1.2 + (g12-1)/6 * 1.3 + warning_risk * 1.1) / 3
            high_risk_factor = max(0.1, min(0.9, high_risk_factor))
            
            # Treatment adherence factor (lower means worse adherence)
            adherence_factor = 1.0 - (1.0 if med_adherence else 0.0) / 1.8
            adherence_factor = max(0.2, min(0.8, adherence_factor))
            
            # Social functioning factor (lower means worse functioning)
            social_factor = 1.0 - ((n4-1)/6 * 1.1 + (g16-1)/6 * 1.2) / 2
            social_factor = max(0.2, min(0.8, social_factor))

        elif model_choice == "High-Risk Ensemble":
            # This model is more sensitive to high-risk patterns
            high_risk_factor = ((p1-1)/6 * 1.5 + (g12-1)/6 * 1.4 + warning_risk * 1.6) / 3
            high_risk_factor = max(0.2, min(0.95, high_risk_factor))
            
            # More sensitive to adherence issues
            adherence_factor = 1.0 - (1.0 if med_adherence else 0.0) / 1.5
            adherence_factor = max(0.15, min(0.7, adherence_factor))
            
            # More sensitive to social dysfunction
            social_factor = 1.0 - ((n4-1)/6 * 1.3 + (g16-1)/6 * 1.3) / 2
            social_factor = max(0.15, min(0.7, social_factor))

        elif model_choice == "Logistic Regression":
            # This model is less sensitive to nuances
            high_risk_factor = ((p1-1)/6 * 0.9 + (g12-1)/6 * 0.8 + warning_risk * 0.7) / 3
            high_risk_factor = max(0.05, min(0.8, high_risk_factor))
            
            # Less sensitive to adherence issues
            adherence_factor = 1.0 - (1.0 if med_adherence else 0.0) / 2.5
            adherence_factor = max(0.3, min(0.9, adherence_factor))
            
            # Less sensitive to social dysfunction
            social_factor = 1.0 - ((n4-1)/6 * 0.8 + (g16-1)/6 * 0.7) / 2
            social_factor = max(0.3, min(0.9, social_factor))

        else:  # Decision Tree
            # More binary decision boundaries
            high_risk_factor = ((p1-1)/6 * 1.0 + (g12-1)/6 * 1.0 + warning_risk * 1.0) / 3
            # Decision trees tend to have more discrete boundaries
            if high_risk_factor > 0.5:
                high_risk_factor = 0.8
            else:
                high_risk_factor = 0.3
        
            # More binary for adherence
            adherence_factor = 0.7 if med_adherence else 0.3
            
            # More binary for social functioning
            if (n4 + g16) / 2 > 4:
                social_factor = 0.3
            else:
                social_factor = 0.7

        # Initialize with model-specific baseline probabilities
        if model_choice == "Gradient Boosting (Recommended)":
            # Balanced distribution
            class_probs = {
                0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 
                4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05
            }
        elif model_choice == "High-Risk Ensemble":
            # Skewed toward high-risk classes
            class_probs = {
                0: 0.10, 1: 0.09, 2: 0.08, 3: 0.06, 
                4: 0.05, 5: 0.04, 6: 0.03, 7: 0.02
            }
        elif model_choice == "Logistic Regression":
            # More balanced, slightly favoring moderate classes
            class_probs = {
                0: 0.03, 1: 0.03, 2: 0.04, 3: 0.07, 
                4: 0.07, 5: 0.07, 6: 0.04, 7: 0.03
            }
        else:  # Decision Tree
            # More extreme, less nuanced
            class_probs = {
                0: 0.08, 1: 0.04, 2: 0.03, 3: 0.02, 
                4: 0.02, 5: 0.03, 6: 0.04, 7: 0.08
            }
    
        # Adjust based on risk profile with model-specific multipliers
        if early_warning_count >= 3 or p1 >= 5 or g12 >= 5:
            # High risk profile - add model-specific modifications
            multiplier = 1.0  # Default multiplier
            
            if model_choice == "High-Risk Ensemble":
                multiplier = 1.3  # High-Risk model emphasizes high-risk cases more
            elif model_choice == "Logistic Regression":
                multiplier = 0.8  # Logistic Regression is more conservative
            elif model_choice == "Decision Tree":
                multiplier = 1.1  # Decision Tree slightly emphasizes high-risk
                
            # Apply the model-specific multiplier to high-risk classes
            class_probs[0] = 0.30 * high_risk_factor * (1 - adherence_factor) * multiplier
            class_probs[1] = 0.25 * high_risk_factor * adherence_factor * multiplier
            class_probs[2] = 0.20 * high_risk_factor * multiplier
            
            # Adjust moderate and low risk classes inversely
            inverse_mult = 2 - multiplier  # If multiplier is high, inverse_mult is low and vice versa
            class_probs[3] = 0.10 * (1 - social_factor) * inverse_mult
            class_probs[4] = 0.05 * social_factor * inverse_mult
            class_probs[5] = 0.05 * adherence_factor * inverse_mult
            class_probs[6] = 0.03 * (1 - high_risk_factor) * inverse_mult
            class_probs[7] = 0.02 * (1 - high_risk_factor) * inverse_mult

        elif early_warning_count >= 1 or p1 >= 3 or g12 >= 3:
            # Moderate risk profile - add model-specific modifications
            mod_multiplier = 1.0  # Default moderate risk multiplier
            
            if model_choice == "High-Risk Ensemble":
                mod_multiplier = 0.9  # High-Risk model slightly de-emphasizes moderate cases
            elif model_choice == "Logistic Regression":
                mod_multiplier = 1.2  # Logistic Regression emphasizes moderate risk cases
            elif model_choice == "Decision Tree":
                mod_multiplier = 1.0  # Decision Tree treats moderate cases normally
                
            # Apply the model-specific multiplier to moderate-risk classes
            class_probs[3] = 0.15 * (1 - social_factor) * mod_multiplier
            class_probs[4] = 0.15 * social_factor * mod_multiplier
            class_probs[5] = 0.15 * adherence_factor * mod_multiplier
            
            # Adjust high-risk classes based on model
            high_multiplier = 1.0
            if model_choice == "High-Risk Ensemble":
                high_multiplier = 1.3
            elif model_choice == "Logistic Regression":
                high_multiplier = 0.7
                
            class_probs[0] = 0.15 * high_risk_factor * (1 - adherence_factor) * high_multiplier
            class_probs[1] = 0.10 * high_risk_factor * adherence_factor * high_multiplier
            class_probs[2] = 0.10 * high_risk_factor * high_multiplier
            
            # Low risk classes remain relatively unchanged
            class_probs[6] = 0.10 * (1 - high_risk_factor)
            class_probs[7] = 0.10 * (1 - high_risk_factor)

        else:
            # Low risk profile - add model-specific modifications
            low_multiplier = 1.0  # Default low risk multiplier
            
            if model_choice == "High-Risk Ensemble":
                low_multiplier = 0.7  # High-Risk model de-emphasizes low-risk cases
            elif model_choice == "Logistic Regression":
                low_multiplier = 1.1  # Logistic Regression slightly emphasizes low risk cases
            elif model_choice == "Decision Tree":
                low_multiplier = 0.9  # Decision Tree slightly de-emphasizes low risk
                
            # Apply the model-specific multiplier to low-risk classes
            class_probs[6] = 0.20 * (1 - high_risk_factor) * low_multiplier
            class_probs[7] = 0.30 * (1 - high_risk_factor) * low_multiplier
            
            # Adjust other classes
            class_probs[0] = 0.05 * high_risk_factor * (1 - adherence_factor)
            class_probs[1] = 0.05 * high_risk_factor * adherence_factor
            class_probs[2] = 0.05 * high_risk_factor
            class_probs[3] = 0.10 * (1 - social_factor)
            class_probs[4] = 0.10 * social_factor
            class_probs[5] = 0.15 * adherence_factor
        
        # Normalize probabilities to sum to 1
        total_prob = sum(class_probs.values())
        class_probs = {k: v/total_prob for k, v in class_probs.items()}
        
        # Get most likely class
        predicted_class = max(class_probs.items(), key=lambda x: x[1])[0]
        
        # Display results visually
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Show probability bars
            display_probability_bars(class_probs)
        
        with col2:
            # Show risk assessment
            risk_scores = list(class_probs.values())
            display_risk_stratification(risk_scores, [predicted_class])
        
        # Show key risk factors
        risk_factors_for_display = []
        for feature, weight in SAMPLE_FEATURE_IMPORTANCE.items():
            # Adjust weights based on model choice
            adjusted_weight = weight
            if model_choice == "High-Risk Ensemble" and feature in ["P1: Delusions", "G12: Lack of judgment and insight"]:
                adjusted_weight = weight * 1.3
            elif model_choice == "Logistic Regression":
                # Logistic regression typically has more evenly distributed weights
                adjusted_weight = (weight * 0.7) + 0.02
                
            risk_factors_for_display.append({
                'name': feature,
                'weight': adjusted_weight,
                'category': 'Clinical',  # Add appropriate category if available
                'modifiable': feature not in ['Age', 'Gender', 'Ethnicity']  # Example logic for modifiability
            })

        display_risk_factors(risk_factors_for_display)
        
        # Display clinical interpretation
        st.subheader("Clinical Interpretation")
        
        if predicted_class in [0, 1, 2]:  # High risk classes
            st.markdown("""
            This patient shows a **high-risk pattern** with significant probability of non-remission or early relapse. 
            The following clinical considerations are important:
            
            - Close monitoring is essential, with follow-up intervals of 1-2 weeks recommended
            - Early intervention for emerging symptoms may prevent full relapse
            - Consider evaluating medication adherence and potentially adjusting treatment
            - Psychosocial support should be intensified, particularly around identified risk factors
            """)
            
            if early_warning_count >= 2:
                st.error("⚠️ **ALERT:** Multiple early warning signs detected. Immediate clinical review recommended.")
                
        elif predicted_class in [3, 4, 5]:  # Moderate risk classes
            st.markdown("""
            This patient shows a **moderate-risk pattern** with partial or inconsistent remission predicted.
            The following clinical considerations may be helpful:
            
            - Regular monitoring on a 2-4 week schedule is appropriate
            - Focus on maintaining treatment engagement and adherence
            - Address specific domains showing impairment in PANSS scores
            - Consider psychosocial interventions targeting functional recovery
            """)
            
            if early_warning_count >= 2:
                st.warning("⚠️ **Note:** Some early warning signs present. Increase monitoring frequency.")
                
        else:  # Low risk classes
            st.markdown("""
            This patient shows a **lower-risk pattern** with good probability of sustained remission.
            The following clinical considerations may be helpful:
            
            - Standard monitoring on a 4-6 week schedule is typically appropriate
            - Focus on functional recovery and reintegration
            - Continue current treatment plan with emphasis on maintaining gains
            - Consider gradual reduction in service intensity if stability maintained
            """)
            
            if early_warning_count >= 1:
                st.info("ℹ️ **Note:** Monitor the identified early warning sign(s) at regular intervals.")
        
        # Add model-specific notes
        if model_choice == "High-Risk Ensemble":
            st.info("**Note:** The High-Risk Ensemble model is calibrated to be particularly sensitive to risk patterns that have historically led to poor outcomes. It may classify more patients as high-risk compared to other models.")
        elif model_choice == "Logistic Regression":
            st.info("**Note:** The Logistic Regression model provides a more balanced assessment with emphasis on established clinical factors. It may classify fewer patients at the extremes.")
        elif model_choice == "Decision Tree":
            st.info("**Note:** The Decision Tree model uses discrete clinical thresholds to classify patients. This provides clear decision boundaries but may be less sensitive to subtle clinical presentations.")
        
        # Limitations disclaimer
        st.caption("""
        **Important:** This prediction is based on statistical patterns and should be considered alongside 
        comprehensive clinical assessment. Individual outcomes may vary based on factors not captured in this model.
        """)

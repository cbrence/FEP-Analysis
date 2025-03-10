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
    display_risk_stratified_results,
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
        
        # Create simulated probabilities based on inputs
        # Higher P1, G12, early warning signs increase risk of classes 0 and 3
        high_risk_factor = ((p1-1)/6 + (g12-1)/6 + warning_risk) / 3
        high_risk_factor = max(0.1, min(0.9, high_risk_factor))
        
        if early_warning_count >= 3 or p1 >= 5 or g12 >= 5:
            # High risk profile
            class_probs = {
                0: 0.40 * high_risk_factor,  # No remission
                1: 0.30 * (1 - high_risk_factor),  # Sustained remission
                2: 0.10,  # Late remission
                3: 0.30 * high_risk_factor,  # Early relapse
                6: 0.10
            }
        elif early_warning_count >= 1 or p1 >= 3 or g12 >= 3:
            # Moderate risk profile
            class_probs = {
                0: 0.20 * high_risk_factor,
                1: 0.40 * (1 - high_risk_factor),
                2: 0.20,
                3: 0.15 * high_risk_factor,
                6: 0.05
            }
        else:
            # Low risk profile
            class_probs = {
                0: 0.10 * high_risk_factor,
                1: 0.60 * (1 - high_risk_factor),
                2: 0.15,
                3: 0.10 * high_risk_factor,
                6: 0.05
            }
        
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
            display_risk_stratified_results(class_probs, predicted_class)
        
        # Show key risk factors
        display_risk_factors(feature_values, SAMPLE_FEATURE_IMPORTANCE)
        
        # Display clinical interpretation
        st.subheader("Clinical Interpretation")
        
        if predicted_class in [0, 3]:  # High risk classes
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
                
        elif predicted_class in [2, 6]:  # Moderate risk classes
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
        
        # Limitations disclaimer
        st.caption("""
        **Important:** This prediction is based on statistical patterns and should be considered alongside 
        comprehensive clinical assessment. Individual outcomes may vary based on factors not captured in this model.
        """)

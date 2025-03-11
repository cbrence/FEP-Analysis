"""
Configuration editor page for the FEP dashboard.

This module implements a configuration interface that allows clinicians to 
customize the clinical weights and thresholds used in the prediction models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import clinical_weights

def show_config_editor():
    """Display the configuration editor interface."""
    st.header("Clinical Configuration Editor")
    
    st.markdown("""
    This page allows you to customize the clinical weights and thresholds used in the FEP prediction models.
    These settings determine how the models prioritize different types of errors and make risk assessments.
    
    Changes you make here will affect:
    - How the models are trained (class weights)
    - How predictions are made (thresholds)
    - How risk assessments are displayed
    - What clinical recommendations are shown
    """)
    
    # Get current configuration
    current_config = clinical_weights.get_config()
    
    # Create tabs for different config sections
    tabs = st.tabs([
        "Risk Categories", 
        "Class Weights", 
        "Prediction Thresholds", 
        "Error Costs",
        "Clinical Recommendations",
        "Save/Load Configuration"
    ])
    
    with tabs[0]:
        show_risk_categories(current_config)
    
    with tabs[1]:
        show_class_weights(current_config)
    
    with tabs[2]:
        show_prediction_thresholds(current_config)
    
    with tabs[3]:
        show_error_costs(current_config)
    
    with tabs[4]:
        show_clinical_recommendations(current_config)
    
    with tabs[5]:
        show_save_load_options(current_config)

def show_risk_categories(config):
    """Show and edit risk category assignments."""
    st.subheader("Risk Categories")
    
    st.markdown("""
    Assign each outcome class to a risk level. This affects how predictions are 
    prioritized and how different types of errors are weighted.
    """)
    
    # Create a mapping of classes to risk levels
    class_to_risk = {}
    for risk_level, classes in config["risk_levels"].items():
        for cls in classes:
            class_to_risk[cls] = risk_level
    
    # Create a table for editing
    class_info = []
    for cls, description in config["class_definitions"].items():
        class_info.append({
            "Class": int(cls),
            "Description": description,
            "Risk Level": class_to_risk.get(int(cls), "unknown")
        })
    
    df = pd.DataFrame(class_info)
    
    # Allow editing of risk levels
    st.markdown("#### Assign Risk Levels")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Class": st.column_config.NumberColumn("Class", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True),
            "Risk Level": st.column_config.SelectboxColumn(
                "Risk Level",
                options=["high_risk", "moderate_risk", "low_risk"],
                required=True
            )
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Update config if changed
    if not df.equals(edited_df):
        # Reconstruct risk_levels from edited dataframe
        new_risk_levels = {
            "high_risk": [],
            "moderate_risk": [],
            "low_risk": []
        }
        
        for _, row in edited_df.iterrows():
            risk_level = row["Risk Level"]
            if risk_level in new_risk_levels:
                new_risk_levels[risk_level].append(int(row["Class"]))
        
        # Update config
        updated_config = {"risk_levels": new_risk_levels}
        if st.button("Save Risk Categories"):
            clinical_weights.update_config(updated_config)
            clinical_weights.refresh_from_config()
            st.success("Risk categories updated successfully!")

def show_class_weights(config):
    """Show and edit class weights."""
    st.subheader("Class Weights")
    
    st.markdown("""
    Class weights determine how much emphasis the models place on correctly predicting 
    each outcome class during training. Higher weights mean the model will prioritize 
    correctly identifying that class, even at the cost of more errors in other classes.
    """)
    
    # Create a table for editing
    class_info = []
    for cls, description in config["class_definitions"].items():
        class_info.append({
            "Class": int(cls),
            "Description": description,
            "Risk Level": clinical_weights.CLASS_TO_RISK_LEVEL.get(int(cls), "unknown"),
            "Weight": config["class_weights"].get(cls, 1.0)
        })
    
    df = pd.DataFrame(class_info)
    
    # Allow editing of weights
    st.markdown("#### Adjust Class Weights")
    st.markdown("*Higher weights prioritize correct prediction of that class*")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Class": st.column_config.NumberColumn("Class", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "Risk Level": st.column_config.TextColumn("Risk Level", disabled=True),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                min_value=0.1,
                max_value=20.0,
                step=0.1,
                format="%.1f"
            )
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Update config if changed
    if not df.equals(edited_df):
        # Reconstruct class_weights from edited dataframe
        new_class_weights = {
            str(row["Class"]): row["Weight"] for _, row in edited_df.iterrows()
        }
        
        # Update config
        updated_config = {"class_weights": new_class_weights}
        if st.button("Save Class Weights"):
            clinical_weights.update_config(updated_config)
            clinical_weights.refresh_from_config()
            st.success("Class weights updated successfully!")

def show_prediction_thresholds(config):
    """Show and edit prediction thresholds."""
    st.subheader("Prediction Thresholds")
    
    st.markdown("""
    Prediction thresholds determine how confident the model must be before assigning 
    a specific class. Lower thresholds increase sensitivity (catching more cases) but 
    may increase false positives. Higher thresholds increase specificity but may miss cases.
    """)
    
    # Create a table for editing
    class_info = []
    for cls, description in config["class_definitions"].items():
        class_info.append({
            "Class": int(cls),
            "Description": description,
            "Risk Level": clinical_weights.CLASS_TO_RISK_LEVEL.get(int(cls), "unknown"),
            "Threshold": config["prediction_thresholds"].get(cls, 0.5)
        })
    
    df = pd.DataFrame(class_info)
    
    # Allow editing of thresholds
    st.markdown("#### Adjust Prediction Thresholds")
    st.markdown("*Lower thresholds catch more cases (higher sensitivity) but may increase false alarms*")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Class": st.column_config.NumberColumn("Class", disabled=True),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "Risk Level": st.column_config.TextColumn("Risk Level", disabled=True),
            "Threshold": st.column_config.NumberColumn(
                "Threshold",
                min_value=0.1,
                max_value=0.9,
                step=0.05,
                format="%.2f"
            )
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Update config if changed
    if not df.equals(edited_df):
        # Reconstruct thresholds from edited dataframe
        new_thresholds = {
            str(row["Class"]): row["Threshold"] for _, row in edited_df.iterrows()
        }
        
        # Update config
        updated_config = {"prediction_thresholds": new_thresholds}
        if st.button("Save Prediction Thresholds"):
            clinical_weights.update_config(updated_config)
            clinical_weights.refresh_from_config()
            st.success("Prediction thresholds updated successfully!")

def show_error_costs(config):
    """Show and edit error costs."""
    st.subheader("Error Costs")
    
    st.markdown("""
    Error costs define the relative harm of different types of errors:
    - **False Negatives**: Missing a case that needs intervention
    - **False Positives**: Unnecessary intervention
    
    These costs are used in evaluation metrics and threshold optimization.
    """)
    
    # Create a table for false negative costs
    fn_costs = config["error_costs"]["false_negative"]
    fn_data = [
        {"Risk Level": "High Risk", "Cost": fn_costs["high_risk"], "Description": "Missing cases that need urgent intervention"},
        {"Risk Level": "Moderate Risk", "Cost": fn_costs["moderate_risk"], "Description": "Missing cases that may benefit from intervention"},
        {"Risk Level": "Low Risk", "Cost": fn_costs["low_risk"], "Description": "Missing cases unlikely to need intervention"}
    ]
    
    st.markdown("#### False Negative Costs (Missing Cases)")
    fn_df = pd.DataFrame(fn_data)
    
    edited_fn_df = st.data_editor(
        fn_df,
        column_config={
            "Risk Level": st.column_config.TextColumn("Risk Level", disabled=True),
            "Cost": st.column_config.NumberColumn(
                "Cost",
                min_value=1.0,
                max_value=20.0,
                step=0.5,
                format="%.1f"
            ),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Create a table for false positive costs
    fp_costs = config["error_costs"]["false_positive"]
    fp_data = [
        {"Risk Level": "High Risk", "Cost": fp_costs["high_risk"], "Description": "Unnecessary intervention for high-risk pattern"},
        {"Risk Level": "Moderate Risk", "Cost": fp_costs["moderate_risk"], "Description": "Unnecessary intervention for moderate-risk pattern"},
        {"Risk Level": "Low Risk", "Cost": fp_costs["low_risk"], "Description": "Unnecessary intervention for low-risk pattern"}
    ]
    
    st.markdown("#### False Positive Costs (Unnecessary Interventions)")
    fp_df = pd.DataFrame(fp_data)
    
    edited_fp_df = st.data_editor(
        fp_df,
        column_config={
            "Risk Level": st.column_config.TextColumn("Risk Level", disabled=True),
            "Cost": st.column_config.NumberColumn(
                "Cost",
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                format="%.1f"
            ),
            "Description": st.column_config.TextColumn("Description", disabled=True, width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Update config if changed
    if not fn_df.equals(edited_fn_df) or not fp_df.equals(edited_fp_df):
        # Reconstruct error costs from edited dataframes
        new_error_costs = {
            "false_negative": {
                "high_risk": edited_fn_df.loc[0, "Cost"],
                "moderate_risk": edited_fn_df.loc[1, "Cost"],
                "low_risk": edited_fn_df.loc[2, "Cost"]
            },
            "false_positive": {
                "high_risk": edited_fp_df.loc[0, "Cost"],
                "moderate_risk": edited_fp_df.loc[1, "Cost"],
                "low_risk": edited_fp_df.loc[2, "Cost"]
            }
        }
        
        # Update config
        updated_config = {"error_costs": new_error_costs}
        if st.button("Save Error Costs"):
            clinical_weights.update_config(updated_config)
            clinical_weights.refresh_from_config()
            st.success("Error costs updated successfully!")

def show_clinical_recommendations(config):
    """Show and edit clinical recommendations."""
    st.subheader("Clinical Recommendations")
    
    st.markdown("""
    These recommendations are shown to users based on the risk level of the 
    predicted outcome. You can customize them based on your clinical guidelines.
    """)
    
    # Create editors for each risk level
    for risk_level in ["high_risk", "moderate_risk", "low_risk"]:
        st.markdown(f"#### {risk_level.replace('_', ' ').title()} Recommendations")
        
        # Get current recommendations
        current_recs = config["clinical_recommendations"].get(risk_level, [])
        
        # Create a text area for editing
        new_recs = st.text_area(
            f"Edit recommendations for {risk_level}",
            value="\n".join(current_recs),
            height=150
        )
        
        # Update if changed
        if "\n".join(current_recs) != new_recs:
            # Split by newlines and remove empty lines
            updated_recs = [line.strip() for line in new_recs.split("\n") if line.strip()]
            
            # Update config
            if st.button(f"Save {risk_level.replace('_', ' ').title()} Recommendations"):
                updated_config = {
                    "clinical_recommendations": {
                        risk_level: updated_recs
                    }
                }
                clinical_weights.update_config(updated_config)
                clinical_weights.refresh_from_config()
                st.success(f"{risk_level.replace('_', ' ').title()} recommendations updated successfully!")

def show_save_load_options(config):
    """Show options for saving and loading configurations."""
    st.subheader("Save and Load Configurations")
    
    st.markdown("""
    You can save your current configuration to a file or load a previously saved configuration.
    This allows different clinicians to use their preferred settings or create settings for
    different clinical scenarios.
    """)
    
    # Set up columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Save Current Configuration")
        
        # Input for configuration name
        config_name = st.text_input("Configuration Name", value="my_clinical_config")
        
        # Ensure valid filename
        if not config_name.endswith(('.yaml', '.json')):
            config_name += '.yaml'
        
        # Button to save configuration
        if st.button("Save Configuration"):
            # Determine save path
            config_dir = clinical_weights.DEFAULT_CONFIG_DIR
            save_path = config_dir / config_name
            
            # Ensure directory exists
            clinical_weights.ensure_config_dir()
            
            # Save configuration
            success = clinical_weights.save_config(clinical_weights.get_config(), save_path)
            
            if success:
                st.success(f"Configuration saved to {save_path}")
            else:
                st.error("Failed to save configuration")
    
    with col2:
        st.markdown("#### Load Configuration")
        
        # Get available configuration files
        config_dir = clinical_weights.DEFAULT_CONFIG_DIR
        clinical_weights.ensure_config_dir()
        
        config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.json"))
        config_file_names = [file.name for file in config_files]
        
        if config_file_names:
            # Select configuration file
            selected_config = st.selectbox(
                "Select Configuration",
                options=config_file_names,
                index=0 if "default_config.yaml" in config_file_names else 0
            )
            
            # Button to load configuration
            if st.button("Load Configuration"):
                # Determine load path
                load_path = config_dir / selected_config
                
                # Load configuration
                loaded_config = clinical_weights.load_config(load_path)
                clinical_weights.refresh_from_config()
                
                st.success(f"Configuration loaded from {load_path}")
                st.info("Refresh page to see updated values")
        else:
            st.info("No saved configurations found. Save one first.")
    
    # Reset to default button
    st.markdown("#### Reset to Default")
    if st.button("Reset to Default Configuration"):
        clinical_weights.reset_to_default()
        clinical_weights.refresh_from_config()
        st.success("Configuration reset to default values")
        st.info("Refresh page to see updated values")
    
    # Show current configuration as JSON
    st.markdown("#### Current Configuration JSON")
    st.json(clinical_weights.get_config())


if __name__ == "__main__":
    show_config_editor()

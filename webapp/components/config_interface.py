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

def get_risk_order():
    """
    Return a list of classes ordered by risk level (highest risk first).
    
    Returns:
    --------
    list : Classes ordered by risk level
    """
    from config import clinical_weights
    
    # Get classes by risk level
    high_risk = clinical_weights.HIGH_RISK_CLASSES
    moderate_risk = clinical_weights.MODERATE_RISK_CLASSES
    low_risk = clinical_weights.LOW_RISK_CLASSES
    
    # Order by risk level: high > moderate > low
    # Within each level, keep original numerical order
    ordered_classes = (
        sorted(high_risk) + 
        sorted(moderate_risk) + 
        sorted(low_risk)
    )
    
    # Ensure all 8 classes are included (in case any are missing from risk levels)
    all_classes = set(range(8))
    missing_classes = all_classes - set(ordered_classes)
    
    return ordered_classes + sorted(list(missing_classes))


def get_hardcoded_risk_level(cls_num):
    """
    Get the correct risk level based on hardcoded mappings that match the sidebar.
    
    Parameters:
    -----------
    cls_num : int
        The class number (0-7)
        
    Returns:
    --------
    str : One of "high_risk", "moderate_risk", or "low_risk"
    """
    # Hardcoded correct risk level assignments
    high_risk_classes = [0, 1, 2]
    moderate_risk_classes = [3, 4, 5]
    low_risk_classes = [6, 7]
    
    if cls_num in high_risk_classes:
        return "high_risk"
    elif cls_num in moderate_risk_classes:
        return "moderate_risk"
    elif cls_num in low_risk_classes:
        return "low_risk"
    else:
        return "unknown"

def get_class_info(config, include_risk_display=True):
    """
    Generate consistent class information for display throughout the interface.
    
    Parameters:
    -----------
    config : dict
        Current configuration dictionary
    include_risk_display : bool, default=True
        Whether to include formatted risk level display string
        
    Returns:
    --------
    list : List of dictionaries with class information
    """
    # Get classes in order from 0-7 for consistent display
    classes_to_display = list(range(8))
    
    class_info = []
    
    # Get class information in the determined order
    for cls in classes_to_display:
        # Get the class description from config, or use default if missing
        description = config["class_definitions"].get(str(cls), f"Class {cls}")
        
        # Get the correct risk level based on hardcoded mapping
        risk_level = get_hardcoded_risk_level(cls)
        
        # Format risk level for display if requested
        if include_risk_display:
            if risk_level == "high_risk":
                risk_display = "🔴 High Risk"
            elif risk_level == "moderate_risk":
                risk_display = "🟠 Moderate Risk"
            elif risk_level == "low_risk":
                risk_display = "🟢 Low Risk"
            else:
                risk_display = "❓ Unknown"
        else:
            risk_display = risk_level
            
        # Build class info dictionary
        info = {
            "Class": cls,
            "Description": description,
            "Risk Level": risk_level,
            "Risk Display": risk_display
        }
        
        class_info.append(info)
    
    # Sort the class info by risk level priority (high to low)
    # And then by class number within each risk level
    def sort_key(info):
        risk_priority = {"high_risk": 0, "moderate_risk": 1, "low_risk": 2, "unknown": 3}
        return (risk_priority.get(info["Risk Level"], 4), info["Class"])
    
    return sorted(class_info, key=sort_key)

def update_class_definitions(config):
    """
    Ensure the class definitions in the config match the ones in clinical_weights.py
    
    Parameters:
    -----------
    config : dict
        Current configuration dictionary
        
    Returns:
    --------
    dict : Updated configuration with correct class definitions
    """
    from config import clinical_weights
    
    # Get the default class definitions from clinical_weights.py
    default_definitions = clinical_weights.DEFAULT_CONFIG["class_definitions"]
    
    # Update the config with these definitions
    if config["class_definitions"] != default_definitions:
        config["class_definitions"] = default_definitions.copy()
    
    return config


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

    # Ensure class definitions are correct
    config = update_class_definitions(config)

    st.subheader("Risk Categories")
    
    st.markdown("""
    Assign each outcome class to a risk level. This affects how predictions are 
    prioritized and how different types of errors are weighted.
    """)
    
    # Display current risk level assignments from sidebar for reference
    st.info("""
    **Current Risk Level Assignments:**
    - 🔴 High Risk: Classes [0, 1, 2]
    - 🟠 Moderate Risk: Classes [3, 4, 5]
    - 🟢 Low Risk: Classes [6, 7]
    """)

    # Use the helper function to get consistent class info
    class_info = get_class_info(config, include_risk_display=False)
    
    # Create a DataFrame for editing
    df = pd.DataFrame([{
        "Class": info["Class"],
        "Description": info["Description"],
        "Risk Level": info["Risk Level"]
    } for info in class_info])
    
    # Allow editing of risk levels
    st.markdown("#### Assign Risk Levels")
    st.markdown("*Each class must be assigned to exactly one risk level*")
    
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
        
        # Validate that all classes are assigned
        all_assigned = set(range(8))
        assigned_classes = set()
        for classes in new_risk_levels.values():
            assigned_classes.update(classes)
        
        missing_classes = all_assigned - assigned_classes
        if missing_classes:
            st.error(f"Classes {missing_classes} are not assigned to any risk level!")
        else:
            # Update config
            updated_config = {"risk_levels": new_risk_levels}
            if st.button("Save Risk Categories"):
                clinical_weights.update_config(updated_config)
                clinical_weights.refresh_from_config()
                st.success("Risk categories updated successfully!")

def show_class_weights(config):
    """Show and edit class weights."""

    # Ensure class definitions are correct
    config = update_class_definitions(config)

    st.subheader("Class Weights")
    
    st.markdown("""
    Class weights determine how much emphasis the models place on correctly predicting 
    each outcome class during training. Higher weights mean the model will prioritize 
    correctly identifying that class, even at the cost of more errors in other classes.
    """)
    
    # Use the helper function to get consistent class info
    class_info = get_class_info(config)
    
    # Create DataFrame for editing with a consistent structure
    df = pd.DataFrame([{
        "Class": info["Class"],
        "Description": info["Description"],
        "Risk Level": info["Risk Display"],
        "Weight": config["class_weights"].get(str(info["Class"]), 1.0)
    } for info in class_info])
    
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
    # Ensure class definitions are correct
    config = update_class_definitions(config)


    st.subheader("Prediction Thresholds")
    
    st.markdown("""
    Prediction thresholds determine how confident the model must be before assigning 
    a specific class. Lower thresholds increase sensitivity (catching more cases) but 
    may increase false positives. Higher thresholds increase specificity but may miss cases.
    """)
    
    # Use the helper function to get consistent class info
    class_info = get_class_info(config)
    
    
    # Create DataFrame for editing with a consistent structure
    df = pd.DataFrame([{
        "Class": info["Class"],
        "Description": info["Description"],
        "Risk Level": info["Risk Display"],
        "Threshold": config["prediction_thresholds"].get(str(info["Class"]), 0.5)
    } for info in class_info])
    
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
        {"Risk Level": "🔴 High Risk", "Cost": fn_costs["high_risk"], "Description": "Missing cases that need urgent intervention"},
        {"Risk Level": "🟠Moderate Risk", "Cost": fn_costs["moderate_risk"], "Description": "Missing cases that may benefit from intervention"},
        {"Risk Level": "🟢Low Risk", "Cost": fn_costs["low_risk"], "Description": "Missing cases unlikely to need intervention"}
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
        {"Risk Level": "🔴High Risk", "Cost": fp_costs["high_risk"], "Description": "Unnecessary intervention for high-risk pattern"},
        {"Risk Level": "🟠Moderate Risk", "Cost": fp_costs["moderate_risk"], "Description": "Unnecessary intervention for moderate-risk pattern"},
        {"Risk Level": "🟢Low Risk", "Cost": fp_costs["low_risk"], "Description": "Unnecessary intervention for low-risk pattern"}
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
    for risk_level, icon in [
        ("high_risk", "🔴"), 
        ("moderate_risk", "🟠"), 
        ("low_risk", "🟢")]:
        
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

    # Ensure class definitions are correct when displaying current configuration
    config = update_class_definitions(config)

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
                
                 # Ensure class definitions match even after loading
                clinical_weights.update_config({
                    "class_definitions": clinical_weights.DEFAULT_CONFIG["class_definitions"]
                })
                
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

    # Ensure all 8 classes are displayed
    config_to_display = clinical_weights.get_config()
    
    # Use the corrected configuration
    st.json(config)

    # Ensure all 8 classes are in class_definitions
    #for cls in range(8):
    #    if str(cls) not in config_to_display["class_definitions"]:
    #        config_to_display["class_definitions"][str(cls)] = f"Class {cls}"
    
    # Ensure all 8 classes have class weights
    #for cls in range(8):
    #    if str(cls) not in config_to_display["class_weights"]:
    #        config_to_display["class_weights"][str(cls)] = 1.0
    
    # Ensure all 8 classes have prediction thresholds
    #for cls in range(8):
    #    if str(cls) not in config_to_display["prediction_thresholds"]:
    #        config_to_display["prediction_thresholds"][str(cls)] = 0.5
    
    #st.json(config_to_display)

if __name__ == "__main__":
    show_config_editor()

"""
Clinical utility calculations for FEP prediction models.

This module provides functions for calculating and visualizing clinical utility,
incorporating the asymmetric costs of different types of errors in FEP prediction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights
from evaluation.metrics import clinical_utility_score

def calculate_error_costs(y_true, y_pred):
    """
    Calculate the costs of different types of errors.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
        
    Returns:
    --------
    dict : Error costs by risk level and type
    """
    # Initialize costs dict
    costs = {
        'high_risk': {'false_neg': 0, 'false_pos': 0},
        'moderate_risk': {'false_neg': 0, 'false_pos': 0},
        'low_risk': {'false_neg': 0, 'false_pos': 0}
    }
    
    # Calculate costs for each class
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Create binary classification problem for this class
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        # Get costs
        fn_cost = clinical_weights.ERROR_COSTS['false_negative'][risk_level]
        fp_cost = clinical_weights.ERROR_COSTS['false_positive'][risk_level]
        
        # Add to totals
        costs[risk_level]['false_neg'] += fn * fn_cost
        costs[risk_level]['false_pos'] += fp * fp_cost
    
    return costs

def compare_model_utilities(models, X_test, y_test):
    """
    Compare clinical utility across multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare.
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
        
    Returns:
    --------
    DataFrame : Clinical utility metrics for each model
    """
    results = []
    
    for name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate clinical utility
        utility, details = clinical_utility_score(y_test, y_pred)
        
        # Calculate error costs
        costs = calculate_error_costs(y_test, y_pred)
        
        # Get false negative rates by risk level
        fn_rates = {}
        for risk_level in ['high_risk', 'moderate_risk', 'low_risk']:
            # Get classes for this risk level
            if risk_level == 'high_risk':
                classes = clinical_weights.HIGH_RISK_CLASSES
            elif risk_level == 'moderate_risk':
                classes = clinical_weights.MODERATE_RISK_CLASSES
            else:
                classes = clinical_weights.LOW_RISK_CLASSES
            
            # Calculate false negative rate
            fn_count = 0
            total = 0
            
            for cls in classes:
                # Create binary classification problem
                y_true_bin = (y_test == cls).astype(int)
                y_pred_bin = (y_pred == cls).astype(int)
                
                # Get confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
                
                # Add to totals
                fn_count += fn
                total += fn + tp
            
            # Calculate rate
            fn_rate = fn_count / total if total > 0 else 0
            fn_rates[risk_level] = fn_rate
        
        # Store results
        results.append({
            'model': name,
            'utility': utility,
            'high_risk_fn_rate': fn_rates['high_risk'],
            'moderate_risk_fn_rate': fn_rates['moderate_risk'],
            'low_risk_fn_rate': fn_rates['low_risk'],
            'high_risk_fn_cost': costs['high_risk']['false_neg'],
            'high_risk_fp_cost': costs['high_risk']['false_pos'],
            'moderate_risk_fn_cost': costs['moderate_risk']['false_neg'],
            'moderate_risk_fp_cost': costs['moderate_risk']['false_pos'],
            'low_risk_fn_cost': costs['low_risk']['false_neg'],
            'low_risk_fp_cost': costs['low_risk']['false_pos'],
            'total_cost': sum(c['false_neg'] + c['false_pos'] for c in costs.values())
        })
    
    return pd.DataFrame(results)

def plot_utility_comparison(utility_df):
    """
    Plot clinical utility comparison across models.
    
    Parameters:
    -----------
    utility_df : DataFrame
        Results from compare_model_utilities.
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot clinical utility
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = ax1.bar(utility_df['model'], utility_df['utility'], color=colors)
    
    # Customize plot
    ax2.set_xticks(x)
    ax2.set_xticklabels(utility_df['model'])
    ax2.set_ylabel('False Negative Rate (lower is better)')
    ax2.set_title('False Negative Rates by Risk Level')
    ax2.legend()
    ax2.set_ylim([0, 0.5])
    
    plt.tight_layout()
    return fig

def plot_cost_breakdown(utility_df):
    """
    Plot breakdown of costs by error type and risk level.
    
    Parameters:
    -----------
    utility_df : DataFrame
        Results from compare_model_utilities.
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for stacked bar chart
    models = utility_df['model']
    
    # Create stacked bars for each cost type
    bar_width = 0.8
    
    # Define colors for different cost types
    colors = {
        'high_risk_fn': '#ff6b6b',
        'high_risk_fp': '#ff9999',
        'moderate_risk_fn': '#feca57',
        'moderate_risk_fp': '#ffd787',
        'low_risk_fn': '#1dd1a1',
        'low_risk_fp': '#87ebd0'
    }
    
    # Create stacked bars
    bottom = np.zeros(len(models))
    
    for cost_type, color in colors.items():
        if cost_type in utility_df.columns:
            ax.bar(models, utility_df[cost_type + '_cost'], bar_width, 
                   bottom=bottom, label=cost_type.replace('_', ' ').title(), 
                   color=color)
            bottom += utility_df[cost_type + '_cost'].values
    
    # Customize plot
    ax.set_ylabel('Error Cost')
    ax.set_title('Breakdown of Error Costs by Type and Risk Level')
    ax.legend()
    
    return fig

def compare_thresholds(model, X_test, y_test, thresholds=None):
    """
    Compare model performance with different classification thresholds.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate.
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    thresholds : dict, default=None
        Dictionary mapping classes to thresholds. If None, uses default 0.5.
        
    Returns:
    --------
    dict : Performance metrics for each threshold set
    """
    # Get class probabilities
    y_probs = model.predict_proba(X_test)
    classes = model.classes_
    
    # Get default predictions (threshold = 0.5)
    default_pred = classes[np.argmax(y_probs, axis=1)]
    
    # Get custom threshold predictions if provided
    if thresholds is not None:
        # Initialize predictions array
        custom_pred = np.zeros_like(default_pred)
        assigned = np.zeros_like(default_pred, dtype=bool)
        
        # Apply thresholds in order of clinical priority
        priority_order = clinical_weights.HIGH_RISK_CLASSES + clinical_weights.MODERATE_RISK_CLASSES + clinical_weights.LOW_RISK_CLASSES
        
        for cls in priority_order:
            if cls in thresholds and cls in classes:
                # Get index for this class
                cls_idx = np.where(classes == cls)[0][0]
                
                # Apply threshold for unassigned samples
                mask = (y_probs[:, cls_idx] >= thresholds[cls]) & (~assigned)
                custom_pred[mask] = cls
                assigned[mask] = True
        
        # For any unassigned samples, use argmax
        unassigned = ~assigned
        if np.any(unassigned):
            custom_pred[unassigned] = classes[np.argmax(y_probs[unassigned], axis=1)]
    else:
        custom_pred = default_pred
    
    # Calculate metrics for both prediction sets
    default_utility, default_details = clinical_utility_score(y_test, default_pred)
    custom_utility, custom_details = clinical_utility_score(y_test, custom_pred)
    
    # Calculate error costs
    default_costs = calculate_error_costs(y_test, default_pred)
    custom_costs = calculate_error_costs(y_test, custom_pred)
    
    # Return comparison
    return {
        'default': {
            'predictions': default_pred,
            'utility': default_utility,
            'details': default_details,
            'costs': default_costs
        },
        'custom': {
            'predictions': custom_pred,
            'utility': custom_utility,
            'details': custom_details,
            'costs': custom_costs
        }
    }

def plot_threshold_comparison(threshold_comparison):
    """
    Plot comparison of default vs custom thresholds.
    
    Parameters:
    -----------
    threshold_comparison : dict
        Results from compare_thresholds.
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure.
    """
    # Extract data
    default_costs = threshold_comparison['default']['costs']
    custom_costs = threshold_comparison['custom']['costs']
    
    # Prepare data for grouped bar chart
    risk_levels = ['high_risk', 'moderate_risk', 'low_risk']
    default_fn_costs = [default_costs[rl]['false_neg'] for rl in risk_levels]
    default_fp_costs = [default_costs[rl]['false_pos'] for rl in risk_levels]
    custom_fn_costs = [custom_costs[rl]['false_neg'] for rl in risk_levels]
    custom_fp_costs = [custom_costs[rl]['false_pos'] for rl in risk_levels]
    
    # Calculate total costs
    default_total = sum(default_fn_costs) + sum(default_fp_costs)
    custom_total = sum(custom_fn_costs) + sum(custom_fp_costs)
    improvement = (default_total - custom_total) / default_total * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot grouped bar chart for costs by risk level
    x = np.arange(len(risk_levels))
    width = 0.35
    
    ax1.bar(x - width/2, default_fn_costs, width/2, label='Default - FN', color='#ff6b6b')
    ax1.bar(x, default_fp_costs, width/2, label='Default - FP', color='#ff9999')
    ax1.bar(x + width/2, custom_fn_costs, width/2, label='Custom - FN', color='#1dd1a1')
    ax1.bar(x + width, custom_fp_costs, width/2, label='Custom - FP', color='#87ebd0')
    
    # Customize plot
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([rl.replace('_', ' ').title() for rl in risk_levels])
    ax1.set_ylabel('Error Cost')
    ax1.set_title('Error Costs by Risk Level and Threshold')
    ax1.legend()
    
    # Plot utility comparison
    utilities = [threshold_comparison['default']['utility'], 
                threshold_comparison['custom']['utility']]
    
    ax2.bar(['Default Thresholds', 'Custom Thresholds'], utilities, color=['#ff9999', '#1dd1a1'])
    
    # Add improvement text
    for i, v in enumerate(utilities):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    ax2.text(0.5, max(utilities) + 0.05, f'Improvement: {improvement:.1f}%', 
             ha='center', fontweight='bold')
    
    # Customize plot
    ax2.set_ylabel('Clinical Utility Score')
    ax2.set_title('Clinical Utility Comparison')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    return fig
    ax1.set_ylim([0, 1.0])
    ax1.set_ylabel('Clinical Utility Score (higher is better)')
    ax1.set_title('Clinical Utility by Model')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot false negative rates by risk level
    x = np.arange(len(utility_df))
    width = 0.25
    
    ax2.bar(x - width, utility_df['high_risk_fn_rate'], width, 
            label='High Risk', color='#ff6b6b')
    ax2.bar(x, utility_df['moderate_risk_fn_rate'], width, 
            label='Moderate Risk', color='#feca57')
    ax2.bar(x + width, utility_df['low_risk_fn_rate'], width, 
            label='Low Risk', color='#1dd1a1')
    
    # Customize plot
    
"""
Threshold optimization for FEP prediction models.

This module provides functions to optimize prediction thresholds based on
clinical utility and asymmetric costs of different types of errors.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights

def find_optimal_threshold(y_true, y_prob, fn_cost=10.0, fp_cost=1.0, thresholds=None):
    """
    Find threshold that minimizes total clinical cost.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    fn_cost : float, default=10.0
        Cost of a false negative (missing a case that needs intervention).
    fp_cost : float, default=1.0
        Cost of a false positive (unnecessary intervention).
    thresholds : array-like, default=None
        Thresholds to test. If None, uses 100 values from 0.01 to 0.99.
        
    Returns:
    --------
    float : Optimal threshold that minimizes total cost.
    dict : Detailed results for each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    results = []
    
    for threshold in thresholds:
        # Make binary predictions using this threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        false_neg_cost = fn * fn_cost
        false_pos_cost = fp * fp_cost
        total_cost = false_neg_cost + false_pos_cost
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Store results
        results.append({
            'threshold': threshold,
            'total_cost': total_cost,
            'false_neg_cost': false_neg_cost,
            'false_pos_cost': false_pos_cost,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find threshold with minimum cost
    min_cost_idx = results_df['total_cost'].idxmin()
    optimal_threshold = results_df.loc[min_cost_idx, 'threshold']
    
    return optimal_threshold, results_df

def find_optimal_thresholds_multiclass(y_true, y_prob, classes=None):
    """
    Find optimal thresholds for each class in a multiclass problem.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.
    classes : array-like, default=None
        Class labels. If None, inferred from data.
        
    Returns:
    --------
    dict : Optimal thresholds for each class.
    dict : Detailed results for each class.
    """
    if classes is None:
        classes = np.unique(y_true)
    
    optimal_thresholds = {}
    results = {}
    
    # For each class, optimize threshold
    for i, cls in enumerate(classes):
        # Create binary classification problem for this class
        y_true_bin = (y_true == cls).astype(int)
        
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Get costs for this risk level
        fn_cost = clinical_weights.ERROR_COSTS['false_negative'][risk_level]
        fp_cost = clinical_weights.ERROR_COSTS['false_positive'][risk_level]
        
        # Find optimal threshold
        opt_threshold, class_results = find_optimal_threshold(
            y_true_bin, y_prob[:, i], fn_cost=fn_cost, fp_cost=fp_cost
        )
        
        optimal_thresholds[cls] = opt_threshold
        results[cls] = class_results
    
    return optimal_thresholds, results

def apply_thresholds(y_prob, thresholds, classes=None, priority_order=None):
    """
    Apply optimized thresholds to predict classes, prioritizing high-risk classes.
    
    Parameters:
    -----------
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted class probabilities.
    thresholds : dict
        Dictionary mapping classes to thresholds.
    classes : array-like, default=None
        Class labels. If None, assumed to match probability columns.
    priority_order : list, default=None
        Order in which to apply thresholds. If None, uses clinical priority.
        
    Returns:
    --------
    array-like : Predicted class labels.
    """
    if classes is None:
        classes = np.arange(y_prob.shape[1])
    
    if priority_order is None:
        # Default priority is by risk level: high, moderate, then low
        high_risk = [cls for cls in classes if clinical_weights.CLASS_TO_RISK_LEVEL.get(cls) == 'high_risk']
        moderate_risk = [cls for cls in classes if clinical_weights.CLASS_TO_RISK_LEVEL.get(cls) == 'moderate_risk']
        low_risk = [cls for cls in classes if clinical_weights.CLASS_TO_RISK_LEVEL.get(cls) == 'low_risk']
        
        priority_order = high_risk + moderate_risk + low_risk
    
    # Initialize predictions
    predictions = np.zeros(len(y_prob), dtype=int) - 1  # -1 indicates no prediction yet
    
    # Apply thresholds in priority order
    for cls in priority_order:
        if cls in thresholds:
            cls_idx = np.where(classes == cls)[0][0]
            threshold = thresholds[cls]
            
            # Identify samples that meet threshold and don't have a prediction yet
            mask = (y_prob[:, cls_idx] >= threshold) & (predictions == -1)
            predictions[mask] = cls
    
    # For any remaining unclassified samples, use argmax
    mask = predictions == -1
    if np.any(mask):
        predictions[mask] = classes[np.argmax(y_prob[mask], axis=1)]
    
    return predictions

def plot_threshold_analysis(results_df, optimal_threshold, class_name, fn_cost, fp_cost):
    """
    Plot threshold analysis results to visualize the impact of different thresholds.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from find_optimal_threshold.
    optimal_threshold : float
        The optimal threshold identified.
    class_name : str
        Name of the class being analyzed.
    fn_cost : float
        Cost of false negatives.
    fp_cost : float
        Cost of false positives.
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot costs
    ax1.plot(results_df['threshold'], results_df['total_cost'], 'k-', label='Total Cost')
    ax1.plot(results_df['threshold'], results_df['false_neg_cost'], 'r--', label='False Negative Cost')
    ax1.plot(results_df['threshold'], results_df['false_pos_cost'], 'b--', label='False Positive Cost')
    
    # Mark optimal threshold
    ax1.axvline(x=optimal_threshold, color='g', linestyle='-', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    ax1.set_ylabel('Cost')
    ax1.set_title(f'Threshold Analysis for Class {class_name}\nFN Cost = {fn_cost}, FP Cost = {fp_cost}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(results_df['threshold'], results_df['sensitivity'], 'r-', label='Sensitivity')
    ax2.plot(results_df['threshold'], results_df['specificity'], 'b-', label='Specificity')
    ax2.plot(results_df['threshold'], results_df['precision'], 'g-', label='Precision')
    ax2.plot(results_df['threshold'], results_df['npv'], 'y-', label='NPV')
    
    # Mark optimal threshold
    ax2.axvline(x=optimal_threshold, color='g', linestyle='-')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

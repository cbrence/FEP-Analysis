"""
Custom evaluation metrics for FEP prediction models.

This module provides specialized metrics that incorporate clinical priorities
for evaluating model performance, focusing on asymmetric costs of errors.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, fbeta_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights

# Setup logging
logger = logging.getLogger(__name__)

def clinical_utility_score(y_true, y_pred):
    """
    Calculate a clinical utility score that incorporates asymmetric costs of errors.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
        
    Returns:
    --------
    float : Clinical utility score (1 is perfect, 0 is worst)
    dict : Detailed metrics including costs and components
    """
    # Initialize costs
    total_cost = 0
    max_possible_cost = 0
    costs_by_class = {}
    
    # Calculate costs for each class
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        # Create binary classification problem for this class
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Get confusion matrix for this class
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Get costs for this risk level
        fn_cost = clinical_weights.ERROR_COSTS['false_negative'][risk_level]
        fp_cost = clinical_weights.ERROR_COSTS['false_positive'][risk_level]
        
        # Calculate class-specific cost
        class_cost = fn * fn_cost + fp * fp_cost
        total_cost += class_cost
        
        # Calculate max possible cost for normalization
        class_max_cost = len(y_true) * max(fn_cost, fp_cost)
        max_possible_cost += class_max_cost
        
        # Store class-specific metrics
        costs_by_class[cls] = {
            'false_negatives': fn,
            'false_positives': fp,
            'true_positives': tp,
            'true_negatives': tn,
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'class_cost': class_cost
        }
    
    # Calculate overall utility score (higher is better)
    utility_score = 1 - (total_cost / max_possible_cost)
    
    # Prepare detailed metrics
    details = {
        'total_cost': total_cost,
        'max_possible_cost': max_possible_cost,
        'utility_score': utility_score,
        'class_details': costs_by_class
    }
    
    return utility_score, details

def weighted_f_score(y_true, y_pred, beta=2):
    """
    Calculate F-beta score with higher beta to prioritize recall.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    beta : float, default=2
        Beta parameter for F-score. Beta > 1 gives more weight to recall.
        
    Returns:
    --------
    float : Weighted F-beta score
    """
    # Calculate F-beta score for each class
    class_scores = {}
    overall_score = 0
    total_samples = len(y_true)
    
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Adjust beta based on risk level
        if risk_level == 'high_risk':
            class_beta = beta * 1.5  # Increase beta for high-risk classes
        elif risk_level == 'low_risk':
            class_beta = beta * 0.75  # Decrease beta for low-risk classes
        else:
            class_beta = beta
        
        # Calculate class-specific F-score
        class_score = fbeta_score(
            (y_true == cls).astype(int),
            (y_pred == cls).astype(int),
            beta=class_beta
        )
        
        # Weight by class frequency
        class_weight = np.sum(y_true == cls) / total_samples
        overall_score += class_score * class_weight
        class_scores[cls] = class_score
    
    return overall_score, class_scores

def time_weighted_error(y_true, y_pred, time_points, time_discount=None):
    """
    Calculate error with higher penalties for early misses.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    time_points : array-like
        Time point for each sample (e.g., days since baseline).
    time_discount : float, default=None
        Factor to discount errors over time (0-1). If None, uses config value.
        
    Returns:
    --------
    float : Time-weighted error score (lower is better)
    """
    if time_discount is None:
        time_discount = clinical_weights.TIME_DISCOUNT_FACTOR
    
    # Calculate errors
    errors = (y_true != y_pred).astype(int)
    
    # Normalize time to 0-1 range
    max_time = np.max(time_points)
    norm_time = time_points / max_time
    
    # Calculate time-based weights (earlier mistakes cost more)
    time_weights = time_discount ** norm_time
    
    # Apply weights to errors
    weighted_errors = errors * time_weights
    
    return np.mean(weighted_errors)

def risk_adjusted_auc(y_true, y_proba, classes=None):
    """
    Calculate AUC with class-specific adjustments for clinical priorities.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    classes : array-like, default=None
        List of classes. If None, derived from data.
        
    Returns:
    --------
    float : Weighted average AUC
    dict : Class-specific AUCs
    """
    if classes is None:
        classes = np.unique(y_true)
    
    # Binarize y_true for multi-class
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # If only 2 classes, reshape y_true_bin
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    
    # Calculate AUC for each class
    class_auc = {}
    weighted_avg_auc = 0
    total_weight = 0
    
    for i, cls in enumerate(classes):
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Calculate weight based on risk level
        if risk_level == 'high_risk':
            weight = 3.0  # Higher weight for high-risk classes
        elif risk_level == 'low_risk':
            weight = 1.0  # Lower weight for low-risk classes
        else:
            weight = 2.0  # Moderate weight for moderate-risk classes
            
        # Calculate class-specific AUC
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            class_auc[cls] = auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"Error calculating AUC for class {cls}: {e}")
            class_auc[cls] = 0.5  # Default to random performance
        
        # Add to weighted average
        weighted_avg_auc += class_auc[cls] * weight
        total_weight += weight
    
    # Normalize weighted average
    weighted_avg_auc /= total_weight
    
    return weighted_avg_auc, class_auc

def precision_at_risk_level(y_true, y_proba, high_risk_level=0.3, classes=None):
    """
    Calculate precision at specific risk thresholds for each class.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    high_risk_level : float, default=0.3
        Threshold for high-risk classification.
    classes : array-like, default=None
        List of classes. If None, derived from data.
        
    Returns:
    --------
    dict : Precision at high-risk level for each class
    """
    if classes is None:
        classes = np.unique(y_true)
    
    # Binarize y_true for multi-class
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # If only 2 classes, reshape y_true_bin
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    
    # Calculate precision at high-risk level for each class
    results = {}
    
    for i, cls in enumerate(classes):
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        
        # Find closest threshold to high_risk_level
        if len(thresholds) > 0:
            idx = np.argmin(np.abs(thresholds - high_risk_level))
            results[cls] = {
                'precision': precision[idx],
                'recall': recall[idx],
                'threshold': thresholds[idx]
            }
        else:
            results[cls] = {
                'precision': 0,
                'recall': 0,
                'threshold': 0
            }
    
    return results

def high_risk_recall_score(y_true, y_pred):
    """
    Calculate recall specifically for high-risk classes.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
        
    Returns:
    --------
    float : High-risk recall score
    dict : Class-specific recall scores
    """
    high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
    
    # Calculate recall for high-risk classes
    high_risk_recall = {}
    avg_high_risk_recall = 0
    total_high_risk = 0
    
    for cls in high_risk_classes:
        # Check if class exists in the data
        if cls not in np.unique(np.concatenate([y_true, y_pred])):
            continue
            
        # Create binary classification problem for this class
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        high_risk_recall[cls] = recall
        
        # Add to weighted average
        total_high_risk += (tp + fn)
        avg_high_risk_recall += recall * (tp + fn)
    
    # Calculate weighted average recall
    if total_high_risk > 0:
        avg_high_risk_recall /= total_high_risk
    else:
        avg_high_risk_recall = np.nan
    
    return avg_high_risk_recall, high_risk_recall

def clinical_classification_report(y_true, y_pred, y_proba=None, classes=None):
    """
    Generate a classification report with clinical priority weighting.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like of shape (n_samples, n_classes), default=None
        Predicted probabilities. If provided, includes probability metrics.
    classes : array-like, default=None
        List of classes. If None, derived from data.
        
    Returns:
    --------
    dict : Classification report with clinical metrics
    """
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Initialize report
    report = {
        'overall': {},
        'class': {}
    }
    
    # Calculate overall metrics
    utility_score, utility_details = clinical_utility_score(y_true, y_pred)
    report['overall']['clinical_utility'] = utility_score
    
    f2_score, class_f2 = weighted_f_score(y_true, y_pred, beta=2)
    report['overall']['weighted_f2_score'] = f2_score
    
    hr_recall, class_hr_recall = high_risk_recall_score(y_true, y_pred)
    report['overall']['high_risk_recall'] = hr_recall
    
    if y_proba is not None:
        weighted_auc, class_auc = risk_adjusted_auc(y_true, y_proba, classes)
        report['overall']['risk_adjusted_auc'] = weighted_auc
    
    # Calculate class-specific metrics
    for cls in classes:
        # Create binary classification problem for this class
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get risk level for this class
        risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk')
        
        # Store class-specific metrics
        report['class'][cls] = {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': class_f2.get(cls, 0),
            'support': tp + fn,
            'risk_level': risk_level,
            'fn_cost': utility_details['class_details'][cls]['fn_cost'],
            'fp_cost': utility_details['class_details'][cls]['fp_cost']
        }
        
        # Add AUC if probabilities provided
        if y_proba is not None:
            report['class'][cls]['auc'] = class_auc.get(cls, 0)
    
    return report

def print_clinical_classification_report(report, include_costs=True):
    """
    Print clinical classification report in a readable format.
    
    Parameters:
    -----------
    report : dict
        Classification report from clinical_classification_report function.
    include_costs : bool, default=True
        Whether to include cost information in the report.
    """
    # Print header
    print("Clinical Classification Report")
    print("=" * 80)
    
    # Print overall metrics
    print("Overall Metrics:")
    for metric, value in report['overall'].items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Print class-specific metrics
    print("Class Metrics:")
    print(f"{'Class':<6} {'Risk Level':<14} {'Precision':<10} {'Recall':<10} {'F2-Score':<10} {'Support':<8}")
    print("-" * 80)
    
    for cls, metrics in report['class'].items():
        print(f"{cls:<6} {metrics['risk_level']:<14} {metrics['precision']:.4f}{' ':>5} {metrics['recall']:.4f}{' ':>5} {metrics['f2_score']:.4f}{' ':>5} {metrics['support']}")
    
    # Print cost information if requested
    if include_costs:
        print("\nCost Information:")
        print(f"{'Class':<6} {'Risk Level':<14} {'FN Cost':<10} {'FP Cost':<10}")
        print("-" * 60)
        
        for cls, metrics in report['class'].items():
            print(f"{cls:<6} {metrics['risk_level']:<14} {metrics['fn_cost']:<10.1f} {metrics['fp_cost']:<10.1f}")
    
    print("=" * 80)

def calibration_curve_data(y_true, y_prob, n_bins=10, normalize=False):
    """
    Generate calibration curve data for predicted probabilities.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for the calibration curve.
    normalize : bool, default=False
        Whether to normalize bin counts.
        
    Returns:
    --------
    tuple : (prob_pred, prob_true, counts)
        prob_pred: Mean predicted probability in each bin
        prob_true: Fraction of positives in each bin
        counts: Number of samples in each bin
    """
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    counts = bin_total[nonzero]
    
    if normalize:
        counts = counts / float(counts.sum())
    
    return prob_pred, prob_true, counts

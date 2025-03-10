"""
Cross-validation strategies for FEP prediction models.

This module provides specialized cross-validation approaches that account
for the class imbalance and small sample sizes typical in FEP studies.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, 
    RepeatedStratifiedKFold, LeaveOneOut,
    train_test_split
)
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights, settings
from evaluation.metrics import clinical_utility_score, risk_adjusted_auc

# Setup logging
logger = logging.getLogger(__name__)

def stratified_risk_split(X, y, test_size=0.3, random_state=None):
    """
    Split data while preserving risk level distribution.
    
    Parameters:
    -----------
    X : array-like
        Features.
    y : array-like
        Target values.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
        Split data.
    """
    # Get risk level for each sample
    risk_levels = np.array([clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk') 
                          for cls in y])
    
    # Split by risk level instead of raw class
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X, y, risk_levels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=risk_levels
    )
    
    return X_train, X_test, y_train, y_test

def bootstrap_evaluation(model, X, y, n_iterations=100, test_size=0.3, random_state=None):
    """
    Evaluate model using bootstrap resampling.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate.
    X : array-like
        Features.
    y : array-like
        Target values.
    n_iterations : int, default=100
        Number of bootstrap iterations.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    dict : Bootstrap evaluation results.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize results
    results = {
        'utility': [],
        'auc': [],
        'high_risk_recall': [],
        'class_metrics': {}
    }
    
    # Initialize class-specific metrics
    for cls in np.unique(y):
        results['class_metrics'][cls] = {
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    # Perform bootstrap iterations
    for i in range(n_iterations):
        # Create bootstrap sample (resample with replacement)
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_boot, y_boot, test_size=test_size, random_state=i
        )
        
        # Fit model on training data
        model.fit(X_train, y_train)
        
        # Get predictions on test data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        utility, _ = clinical_utility_score(y_test, y_pred)
        results['utility'].append(utility)
        
        # Calculate AUC if possible
        try:
            auc, _ = risk_adjusted_auc(y_test, y_proba)
            results['auc'].append(auc)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
        
        # Calculate high-risk recall
        high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
        high_risk_mask = np.isin(y_test, high_risk_classes)
        
        if np.any(high_risk_mask):
            high_risk_y_true = y_test[high_risk_mask]
            high_risk_y_pred = y_pred[high_risk_mask]
            
            # Calculate recall for high-risk samples
            correct = np.sum(high_risk_y_true == high_risk_y_pred)
            total = len(high_risk_y_true)
            high_risk_recall = correct / total if total > 0 else np.nan
            
            results['high_risk_recall'].append(high_risk_recall)
        
        # Calculate class-specific metrics
        for cls in np.unique(y_test):
            # Binary predictions for this class
            y_true_bin = (y_test == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            
            # Calculate metrics if class is present
            if np.sum(y_true_bin) > 0:
                # Calculate precision
                precision = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / np.sum(y_pred_bin == 1) if np.sum(y_pred_bin == 1) > 0 else 0
                results['class_metrics'][cls]['precision'].append(precision)
                
                # Calculate recall
                recall = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / np.sum(y_true_bin == 1)
                results['class_metrics'][cls]['recall'].append(recall)
                
                # Calculate F1
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                results['class_metrics'][cls]['f1'].append(f1)
    
    # Calculate summary statistics
    summary = {
        'utility': {
            'mean': np.mean(results['utility']),
            'std': np.std(results['utility']),
            'ci_lower': np.percentile(results['utility'], 2.5),
            'ci_upper': np.percentile(results['utility'], 97.5)
        },
        'auc': {
            'mean': np.mean(results['auc']) if results['auc'] else np.nan,
            'std': np.std(results['auc']) if results['auc'] else np.nan,
            'ci_lower': np.percentile(results['auc'], 2.5) if results['auc'] else np.nan,
            'ci_upper': np.percentile(results['auc'], 97.5) if results['auc'] else np.nan
        },
        'high_risk_recall': {
            'mean': np.mean(results['high_risk_recall']) if results['high_risk_recall'] else np.nan,
            'std': np.std(results['high_risk_recall']) if results['high_risk_recall'] else np.nan,
            'ci_lower': np.percentile(results['high_risk_recall'], 2.5) if results['high_risk_recall'] else np.nan,
            'ci_upper': np.percentile(results['high_risk_recall'], 97.5) if results['high_risk_recall'] else np.nan
        },
        'class_metrics': {}
    }
    
    # Calculate class-specific summary statistics
    for cls in results['class_metrics']:
        if results['class_metrics'][cls]['precision']:
            summary['class_metrics'][cls] = {
                'precision': {
                    'mean': np.mean(results['class_metrics'][cls]['precision']),
                    'std': np.std(results['class_metrics'][cls]['precision']),
                    'ci_lower': np.percentile(results['class_metrics'][cls]['precision'], 2.5),
                    'ci_upper': np.percentile(results['class_metrics'][cls]['precision'], 97.5)
                },
                'recall': {
                    'mean': np.mean(results['class_metrics'][cls]['recall']),
                    'std': np.std(results['class_metrics'][cls]['recall']),
                    'ci_lower': np.percentile(results['class_metrics'][cls]['recall'], 2.5),
                    'ci_upper': np.percentile(results['class_metrics'][cls]['recall'], 97.5)
                },
                'f1': {
                    'mean': np.mean(results['class_metrics'][cls]['f1']),
                    'std': np.std(results['class_metrics'][cls]['f1']),
                    'ci_lower': np.percentile(results['class_metrics'][cls]['f1'], 2.5),
                    'ci_upper': np.percentile(results['class_metrics'][cls]['f1'], 97.5)
                }
            }
    
    return {
        'raw_results': results,
        'summary': summary
    }

class HighRiskStratifiedKFold:
    """
    Stratified K-fold cross-validation with risk level stratification.
    
    This cross-validation strategy ensures that high-risk classes are
    proportionally represented in both training and test sets.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initialize the cross-validator.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of folds.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state if random_state is not None else settings.RANDOM_SEED
        
        # Create underlying KFold
        self.stratified_kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=self.random_state
        )
    
    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Parameters:
        -----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        groups : array-like, default=None
            Group labels for samples. Not used, included for API compatibility.
            
        Yields:
        -------
        tuple : (train_index, test_index)
            Indices for training and test sets.
        """
        # Get risk level for each sample
        risk_levels = np.array([clinical_weights.CLASS_TO_RISK_LEVEL.get(cls, 'moderate_risk') 
                              for cls in y])
        
        # Use stratified k-fold on risk levels
        for train_index, test_index in self.stratified_kfold.split(X, risk_levels):
            yield train_index, test_index
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations.
        
        Parameters:
        -----------
        X : array-like, default=None
            Not used, included for API compatibility.
        y : array-like, default=None
            Not used, included for API compatibility.
        groups : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        int : Number of splits.
        """
        return self.n_splits

class BalancedGroupKFold:
    """
    Group K-fold cross-validation with balancing of class distributions.
    
    This cross-validator ensures that groups are not split across folds, 
    while also attempting to balance class distributions in each fold.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initialize the cross-validator.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of folds.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state if random_state is not None else settings.RANDOM_SEED
    
    def split(self, X, y, groups):
        """
        Generate indices to split data into training and test sets.
        
        Parameters:
        -----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        groups : array-like
            Group labels for samples.
            
        Yields:
        -------
        tuple : (train_index, test_index)
            Indices for training and test sets.
        """
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
        
        # Convert to numpy arrays
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        # Get unique groups
        unique_groups = np.unique(groups)
        
        if self.shuffle:
            # Shuffle the groups
            rng.shuffle(unique_groups)
        
        # Get group class distributions
        group_class_counts = {}
        for group in unique_groups:
            group_mask = groups == group
            group_y = y[group_mask]
            group_class_counts[group] = {cls: np.sum(group_y == cls) for cls in np.unique(y)}
        
        # Create folds with balanced class distributions
        folds = [[] for _ in range(self.n_splits)]
        fold_class_counts = [{cls: 0 for cls in np.unique(y)} for _ in range(self.n_splits)]
        
        # Sort groups by size (largest first)
        sorted_groups = sorted(unique_groups, key=lambda g: -np.sum(groups == g))
        
        # Assign each group to the fold with the most balanced class distribution
        for group in sorted_groups:
            # Calculate imbalance for each fold if this group is added
            imbalance_scores = []
            for i in range(self.n_splits):
                # Create temporary counts with this group added
                temp_counts = fold_class_counts[i].copy()
                for cls, count in group_class_counts[group].items():
                    temp_counts[cls] += count
                
                # Calculate imbalance score (variance of class proportions)
                total = sum(temp_counts.values())
                if total > 0:
                    class_proportions = [count / total for count in temp_counts.values()]
                    imbalance = np.var(class_proportions)
                else:
                    imbalance = 0
                
                imbalance_scores.append(imbalance)
            
            # Assign to fold with lowest imbalance
            fold_idx = np.argmin(imbalance_scores)
            folds[fold_idx].append(group)
            
            # Update fold class counts
            for cls, count in group_class_counts[group].items():
                fold_class_counts[fold_idx][cls] += count
        
        # Generate train/test indices for each fold
        for i in range(self.n_splits):
            test_groups = folds[i]
            train_groups = [g for j, fold in enumerate(folds) for g in fold if j != i]
            
            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(np.isin(groups, train_groups))[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations.
        
        Parameters:
        -----------
        X : array-like, default=None
            Not used, included for API compatibility.
        y : array-like, default=None
            Not used, included for API compatibility.
        groups : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        int : Number of splits.
        """
        return self.n_splits

def cross_validate_with_metrics(model, X, y, cv=None, groups=None, scoring=None):
    """
    Cross-validate model with clinical metrics.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate.
    X : array-like
        Features.
    y : array-like
        Target values.
    cv : cross-validation generator, default=None
        CV generator. If None, uses HighRiskStratifiedKFold.
    groups : array-like, default=None
        Group labels for samples. Only used if cv is a GroupKFold.
    scoring : list, default=None
        Metrics to evaluate. If None, uses clinical utility and AUC.
        
    Returns:
    --------
    dict : Cross-validation results.
    """
    from sklearn.base import clone
    
    # Default CV
    if cv is None:
        cv = HighRiskStratifiedKFold(n_splits=5)
    
    # Default scoring
    if scoring is None:
        scoring = ['clinical_utility', 'risk_adjusted_auc', 'high_risk_recall']
    
    # Initialize results
    results = {
        'fold': [],
        'train_indices': [],
        'test_indices': [],
        'clinical_utility': [],
        'risk_adjusted_auc': [],
        'high_risk_recall': [],
        'train_time': [],
        'predict_time': [],
        'class_metrics': {}
    }
    
    # Initialize class-specific metrics
    for cls in np.unique(y):
        results['class_metrics'][cls] = {
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        # Get train/test split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Clone model
        model_clone = clone(model)
        
        # Fit model on training data
        import time
        start_time = time.time()
        model_clone.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Get predictions on test data
        start_time = time.time()
        y_pred = model_clone.predict(X_test)
        predict_time = time.time() - start_time
        
        # Store fold info
        results['fold'].append(fold)
        results['train_indices'].append(train_index)
        results['test_indices'].append(test_index)
        results['train_time'].append(train_time)
        results['predict_time'].append(predict_time)
        
        # Calculate clinical utility
        if 'clinical_utility' in scoring:
            utility, _ = clinical_utility_score(y_test, y_pred)
            results['clinical_utility'].append(utility)
        
        # Calculate AUC if possible
        if 'risk_adjusted_auc' in scoring:
            try:
                y_proba = model_clone.predict_proba(X_test)
                auc, _ = risk_adjusted_auc(y_test, y_proba)
                results['risk_adjusted_auc'].append(auc)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                results['risk_adjusted_auc'].append(np.nan)
        
        # Calculate high-risk recall
        if 'high_risk_recall' in scoring:
            high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
            high_risk_mask = np.isin(y_test, high_risk_classes)
            
            if np.any(high_risk_mask):
                high_risk_y_true = y_test[high_risk_mask]
                high_risk_y_pred = y_pred[high_risk_mask]
                
                # Calculate recall for high-risk samples
                correct = np.sum(high_risk_y_true == high_risk_y_pred)
                total = len(high_risk_y_true)
                high_risk_recall = correct / total if total > 0 else np.nan
                
                results['high_risk_recall'].append(high_risk_recall)
            else:
                results['high_risk_recall'].append(np.nan)
        
        # Calculate class-specific metrics
        for cls in np.unique(y):
            # Check if class is in test set
            if cls not in y_test:
                for metric in ['precision', 'recall', 'f1']:
                    results['class_metrics'][cls][metric].append(np.nan)
                continue
            
            # Binary predictions for this class
            y_true_bin = (y_test == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['class_metrics'][cls]['precision'].append(precision)
            results['class_metrics'][cls]['recall'].append(recall)
            results['class_metrics'][cls]['f1'].append(f1)
    
    # Calculate mean and std for each metric
    summary = {}
    
    for metric in ['clinical_utility', 'risk_adjusted_auc', 'high_risk_recall', 'train_time', 'predict_time']:
        if metric in results and results[metric]:
            summary[metric] = {
                'mean': np.mean(results[metric]),
                'std': np.std(results[metric])
            }
    
    # Calculate class-specific summary
    summary['class_metrics'] = {}
    for cls in results['class_metrics']:
        summary['class_metrics'][cls] = {}
        for metric in ['precision', 'recall', 'f1']:
            if results['class_metrics'][cls][metric]:
                summary['class_metrics'][cls][metric] = {
                    'mean': np.mean(results['class_metrics'][cls][metric]),
                    'std': np.std(results['class_metrics'][cls][metric])
                }
    
    return {
        'results': results,
        'summary': summary
    }

def temporal_cross_validation(model, temporal_data, id_col, time_col, feature_cols, target_col,
                             initial_train_periods=2, step=1, max_train_periods=None):
    """
    Perform temporal cross-validation for time series data.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate.
    temporal_data : pandas.DataFrame
        Temporal data with multiple time points per subject.
    id_col : str
        Column containing subject identifiers.
    time_col : str
        Column containing time points.
    feature_cols : list
        Columns to use as features.
    target_col : str
        Column to use as target.
    initial_train_periods : int, default=2
        Number of initial time periods to use for training.
    step : int, default=1
        Number of time periods to step forward in each iteration.
    max_train_periods : int, default=None
        Maximum number of time periods to use for training.
        If None, uses all available periods.
        
    Returns:
    --------
    dict : Temporal cross-validation results.
    """
    from sklearn.base import clone
    
    # Get unique time points
    time_points = sorted(temporal_data[time_col].unique())
    
    # Determine max_train_periods if not specified
    if max_train_periods is None:
        max_train_periods = len(time_points) - 1
    
    # Initialize results
    results = {
        'train_periods': [],
        'test_period': [],
        'n_train_samples': [],
        'n_test_samples': [],
        'clinical_utility': [],
        'high_risk_recall': [],
        'class_distributions': []
    }
    
    # Iterate over time points
    for i in range(initial_train_periods, len(time_points), step):
        # Get train and test periods
        train_periods = time_points[:i]
        test_period = time_points[i]
        
        # Skip if exceeding max_train_periods
        if len(train_periods) > max_train_periods:
            train_periods = train_periods[-max_train_periods:]
        
        # Split data
        train_mask = temporal_data[time_col].isin(train_periods)
        test_mask = temporal_data[time_col] == test_period
        
        train_data = temporal_data[train_mask]
        test_data = temporal_data[test_mask]
        
        # Skip if not enough data
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        
        # Get features and target
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        # Clone model
        model_clone = clone(model)
        
        # Fit model
        model_clone.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        utility, _ = clinical_utility_score(y_test, y_pred)
        
        # Calculate high-risk recall
        high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
        high_risk_mask = np.isin(y_test, high_risk_classes)
        
        if np.any(high_risk_mask):
            high_risk_y_true = y_test[high_risk_mask]
            high_risk_y_pred = y_pred[high_risk_mask]
            
            # Calculate recall for high-risk samples
            correct = np.sum(high_risk_y_true == high_risk_y_pred)
            total = len(high_risk_y_true)
            high_risk_recall = correct / total if total > 0 else np.nan
        else:
            high_risk_recall = np.nan
        
        # Store results
        results['train_periods'].append(train_periods)
        results['test_period'].append(test_period)
        results['n_train_samples'].append(len(train_data))
        results['n_test_samples'].append(len(test_data))
        results['clinical_utility'].append(utility)
        results['high_risk_recall'].append(high_risk_recall)
        
        # Store class distributions
        train_class_dist = {cls: np.sum(y_train == cls) for cls in np.unique(y_train)}
        test_class_dist = {cls: np.sum(y_test == cls) for cls in np.unique(y_test)}
        
        results['class_distributions'].append({
            'train': train_class_dist,
            'test': test_class_dist
        })
    
    # Calculate summary
    if results['clinical_utility']:
        summary = {
            'clinical_utility': {
                'mean': np.mean(results['clinical_utility']),
                'std': np.std(results['clinical_utility']),
                'trend': results['clinical_utility']
            },
            'high_risk_recall': {
                'mean': np.nanmean(results['high_risk_recall']),
                'std': np.nanstd(results['high_risk_recall']),
                'trend': results['high_risk_recall']
            }
        }
    else:
        summary = {}
    
    return {
        'results': results,
        'summary': summary
    }

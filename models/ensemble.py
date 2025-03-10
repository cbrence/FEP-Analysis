"""
Custom ensemble model focused on high-risk cases.

This module implements a specialized ensemble approach that prioritizes
the correct identification of high-risk cases (no remission and early relapse)
to prevent FEP relapses.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import GradientBoostingClassifier

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights
from models.base import BaseFEPModel

class HighRiskFocusedEnsemble(BaseFEPModel):
    """
    Ensemble model that combines a general model with a specialized high-risk detector.
    
    This model uses a two-level approach:
    1. A main model for general classification
    2. A specialized model focused only on high-risk classes
    
    The specialized model can override the main model when it detects a high
    probability of a high-risk outcome.
    """
    
    def __init__(self, main_model=None, high_risk_threshold=0.7, random_state=42):
        """
        Initialize the high-risk focused ensemble.
        
        Parameters:
        -----------
        main_model : estimator, default=None
            The main classifier. If None, uses GradientBoostingClassifier.
        high_risk_threshold : float, default=0.7
            Threshold for high-risk model to override the main model.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        super().__init__(random_state=random_state)
        self.main_model = main_model or GradientBoostingClassifier(random_state=random_state)
        self.high_risk_threshold = high_risk_threshold
        self.high_risk_models = {}  # One model per high-risk class
    
    def _fit_model(self, X, y, sample_weight=None):
        """
        Fit both the main model and specialized high-risk models.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        """
        # Fit the main model
        self.main_model.fit(X, y, sample_weight=sample_weight)
        
        # Identify high-risk classes
        high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
        
        # For each high-risk class, train a specialized binary classifier
        for cls in high_risk_classes:
            if cls in np.unique(y):
                # Create binary target (this class vs others)
                y_binary = (y == cls).astype(int)
                
                # Create a specialized model with high recall
                high_risk_model = GradientBoostingClassifier(
                    random_state=self.random_state,
                    # Parameters optimized for recall
                    learning_rate=0.1,
                    n_estimators=200,
                    max_depth=4,
                    subsample=0.8,
                    min_samples_leaf=10
                )
                
                # Create sample weights that heavily penalize false negatives
                if sample_weight is None:
                    hr_sample_weight = np.ones(len(y))
                else:
                    hr_sample_weight = sample_weight.copy()
                
                # Increase weight for the positive class (high-risk)
                hr_sample_weight[y_binary == 1] *= 5.0
                
                # Fit the high-risk model
                high_risk_model.fit(X, y_binary, sample_weight=hr_sample_weight)
                
                # Store the model
                self.high_risk_models[cls] = high_risk_model
    
    def _predict_proba_implementation(self, X):
        """
        Predict class probabilities using the ensemble approach.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples, n_classes)
            Class probabilities.
        """
        # Get predictions from main model
        main_probs = self.main_model.predict_proba(X)
        
        # Initialize probabilities array
        probas = main_probs.copy()
        
        # For each high-risk class, potentially override predictions
        for cls in self.high_risk_models:
            # Get the binary classifier for this high-risk class
            high_risk_model = self.high_risk_models[cls]
            
            # Get binary predictions (probability of this class)
            hr_probs = high_risk_model.predict_proba(X)[:, 1]
            
            # Find the index of this class in the main model's classes
            cls_idx = np.where(self.main_model.classes_ == cls)[0][0]
            
            # For samples with high probability from the specialized model,
            # boost the probability from the main model
            boost_mask = hr_probs >= self.high_risk_threshold
            
            if np.any(boost_mask):
                # Calculate boost amount (how much to increase probability)
                # This is a weighted average of the two models
                boost = 0.7 * hr_probs[boost_mask] + 0.3 * probas[boost_mask, cls_idx]
                
                # Apply the boost, ensuring we don't exceed 1.0
                probas[boost_mask, cls_idx] = np.minimum(boost, 0.99)
                
                # Renormalize the row to ensure probabilities sum to 1
                row_sums = probas[boost_mask].sum(axis=1)
                probas[boost_mask] = probas[boost_mask] / row_sums[:, np.newaxis]
        
        return probas

class TimeDecayEnsemble(BaseFEPModel):
    """
    Ensemble model that incorporates time-based risk in predictions.
    
    This model adjusts predictions based on the time since baseline,
    with more emphasis on early detection for high-risk cases.
    """
    
    def __init__(self, base_model=None, time_decay=0.9, random_state=42):
        """
        Initialize the time decay ensemble.
        
        Parameters:
        -----------
        base_model : estimator, default=None
            The base classifier. If None, uses GradientBoostingClassifier.
        time_decay : float, default=0.9
            Time decay factor for adjusting predictions over time.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        super().__init__(random_state=random_state)
        self.base_model = base_model or GradientBoostingClassifier(random_state=random_state)
        self.time_decay = time_decay
    
    def _fit_model(self, X, y, sample_weight=None):
        """
        Fit the base model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        """
        # Check if time column is present
        if not hasattr(self, 'time_col') or self.time_col not in X.columns:
            raise ValueError("Time column not specified or not found in data")
        
        # Extract time values
        self.time_values_ = X[self.time_col].values
        
        # Drop time column for model training
        X_model = X.drop(columns=[self.time_col])
        
        # Fit the base model
        self.base_model.fit(X_model, y, sample_weight=sample_weight)
    
    def fit(self, X, y, time_col='days_since_baseline', sample_weight=None):
        """
        Fit the model with time information.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        time_col : str, default='days_since_baseline'
            Column containing time information.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        self.time_col = time_col
        return super().fit(X, y, sample_weight=sample_weight)
    
    def _predict_proba_implementation(self, X):
        """
        Predict class probabilities with time-based adjustments.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples, n_classes)
            Class probabilities.
        """
        # Check if time column is present
        if self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in input data")
        
        # Extract time values
        time_values = X[self.time_col].values
        
        # Drop time column for prediction
        X_model = X.drop(columns=[self.time_col])
        
        # Get base predictions
        base_probs = self.base_model.predict_proba(X_model)
        
        # Adjust probabilities based on time
        adjusted_probs = base_probs.copy()
        
        # Normalize time to 0-1 range
        max_time = np.max(time_values)
        norm_time = time_values / max_time if max_time > 0 else time_values
        
        # Calculate time-based weights (earlier times get stronger adjustments)
        time_weights = self.time_decay ** norm_time
        
        # For each high-risk class, adjust probabilities based on time
        high_risk_classes = clinical_weights.HIGH_RISK_CLASSES
        for cls in high_risk_classes:
            if cls in self.base_model.classes_:
                cls_idx = np.where(self.base_model.classes_ == cls)[0][0]
                
                # Adjust high-risk class probabilities
                for i in range(len(adjusted_probs)):
                    # Increase probability for earlier times
                    weight = time_weights[i]
                    
                    # Original probability
                    orig_prob = adjusted_probs[i, cls_idx]
                    
                    # Adjusted probability - higher for earlier times
                    adj_prob = orig_prob + (1 - orig_prob) * (1 - weight) * 0.2
                    
                    # Update probability
                    adjusted_probs[i, cls_idx] = adj_prob
                
                # Renormalize probabilities to sum to 1
                row_sums = adjusted_probs.sum(axis=1)
                adjusted_probs = adjusted_probs / row_sums[:, np.newaxis]
        
        return adjusted_probs
    
    def predict(self, X):
        """
        Predict class labels with time-based adjustments.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples,)
            Predicted class labels.
        """
        return super().predict(X)

def stacked_prediction(X, main_model, high_risk_model, high_risk_threshold=0.7):
    """
    Combine predictions from a main model and a high-risk focused model.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    main_model : estimator
        The main classifier for all classes.
    high_risk_model : estimator
        Specialized model for detecting high-risk cases.
    high_risk_threshold : float, default=0.7
        Threshold for high-risk model to override the main model.
        
    Returns:
    --------
    array-like : Predicted probabilities combining both models.
    """
    # Get predictions from both models
    main_probs = main_model.predict_proba(X)
    high_risk_probs = high_risk_model.predict_proba(X)
    
    # Initialize combined predictions with main model's predictions
    combined_probs = main_probs.copy()
    
    # Identify samples where high risk model is confident
    high_risk_idx = np.where(high_risk_probs[:, 1] > high_risk_threshold)[0]
    
    if len(high_risk_idx) > 0:
        # For these samples, increase probability of high-risk class
        high_risk_class_idx = 0  # Assuming first class is the high-risk class in main model
        
        # Calculate boost (weighted combination)
        boost = 0.7 * high_risk_probs[high_risk_idx, 1] + 0.3 * combined_probs[high_risk_idx, high_risk_class_idx]
        
        # Apply boost
        combined_probs[high_risk_idx, high_risk_class_idx] = boost
        
        # Renormalize
        row_sums = combined_probs[high_risk_idx].sum(axis=1)
        combined_probs[high_risk_idx] = combined_probs[high_risk_idx] / row_sums[:, np.newaxis]
    
    return combined_probs
"""
Base model class for FEP outcome prediction.

This module defines a base class for all predictive models with methods
for cost-sensitive training and evaluation to prioritize preventing relapse.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import clinical_weights, settings

# Setup logging
logger = logging.getLogger(__name__)

class BaseFEPModel(ABC, BaseEstimator, ClassifierMixin):
    """
    Base class for FEP outcome prediction models.
    
    Implements cost-sensitive training and evaluation methods that prioritize
    the correct identification of high-risk cases. All specific model implementations
    should inherit from this class.
    """
    
    def __init__(self, class_weight='clinical', random_state=None, model_name=None):
        """
        Initialize the base model with cost-sensitive parameters.
        
        Parameters:
        -----------
        class_weight : str or dict, default='clinical'
            Class weights to use in training. Options:
            - 'clinical': Use weights defined in clinical_weights.py
            - 'balanced': Automatically adjust weights inversely proportional to class frequencies
            - dict: Custom dictionary mapping classes to weights
        random_state : int, default=None
            Random seed for reproducibility. If None, uses setting from settings.py
        model_name : str, default=None
            Name for the model. If None, uses class name.
        """
        self.class_weight = class_weight
        self.random_state = random_state if random_state is not None else settings.RANDOM_SEED
        self.thresholds = None
        self.model_name = model_name if model_name is not None else self.__class__.__name__
        self.feature_names_ = None
        self.training_history_ = {}
        
    def _get_class_weights(self, y):
        """
        Get class weights based on the specified strategy.
        
        Parameters:
        -----------
        y : array-like
            Target values.
            
        Returns:
        --------
        dict : Dictionary mapping classes to weights.
        """
        if self.class_weight == 'clinical':
            logger.info("Using clinical weights from configuration")
            return clinical_weights.CLASS_WEIGHTS
        elif self.class_weight == 'balanced':
            logger.info("Computing balanced class weights")
            # Calculate weights inversely proportional to class frequencies
            class_counts = np.bincount(y)
            total_samples = len(y)
            weights = {cls: total_samples / (len(class_counts) * count) 
                      for cls, count in enumerate(class_counts) if count > 0}
            return weights
        elif isinstance(self.class_weight, dict):
            logger.info("Using custom class weights")
            return self.class_weight
        else:
            logger.info("No specific class weights applied")
            return None
    
    def fit(self, X, y, sample_weight=None, feature_names=None, optimize_thresholds=True):
        """
        Fit the model with cost-sensitive considerations.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None and class_weight is specified, 
            sample weights will be derived from class weights.
        feature_names : list, default=None
            Names of features. Used for feature importance reporting.
        optimize_thresholds : bool, default=True
            Whether to optimize prediction thresholds after training.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Start timing
        start_time = pd.Timestamp.now()
        
        # Record fitting started
        logger.info(f"Fitting {self.model_name} model")
        
        # Check inputs
        if isinstance(X, pd.DataFrame):
            # Save feature names if available
            self.feature_names_ = X.columns.tolist() if feature_names is None else feature_names
            X_array = X.values
        else:
            X_array = X
            self.feature_names_ = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        
        X_array, y = check_X_y(X_array, y)
        
        # Apply clinical weights to samples if needed
        if sample_weight is None and self.class_weight is not None:
            class_weights = self._get_class_weights(y)
            if class_weights is not None:
                logger.info("Deriving sample weights from class weights")
                sample_weight = np.ones(len(y))
                for cls, weight in class_weights.items():
                    sample_weight[y == cls] = weight
                logger.debug(f"Sample weights: min={sample_weight.min()}, max={sample_weight.max()}, mean={sample_weight.mean()}")
        
        # Store fitted values
        self.classes_ = np.unique(y)
        self.n_features_in_ = X_array.shape[1]
        self.n_classes_ = len(self.classes_)
        
        # Fit the model - implementation depends on the specific model
        logger.info("Calling model-specific _fit_model implementation")
        self._fit_model(X_array, y, sample_weight=sample_weight)
        
        # Optimize prediction thresholds using a portion of training data
        if optimize_thresholds:
            logger.info("Optimizing prediction thresholds")
            self._optimize_thresholds(X_array, y)
        
        # Record training history
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        self.training_history_['fit_time'] = training_time
        self.training_history_['n_samples'] = len(y)
        self.training_history_['class_distribution'] = {cls: np.sum(y == cls) for cls in self.classes_}
        
        logger.info(f"Model fitting completed in {training_time:.2f} seconds")
        
        return self
    
    @abstractmethod
    def _fit_model(self, X, y, sample_weight=None):
        """
        Actual model fitting implementation to be defined by subclasses.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        """
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples, n_classes)
            Class probabilities.
        """
        # Check if the model is fitted
        check_is_fitted(self, ['classes_', 'n_features_in_'])
        
        # Convert to array and check input data
        if isinstance(X, pd.DataFrame):
            # If we have feature_names_, try to use only those features if available
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                common_features = [f for f in self.feature_names_ if f in X.columns]
                if len(common_features) == self.n_features_in_:
                    X = X[common_features]
                    logger.info(f"Selected {len(common_features)} matching features for prediction")
                elif len(common_features) > 0:
                    logger.warning(f"Only found {len(common_features)} of {self.n_features_in_} expected features")
                    # Continue with what we have - will be caught by the feature count check below
                else:
                    logger.warning("No matching features found between model and input data")
            
            X_array = X.values
        else:
            X_array = X
        
        X_array = check_array(X_array)
        
        # Handle feature count mismatch
        if X_array.shape[1] != self.n_features_in_:
            if X_array.shape[1] > self.n_features_in_:
                # If we have more features than needed, use only the first n_features_in_ features
                logger.warning(f"X has {X_array.shape[1]} features, but {self.model_name} is expecting {self.n_features_in_} features. Using first {self.n_features_in_} features.")
                X_array = X_array[:, :self.n_features_in_]
            else:
                # If we have fewer features than needed, this is a more serious problem
                raise ValueError(f"X has {X_array.shape[1]} features, but {self.model_name} is expecting {self.n_features_in_} features")
        
        # Return probabilities - implementation depends on the specific model
        return self._predict_proba_implementation(X_array)
    
    @abstractmethod
    def _predict_proba_implementation(self, X):
        """
        Actual probability prediction implementation to be defined by subclasses.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples, n_classes)
            Class probabilities.
        """
        pass
    
    def predict(self, X):
        """
        Predict class labels for X using optimized thresholds.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        array of shape (n_samples,)
            Predicted class labels.
        """
        # Get probabilities
        proba = self.predict_proba(X)
        
        # If thresholds are not set, use default argmax
        if self.thresholds is None:
            logger.warning("No optimized thresholds found, using argmax")
            return self.classes_[np.argmax(proba, axis=1)]
        
        # Apply custom thresholds for improved clinical utility
        predictions = np.zeros(len(X), dtype=int)
        
        # Create a mask to track which samples have been assigned a prediction
        assigned = np.zeros(len(X), dtype=bool)
        
        # Apply thresholds class by class, prioritizing high-risk classes
        # Get priority order from clinical weights
        priority_order = (clinical_weights.HIGH_RISK_CLASSES + 
                         clinical_weights.MODERATE_RISK_CLASSES + 
                         clinical_weights.LOW_RISK_CLASSES)
        
        # Ensure all classes are covered
        all_classes = set(self.classes_)
        missing_classes = all_classes - set(priority_order)
        if missing_classes:
            logger.warning(f"Classes {missing_classes} not found in clinical priority order, adding at end")
            priority_order = list(priority_order) + list(missing_classes)
        
        # Start with highest priority class and work down
        for cls in priority_order:
            if cls in self.classes_:
                cls_idx = np.where(self.classes_ == cls)[0][0]
                threshold = self.thresholds.get(cls, 0.5)  # Default to 0.5 if not specified
                
                # Find samples that exceed the threshold and haven't been assigned yet
                mask = (proba[:, cls_idx] >= threshold) & (~assigned)
                
                # Assign predictions
                predictions[mask] = cls
                assigned[mask] = True
        
        # For any remaining unassigned samples, use argmax
        mask = ~assigned
        if np.any(mask):
            logger.debug(f"Assigning {np.sum(mask)} samples using argmax")
            predictions[mask] = self.classes_[np.argmax(proba[mask], axis=1)]
        
        return predictions
    
    def _optimize_thresholds(self, X, y):
        """
        Optimize prediction thresholds to maximize clinical utility.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Check if not enough data for threshold optimization
        if len(y) < 50:  # arbitrary minimum size
            logger.warning("Not enough samples for threshold optimization")
            self.thresholds = {cls: clinical_weights.PREDICTION_THRESHOLDS.get(cls, 0.5) 
                              for cls in self.classes_}
            return
        
        # Use a portion of training data for threshold optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Refit on the training portion
        self._fit_model(X_train, y_train)
        
        # Predict probabilities on validation set
        val_probs = self._predict_proba_implementation(X_val)
        
        # Initialize thresholds with clinical default values
        self.thresholds = {}
        
        # For each class, find threshold that maximizes clinical utility
        for cls in self.classes_:
            cls_idx = np.where(self.classes_ == cls)[0][0]
            
            # Get risk level for this class
            risk_level = clinical_weights.CLASS_TO_RISK_LEVEL.get(int(cls), 'moderate_risk')
            
            # Get costs for false negatives and false positives for this risk level
            fn_cost = clinical_weights.ERROR_COSTS['false_negative'][risk_level]
            fp_cost = clinical_weights.ERROR_COSTS['false_positive'][risk_level]
            
            # Find optimal threshold
            thresholds = np.linspace(0.1, 0.9, 50)
            best_threshold = 0.5  # Default
            best_cost = float('inf')
            
            for threshold in thresholds:
                # Make binary prediction for this class
                y_pred = (val_probs[:, cls_idx] >= threshold).astype(int)
                y_true = (y_val == cls).astype(int)
                
                # Calculate false negatives and false positives
                fn = np.sum((y_true == 1) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                
                # Calculate weighted cost
                total_cost = fn * fn_cost + fp * fp_cost
                
                # Update best threshold if this is better
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_threshold = threshold
            
            # Use default threshold from clinical weights if available
            default_threshold = clinical_weights.PREDICTION_THRESHOLDS.get(int(cls), 0.5)
            
            # Only use optimized threshold if significantly different from default
            if abs(best_threshold - default_threshold) > 0.1:
                logger.info(f"Using optimized threshold {best_threshold:.2f} for class {cls} (default was {default_threshold:.2f})")
                self.thresholds[cls] = best_threshold
            else:
                logger.info(f"Using default threshold {default_threshold:.2f} for class {cls}")
                self.thresholds[cls] = default_threshold
        
        # Refit on the entire dataset
        self._fit_model(X, y)
        
        # Save thresholds to training history
        self.training_history_['optimized_thresholds'] = self.thresholds.copy()
    
    def get_sample_weights(self, y):
        """
        Generate sample weights based on clinical priorities.
        
        Parameters:
        -----------
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Sample weights.
        """
        weights = np.ones(len(y))
        
        # Apply class weights
        class_weights = self._get_class_weights(y)
        if class_weights:
            for cls, weight in class_weights.items():
                weights[y == cls] = weight
            
        return weights
    
    def get_feature_importance(self):
        """
        Get feature importance scores if the model supports it.
        
        Returns:
        --------
        pandas.DataFrame : DataFrame with feature names and importance scores.
        """
        try:
            importances = self._get_feature_importance_values()
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance': importances
            })
            
            # Sort by importance
            return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
        except (NotImplementedError, NotFittedError) as e:
            logger.warning(f"Feature importance not available: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance'])
    
    def _get_feature_importance_values(self):
        """
        Extract raw feature importance values from the model.
        
        This method should be implemented by subclasses if they support
        feature importance extraction.
        
        Returns:
        --------
        array-like : Feature importance values.
        
        Raises:
        -------
        NotImplementedError : If the model doesn't support feature importance.
        """
        raise NotImplementedError("This model doesn't provide feature importance values")
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str, default=None
            Path to save the model. If None, uses default path based on model name.
            
        Returns:
        --------
        str : Path where the model was saved.
        """
        if filepath is None:
            # Create default filename based on model name
            models_dir = settings.MODELS_DIR
            filepath = os.path.join(models_dir, f"{self.model_name.lower().replace(' ', '_')}.joblib")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        BaseFEPModel : Loaded model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns:
        --------
        dict : Parameter names mapped to their values.
        """
        return {"class_weight": self.class_weight, 
                "random_state": self.random_state,
                "model_name": self.model_name}
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **parameters : dict
            Estimator parameters.
            
        Returns:
        --------
        self : object
            Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def __str__(self):
        """String representation of the model."""
        return f"{self.model_name} (BaseFEPModel)"
    
    def __repr__(self):
        """Detailed string representation of the model."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

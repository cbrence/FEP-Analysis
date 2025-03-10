"""
Feature selection utilities for FEP analysis.

This module provides functions for selecting the most relevant features
for predicting FEP outcomes, helping to reduce dimensionality and improve model interpretability.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Setup logging
logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature selector combining multiple methods.
    
    This transformer combines various feature selection techniques and
    allows for a flexible, multi-step approach to feature selection.
    """
    
    def __init__(self, method='importance', threshold=None, n_features=None, 
                 model=None, cv=5, random_state=None):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        method : str or list, default='importance'
            Feature selection method(s):
            - 'importance': Select based on feature importance from a tree-based model
            - 'statistical': Select based on statistical tests (f_classif)
            - 'mutual_info': Select based on mutual information
            - 'variance': Remove low-variance features
            - 'rfe': Recursive feature elimination
            - 'pca': Principal Component Analysis (transforms features)
            - If list, applies methods in order
        threshold : float, default=None
            Threshold for feature importance (if method is 'importance' or 'variance').
            If None, uses n_features instead.
        n_features : int, default=None
            Number of features to select.
            If None and threshold is None, keeps all features.
        model : estimator, default=None
            Model to use for importance-based selection.
            If None, uses RandomForestClassifier.
        cv : int, default=5
            Number of cross-validation folds for RFECV.
        random_state : int, default=None
            Random seed for reproducibility. If None, uses settings.
        """
        self.method = method
        self.threshold = threshold
        self.n_features = n_features
        self.model = model
        self.cv = cv
        self.random_state = random_state if random_state is not None else settings.RANDOM_SEED
        
        # Initialize attributes
        self.selected_features_ = None
        self.feature_importances_ = None
        self.pca_components_ = None
        self.support_mask_ = None
    
    def fit(self, X, y):
        """
        Identify important features and create selection mask.
        
        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input features.
        y : array-like
            Target variable.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Extract feature names
        feature_names = X.columns.tolist()
        
        # Handle list of methods
        if isinstance(self.method, list):
            logger.info(f"Applying multiple feature selection methods: {self.method}")
            
            # Create a copy to avoid modifying the original
            X_selected = X.copy()
            
            # Apply each method in sequence
            for method in self.method:
                selector = FeatureSelector(
                    method=method, 
                    threshold=self.threshold,
                    n_features=self.n_features,
                    model=self.model,
                    cv=self.cv,
                    random_state=self.random_state
                )
                selector.fit(X_selected, y)
                X_selected = selector.transform(X_selected)
            
            # Store results from the final selection
            self.selected_features_ = X_selected.columns.tolist()
            self.support_mask_ = np.isin(feature_names, self.selected_features_)
            self.feature_importances_ = pd.Series(
                np.zeros(len(feature_names)),
                index=feature_names
            )
            self.feature_importances_[self.selected_features_] = 1.0
            
            return self
        
        # Initialize importance series
        self.feature_importances_ = pd.Series(np.zeros(len(feature_names)), index=feature_names)
        
        # Apply the specified method
        if self.method == 'importance':
            self._select_by_importance(X, y)
        elif self.method == 'statistical':
            self._select_by_statistical_test(X, y)
        elif self.method == 'mutual_info':
            self._select_by_mutual_info(X, y)
        elif self.method == 'variance':
            self._select_by_variance(X)
        elif self.method == 'rfe':
            self._select_by_rfe(X, y)
        elif self.method == 'pca':
            self._select_by_pca(X)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        
        # Create mask for selected features
        if self.method != 'pca':  # PCA transforms features rather than selecting them
            if self.threshold is not None:
                self.support_mask_ = self.feature_importances_ >= self.threshold
            elif self.n_features is not None:
                # Select top n_features
                top_features = self.feature_importances_.nlargest(self.n_features).index
                self.support_mask_ = np.isin(feature_names, top_features)
            else:
                # Keep all features
                self.support_mask_ = np.ones(len(feature_names), dtype=bool)
            
            # Get list of selected features
            self.selected_features_ = [feature_names[i] for i in range(len(feature_names)) if self.support_mask_[i]]
            logger.info(f"Selected {len(self.selected_features_)} features using method '{self.method}'")
        
        return self
    
    def transform(self, X):
        """
        Transform the data using the selected features.
        
        Parameters:
        -----------
        X : pandas.DataFrame or array-like
            Input features.
            
        Returns:
        --------
        pandas.DataFrame : Transformed data with selected features.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply transformation based on method
        if self.method == 'pca':
            if self.pca_components_ is not None:
                # Transform using PCA
                pca_result = self.pca_components_.transform(X)
                return pd.DataFrame(
                    pca_result, 
                    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
                    index=X.index
                )
            else:
                raise ValueError("PCA components not found. Call fit() first.")
        else:
            # Ensure fit has been called
            if self.selected_features_ is None:
                raise ValueError("Feature selection not performed. Call fit() first.")
            
            # Select features
            return X[self.selected_features_]
    
    def _select_by_importance(self, X, y):
        """Select features based on importance from a tree-based model."""
        logger.info("Selecting features by importance")
        
        # Create model if not provided
        if self.model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            model = self.model
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Model does not provide feature importances")
        
        # Store feature importances
        self.feature_importances_ = pd.Series(importances, index=X.columns)
    
    def _select_by_statistical_test(self, X, y):
        """Select features based on statistical tests."""
        logger.info("Selecting features by statistical tests")
        
        # Determine number of features to select
        k = self.n_features if self.n_features is not None else 'all'
        
        # Create selector
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = selector.scores_
        
        # Store feature importances
        self.feature_importances_ = pd.Series(scores, index=X.columns)
    
    def _select_by_mutual_info(self, X, y):
        """Select features based on mutual information."""
        logger.info("Selecting features by mutual information")
        
        # Determine number of features to select
        k = self.n_features if self.n_features is not None else 'all'
        
        # Create selector
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = selector.scores_
        
        # Store feature importances
        self.feature_importances_ = pd.Series(scores, index=X.columns)
    
    def _select_by_variance(self, X):
        """Remove low-variance features."""
        logger.info("Selecting features by variance threshold")
        
        # Determine threshold
        threshold = self.threshold if self.threshold is not None else 0.0
        
        # Create selector
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get support mask
        support = selector.get_support()
        
        # Convert support mask to feature importances
        importances = np.zeros(len(X.columns))
        importances[support] = 1.0
        
        # Also store variances as feature importances
        variances = selector.variances_
        
        # Store feature importances
        self.feature_importances_ = pd.Series(variances, index=X.columns)
        self.support_mask_ = support
    
    def _select_by_rfe(self, X, y):
        """Select features using recursive feature elimination."""
        logger.info("Selecting features by recursive feature elimination")
        
        # Create model if not provided
        if self.model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            model = self.model
        
        # Determine if we should use cross-validation
        if self.n_features is None:
            # Use cross-validation to determine optimal number of features
            selector = RFECV(
                estimator=model,
                step=1,
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1
            )
        else:
            # Use specified number of features
            selector = RFE(
                estimator=model,
                n_features_to_select=self.n_features,
                step=1
            )
        
        # Fit selector
        selector.fit(X, y)
        
        # Get support mask and feature ranking
        support = selector.support_
        ranking = selector.ranking_
        
        # Map ranking to importance (lower rank = higher importance)
        max_rank = np.max(ranking)
        importances = max_rank - ranking + 1
        importances = importances / max_rank  # Normalize to 0-1
        
        # Store feature importances
        self.feature_importances_ = pd.Series(importances, index=X.columns)
        self.support_mask_ = support
    
    def _select_by_pca(self, X):
        """Transform features using PCA."""
        logger.info("Transforming features using PCA")
        
        # Determine number of components
        n_components = self.n_features if self.n_features is not None else 'mle'
        
        # Create PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(X)
        
        # Store PCA components for transform
        self.pca_components_ = pca
        
        # Calculate feature importances based on component loadings
        loadings = pca.components_.T
        
        # Use average absolute loading across all components as importance
        importances = np.abs(loadings).mean(axis=1)
        
        # Store feature importances
        self.feature_importances_ = pd.Series(importances, index=X.columns)
        
        # For PCA, we'll set n_components as selected_features_
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance >= 0.95) + 1 if n_components == 'mle' else pca.n_components_
        self.selected_features_ = [f'PC{i+1}' for i in range(n_components)]
        logger.info(f"PCA selected {n_components} components explaining 95% of variance")
    
    def plot_feature_importance(self, n_features=20, figsize=(12, 8)):
        """
        Plot feature importances.
        
        Parameters:
        -----------
        n_features : int, default=20
            Number of top features to show.
        figsize : tuple, default=(12, 8)
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure.
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Call fit() first.")
        
        # Get top features
        top_features = self.feature_importances_.nlargest(n_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        top_features.sort_values().plot(kind='barh', ax=ax)
        
        # Add labels
        ax.set_title(f'Top {n_features} Features by Importance')
        ax.set_xlabel('Importance')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        return fig

def select_features_by_importance(X, y, n_features=None, threshold=None, model=None, random_state=None):
    """
    Select features based on their importance from a tree-based model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : array-like
        Target variable.
    n_features : int, default=None
        Number of top features to select.
    threshold : float, default=None
        Importance threshold for selecting features.
    model : estimator, default=None
        Model to use for extracting feature importance.
        If None, uses RandomForestClassifier.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with selected features.
    pandas.Series : Feature importances.
    """
    # Create selector
    selector = FeatureSelector(
        method='importance',
        threshold=threshold,
        n_features=n_features,
        model=model,
        random_state=random_state
    )
    
    # Fit and transform
    selector.fit(X, y)
    X_selected = selector.transform(X)
    
    return X_selected, selector.feature_importances_

def select_features_by_correlation(X, y=None, threshold=0.8, target_col=None):
    """
    Select features by removing highly correlated features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : array-like, default=None
        Target variable (not used, included for API consistency).
    threshold : float, default=0.8
        Correlation threshold for feature removal.
    target_col : str, default=None
        Column name of the target variable in X.
        If provided, correlations with the target are considered in selection.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with selected features.
    pandas.DataFrame : Correlation matrix of selected features.
    """
    # Make a copy to avoid modifying the original
    X_selected = X.copy()
    
    # Calculate correlation matrix
    corr_matrix = X_selected.corr().abs()
    
    # If target column is provided, keep feature with higher correlation to target
    target_corr = None
    if target_col is not None and target_col in X_selected.columns:
        target_corr = corr_matrix[target_col].drop(target_col)
    
    # Create an upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation higher than threshold
    to_drop = []
    for col in upper.columns:
        # Get features correlated with current column
        correlated_features = upper.index[upper[col] > threshold]
        
        # Skip if no highly correlated features
        if len(correlated_features) == 0:
            continue
        
        # If target correlation available, keep feature with higher target correlation
        if target_corr is not None:
            for feat in correlated_features:
                # Skip target column
                if feat == target_col:
                    continue
                
                # Keep feature with higher target correlation
                if target_corr[col] < target_corr[feat]:
                    if col not in to_drop:
                        to_drop.append(col)
                else:
                    if feat not in to_drop:
                        to_drop.append(feat)
        else:
            # Without target information, drop correlated features
            to_drop.extend(correlated_features)
    
    # Drop highly correlated features
    to_drop = list(set(to_drop))  # Remove duplicates
    logger.info(f"Dropping {len(to_drop)} highly correlated features")
    X_selected = X_selected.drop(columns=to_drop)
    
    # Recalculate correlation matrix for selected features
    corr_matrix_selected = X_selected.corr().abs()
    
    return X_selected, corr_matrix_selected

def select_features_for_multicollinearity(X, y=None, vif_threshold=10):
    """
    Select features by removing those with high multicollinearity using VIF.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features (numerical only).
    y : array-like, default=None
        Target variable (not used, included for API consistency).
    vif_threshold : float, default=10
        VIF threshold for feature removal.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with selected features.
    pandas.DataFrame : VIF values for initial and final feature sets.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.error("statsmodels is required for VIF calculation. Install with 'pip install statsmodels'")
        return X.copy(), pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    X_selected = X.copy()
    
    # Select only numerical columns
    numerical_cols = X_selected.select_dtypes(include=['int64', 'float64']).columns
    X_numerical = X_selected[numerical_cols]
    
    # Calculate initial VIF for all features
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_numerical.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numerical.values, i) 
                       for i in range(X_numerical.shape[1])]
    
    # Save initial VIF values
    initial_vif = vif_data.copy()
    
    # Iteratively remove features with high VIF
    high_vif = True
    while high_vif:
        # Find feature with highest VIF
        max_vif_feature = vif_data.loc[vif_data["VIF"].idxmax()]
        
        # Check if max VIF exceeds threshold
        if max_vif_feature["VIF"] > vif_threshold:
            # Remove feature with highest VIF
            feature_to_remove = max_vif_feature["feature"]
            logger.info(f"Removing feature {feature_to_remove} with VIF {max_vif_feature['VIF']:.2f}")
            X_numerical = X_numerical.drop(columns=[feature_to_remove])
            
            # Recalculate VIF
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_numerical.columns
            vif_data["VIF"] = [variance_inflation_factor(X_numerical.values, i) 
                               for i in range(X_numerical.shape[1])]
        else:
            high_vif = False
    
    # Get list of selected features
    selected_features = list(X_numerical.columns)
    
    # Create final DataFrame with all original columns that were not removed
    X_final = X_selected[list(set(X_selected.columns) - set(numerical_cols.difference(selected_features)))]
    
    # Create DataFrame with initial and final VIF values
    vif_comparison = pd.DataFrame({
        'feature': initial_vif['feature'],
        'initial_VIF': initial_vif['VIF'],
        'selected': initial_vif['feature'].isin(vif_data['feature'])
    })
    
    # Add final VIF for selected features
    final_vif_dict = dict(zip(vif_data['feature'], vif_data['VIF']))
    vif_comparison['final_VIF'] = vif_comparison['feature'].map(
        lambda x: final_vif_dict.get(x, np.nan)
    )
    
    return X_final, vif_comparison

def select_best_features(X, y, method='combined', n_features=None, threshold=None, random_state=None):
    """
    Comprehensive feature selection combining multiple techniques.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : array-like
        Target variable.
    method : str or list, default='combined'
        Feature selection method(s). If 'combined', uses a combination of methods.
    n_features : int, default=None
        Number of top features to select.
    threshold : float, default=None
        Threshold for feature selection.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with selected features.
    dict : Additional information about selection.
    """
    # Default to combined approach
    if method == 'combined':
        method = ['variance', 'correlation', 'importance']
    
    # Create selector
    selector = FeatureSelector(
        method=method,
        threshold=threshold,
        n_features=n_features,
        random_state=random_state
    )
    
    # Fit and transform
    selector.fit(X, y)
    X_selected = selector.transform(X)
    
    # Create information dictionary
    info = {
        'feature_importances': selector.feature_importances_,
        'selected_features': selector.selected_features_,
        'n_original_features': X.shape[1],
        'n_selected_features': X_selected.shape[1],
        'selection_method': method
    }
    
    return X_selected, info

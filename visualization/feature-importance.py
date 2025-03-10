"""
Feature importance visualization functions for FEP analysis.

This module provides specialized functions for visualizing feature importance,
relationships between features, and feature groupings for FEP prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


def plot_top_features(feature_names: List[str],
                     importance_values: List[float],
                     title: str = "Top Features by Importance",
                     top_n: int = 15,
                     figsize: Tuple[int, int] = (10, 8),
                     color: str = "#1f77b4",
                     show_values: bool = True,
                     group_colors: Optional[Dict[str, str]] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the top N most important features.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importance_values : List[float]
        Importance values corresponding to features
    title : str, default="Top Features by Importance"
        Title of the plot
    top_n : int, default=15
        Number of top features to display
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    color : str, default="#1f77b4"
        Bar color, used if group_colors is None
    show_values : bool, default=True
        Whether to show importance values on bars
    group_colors : Optional[Dict[str, str]], default=None
        Dictionary mapping feature prefixes to colors
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create dataframe for easier sorting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Select top_n features
    df = df.head(top_n)
    
    # Sort by importance for better visualization
    df = df.sort_values('Importance')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine colors for each bar if group_colors is provided
    if group_colors:
        colors = []
        for feature in df['Feature']:
            # Find matching prefix
            for prefix, color in group_colors.items():
                if feature.startswith(prefix):
                    colors.append(color)
                    break
            else:
                # Default color if no prefix matches
                colors.append(color)
    else:
        colors = color
    
    # Plot horizontal bars
    bars = ax.barh(df['Feature'], df['Importance'], color=colors)
    
    # Show values on bars if requested
    if show_values:
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 * max(df['Importance'])
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', va='center')
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Add legend if group colors are used
    if group_colors:
        # Create patches for legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color, label=prefix) 
                 for prefix, color in group_colors.items()]
        ax.legend(handles=patches, title="Feature Groups", loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_groups(X: pd.DataFrame,
                       y: Optional[np.ndarray] = None,
                       feature_groups: Optional[Dict[str, List[str]]] = None,
                       title: str = "Feature Group Importance",
                       metric: str = 'mutual_info',
                       figsize: Tuple[int, int] = (10, 8),
                       color_palette: str = "viridis",
                       show_values: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot importance of feature groups.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    y : Optional[np.ndarray], default=None
        Target variable, required for some metrics
    feature_groups : Optional[Dict[str, List[str]]], default=None
        Dictionary mapping group names to lists of feature names.
        If None, tries to infer groups from feature name prefixes.
    title : str, default="Feature Group Importance"
        Title of the plot
    metric : str, default='mutual_info'
        Metric to use for importance. Options: 'mutual_info', 'variance', 'mean'
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    color_palette : str, default="viridis"
        Color palette for the bars
    show_values : bool, default=True
        Whether to show importance values on bars
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Infer feature groups if not provided
    if feature_groups is None:
        feature_groups = {}
        for col in X.columns:
            # Try to split on first underscore or dot
            for separator in ['_', '.']:
                if separator in col:
                    prefix = col.split(separator)[0]
                    if prefix not in feature_groups:
                        feature_groups[prefix] = []
                    feature_groups[prefix].append(col)
                    break
        
        # Handle features that didn't match any pattern
        ungrouped = [col for col in X.columns if not any(col in group for group in feature_groups.values())]
        if ungrouped:
            feature_groups['Other'] = ungrouped
    
    # Calculate importance for each group
    group_importance = {}
    
    if metric == 'mutual_info' and y is not None:
        # Calculate mutual information for each feature
        selector = SelectKBest(mutual_info_classif, k='all')
        selector.fit(X, y)
        feature_importance = {feature: score for feature, score in zip(X.columns, selector.scores_)}
        
        # Calculate group importance as mean of feature importances
        for group, features in feature_groups.items():
            valid_features = [f for f in features if f in feature_importance]
            if valid_features:
                group_importance[group] = np.mean([feature_importance[f] for f in valid_features])
            else:
                group_importance[group] = 0
    
    elif metric == 'variance':
        # Calculate variance of each feature
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        feature_variance = {feature: X_scaled[feature].var() for feature in X.columns}
        
        # Calculate group importance as mean of feature variances
        for group, features in feature_groups.items():
            valid_features = [f for f in features if f in feature_variance]
            if valid_features:
                group_importance[group] = np.mean([feature_variance[f] for f in valid_features])
            else:
                group_importance[group] = 0
    
    elif metric == 'mean':
        # Simply use the number of features in each group
        group_importance = {group: len(features) for group, features in feature_groups.items()}
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Group': list(group_importance.keys()),
        'Importance': list(group_importance.values()),
        'Count': [len(feature_groups[group]) for group in group_importance.keys()]
    }).sort_values('Importance', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color palette based on number of groups
    colors = sns.color_palette(color_palette, len(df))
    
    # Plot horizontal bars
    bars = ax.barh(df['Group'], df['Importance'], color=colors)
    
    # Show values on bars if requested
    if show_values:
        for i, bar in enumerate(bars):
            width = bar.get_width()
            count = df.iloc[i]['Count']
            label_x_pos = width + 0.01 * max(df['Importance'])
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f} ({count} features)', va='center')
    
    # Set labels and title
    if metric == 'mutual_info':
        ax.set_xlabel('Mean Mutual Information')
    elif metric == 'variance':
        ax.set_xlabel('Mean Variance')
    else:
        ax.set_xlabel('Number of Features')
    
    ax.set_ylabel('Feature Group')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_correlations(X: pd.DataFrame,
                            threshold: float = 0.7,
                            method: str = 'pearson',
                            title: str = "Highly Correlated Features",
                            figsize: Tuple[int, int] = (12, 10),
                            cmap: str = "coolwarm",
                            annotate: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a heatmap of highly correlated features.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    threshold : float, default=0.7
        Correlation threshold for highlighting
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', 'spearman')
    title : str, default="Highly Correlated Features"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    cmap : str, default="coolwarm"
        Colormap for the heatmap
    annotate : bool, default=True
        Whether to annotate cells with correlation values
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Calculate correlation matrix
    corr = X.corr(method=method)
    
    # Create mask for low correlations and self-correlations
    mask = np.triu(np.ones_like(corr, dtype=bool))
    abs_corr = np.abs(corr)
    mask = (abs_corr < threshold) | mask
    
    # If all values are masked, adjust threshold
    if mask.all():
        # Find the highest correlation value
        max_corr = abs_corr.max()
        if max_corr < threshold:
            threshold = max_corr * 0.8
            mask = np.triu(np.ones_like(corr, dtype=bool))
            mask = (abs_corr < threshold) | mask
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
               center=0, annot=annotate, fmt='.2f', square=True, 
               linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Set title with threshold
    ax.set_title(f"{title} (|corr| > {threshold:.2f})")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_cluster_map(X: pd.DataFrame,
                            method: str = 'ward',
                            metric: str = 'euclidean',
                            title: str = "Feature Clustering",
                            figsize: Tuple[int, int] = (15, 12),
                            cmap: str = "vlag",
                            row_cluster: bool = True,
                            col_cluster: bool = True,
                            standard_scale: int = 1,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a clustered heatmap of features.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    title : str, default="Feature Clustering"
        Title of the plot
    figsize : Tuple[int, int], default=(15, 12)
        Figure size
    cmap : str, default="vlag"
        Colormap for the heatmap
    row_cluster : bool, default=True
        Whether to cluster rows (samples)
    col_cluster : bool, default=True
        Whether to cluster columns (features)
    standard_scale : int, default=1
        Whether to standardize data 0: no scaling, 1: scale features, 2: scale samples
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create clustered heatmap
    g = sns.clustermap(X, method=method, metric=metric, 
                      figsize=figsize, cmap=cmap,
                      row_cluster=row_cluster, col_cluster=col_cluster,
                      standard_scale=standard_scale, 
                      yticklabels=False if X.shape[0] > 50 else True)
    
    # Set title
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return g.fig


def plot_dendrogram(X: pd.DataFrame,
                   title: str = "Feature Similarity Dendrogram",
                   figsize: Tuple[int, int] = (12, 8),
                   method: str = 'ward',
                   metric: str = 'euclidean',
                   color_threshold: Optional[float] = None,
                   orientation: str = 'top',
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dendrogram showing feature similarity.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features (features should be columns)
    title : str, default="Feature Similarity Dendrogram"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    color_threshold : Optional[float], default=None
        Threshold for coloring branches in dendrogram
    orientation : str, default='top'
        Orientation of the dendrogram ('top', 'bottom', 'left', 'right')
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transpose X to cluster features
    X_T = X.T
    
    # Calculate distance matrix
    distances = pdist(X_T, metric=metric)
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distances, method=method)
    
    # Create dendrogram
    hierarchy.dendrogram(
        linkage_matrix,
        labels=X.columns,
        ax=ax,
        orientation=orientation,
        leaf_rotation=90 if orientation in ['top', 'bottom'] else 0,
        color_threshold=color_threshold
    )
    
    # Set title
    ax.set_title(title)
    
    # Add metric and method to the plot
    plt.figtext(0.5, 0.01, f"Method: {method}, Metric: {metric}", 
               ha="center", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_distributions(X: pd.DataFrame,
                             features: Optional[List[str]] = None,
                             title: str = "Feature Distributions",
                             figsize: Tuple[int, int] = (15, 10),
                             n_cols: int = 3,
                             color: str = "#1f77b4",
                             kde: bool = True,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histograms of feature distributions.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    features : Optional[List[str]], default=None
        List of features to plot. If None, all features are plotted.
    title : str, default="Feature Distributions"
        Title of the plot
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    n_cols : int, default=3
        Number of columns in the grid
    color : str, default="#1f77b4"
        Color for the histograms
    kde : bool, default=True
        Whether to plot kernel density estimate
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Select features to plot
    if features is None:
        features = X.columns.tolist()
    
    # Calculate number of rows
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(X[feature], ax=ax, kde=kde, color=color)
            ax.set_title(feature)
            ax.set_xlabel('')
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_pca_components(X: pd.DataFrame,
                       n_components: int = 2,
                       title: str = "PCA Feature Contributions",
                       figsize: Tuple[int, int] = (12, 8),
                       cmap: str = "RdBu",
                       absolute_values: bool = False,
                       top_n_features: Optional[int] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature contributions to principal components.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    n_components : int, default=2
        Number of PCA components to plot
    title : str, default="PCA Feature Contributions"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    cmap : str, default="RdBu"
        Colormap for the heatmap
    absolute_values : bool, default=False
        Whether to plot absolute values of contributions
    top_n_features : Optional[int], default=None
        Number of top contributing features to show. If None, all features are shown.
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Perform PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Get component contributions
    components = pca.components_
    feature_names = X.columns
    
    # Create dataframe with component loadings
    loadings = pd.DataFrame(components.T, index=feature_names, 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Take absolute values if requested
    if absolute_values:
        loadings = loadings.abs()
    
    # Select top contributing features if specified
    if top_n_features is not None:
        # Calculate overall importance of features across components
        importance = loadings.abs().sum(axis=1)
        top_features = importance.nlargest(top_n_features).index
        loadings = loadings.loc[top_features]
    
    # Sort features by their contribution to first component
    loadings = loadings.sort_values(by='PC1', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(loadings, cmap=cmap, center=0 if not absolute_values else None,
               annot=True, fmt='.2f', ax=ax)
    
    # Set title
    ax.set_title(title)
    
    # Add variance explained as subtitle
    explained_var = pca.explained_variance_ratio_
    explained_var_text = ", ".join([f"PC{i+1}: {var:.1%}" for i, var in enumerate(explained_var)])
    ax.annotate(f"Explained variance: {explained_var_text}",
               xy=(0.5, -0.05), xycoords="axes fraction",
               ha="center", va="center",
               fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_mutual_info_matrix(X: pd.DataFrame,
                           y: np.ndarray,
                           title: str = "Feature-Target Mutual Information",
                           figsize: Tuple[int, int] = (10, 8),
                           top_n: Optional[int] = None,
                           group_features: bool = False,
                           feature_groups: Optional[Dict[str, List[str]]] = None,
                           color: str = "viridis",
                           show_values: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot mutual information between features and target variable.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing features
    y : np.ndarray
        Target variable
    title : str, default="Feature-Target Mutual Information"
        Title of the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    top_n : Optional[int], default=None
        Number of top features to display. If None, all features are shown.
    group_features : bool, default=False
        Whether to group features by prefix
    feature_groups : Optional[Dict[str, List[str]]], default=None
        Dictionary mapping group names to lists of feature names.
        Required if group_features is True and groups cannot be inferred.
    color : str, default="viridis"
        Color for the bars or colormap for grouped features
    show_values : bool, default=True
        Whether to show mutual information values on bars
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Calculate mutual information for each feature
    from sklearn.feature_selection import mutual_info_classif
    
    mi_scores = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi_scores})
    
    # Sort by mutual information
    mi_df = mi_df.sort_values('MI', ascending=False)
    
    # Select top features if specified
    if top_n is not None:
        mi_df = mi_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if group_features:
        # Infer feature groups if not provided
        if feature_groups is None:
            feature_groups = {}
            for col in X.columns:
                # Try to split on first underscore or dot
                for separator in ['_', '.']:
                    if separator in col:
                        prefix = col.split(separator)[0]
                        if prefix not in feature_groups:
                            feature_groups[prefix] = []
                        feature_groups[prefix].append(col)
                        break
            
            # Handle features that didn't match any pattern
            ungrouped = [col for col in X.columns if not any(col in group for group in feature_groups.values())]
            if ungrouped:
                feature_groups['Other'] = ungrouped
        
        # Calculate mean MI for each group
        group_mi = {}
        for group, features in feature_groups.items():
            # Filter to features in mi_df
            valid_features = [f for f in features if f in mi_df['Feature'].values]
            if valid_features:
                group_scores = mi_df[mi_df['Feature'].isin(valid_features)]['MI']
                group_mi[group] = group_scores.mean()
        
        # Create dataframe for plotting
        group_df = pd.DataFrame({
            'Group': list(group_mi.keys()),
            'MI': list(group_mi.values()),
            'Count': [len([f for f in feature_groups[g] if f in mi_df['Feature'].values]) 
                     for g in group_mi.keys()]
        }).sort_values('MI', ascending=False)
        
        # Create color palette
        colors = sns.color_palette(color, len(group_df))
        
        # Plot horizontal bars
        bars = ax.barh(group_df['Group'], group_df['MI'], color=colors)
        
        # Show values on bars if requested
        if show_values:
            for i, bar in enumerate(bars):
                width = bar.get_width()
                count = group_df.iloc[i]['Count']
                label_x_pos = width + 0.01 * max(group_df['MI'])
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f} ({count} features)', va='center')
        
        # Set labels and title
        ax.set_xlabel('Mean Mutual Information')
        ax.set_ylabel('Feature Group')
        ax.set_title(title)
    
    else:
        # Sort for better visualization (smallest to largest)
        mi_df = mi_df.sort_values('MI')
        
        # Plot horizontal bars
        bars = ax.barh(mi_df['Feature'], mi_df['MI'], color=color)
        
        # Show values on bars if requested
        if show_values:
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 * max(mi_df['MI'])
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', va='center')
        
        # Set labels and title
        ax.set_xlabel('Mutual Information')
        ax.set_ylabel('Feature')
        ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

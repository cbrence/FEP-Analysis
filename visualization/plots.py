"""
Basic visualization functions for FEP analysis.

This module provides standard plotting functions for visualizing data,
model performance, and clinical outcomes related to First Episode Psychosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import List, Dict, Tuple, Union, Optional, Any


def set_plotting_style(style: str = "whitegrid", context: str = "talk", palette: str = "colorblind") -> None:
    """
    Set consistent plotting style for all visualizations.
    
    Parameters
    ----------
    style : str, default="whitegrid"
        The seaborn style to use
    context : str, default="talk"
        The seaborn context to use
    palette : str, default="colorblind"
        The seaborn color palette to use
    """
    sns.set_style(style)
    sns.set_context(context)
    sns.set_palette(palette)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_feature_importance(feature_names: List[str], 
                            importance_values: List[float], 
                            title: str = "Feature Importance", 
                            top_n: Optional[int] = None, 
                            figsize: Tuple[int, int] = (10, 8),
                            color: str = "#1f77b4",
                            show_values: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importance_values : List[float]
        Importance values corresponding to features
    title : str, default="Feature Importance"
        Title of the plot
    top_n : Optional[int], default=None
        Number of top features to display, if None, all features are shown
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    color : str, default="#1f77b4"
        Bar color
    show_values : bool, default=True
        Whether to show importance values on bars
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
    
    # Select top_n features if specified
    if top_n is not None:
        df = df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    bars = ax.barh(df['Feature'], df['Importance'], color=color)
    
    # Show values on bars if requested
    if show_values:
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center')
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          class_names: Optional[List[str]] = None,
                          normalize: bool = False,
                          title: str = "Confusion Matrix",
                          cmap: str = "Blues",
                          figsize: Tuple[int, int] = (8, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : Optional[List[str]], default=None
        Names of classes, if None, class indices are used
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    title : str, default="Confusion Matrix"
        Title of the plot
    cmap : str, default="Blues"
        Colormap for the plot
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Set class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_roc_curves(y_true: np.ndarray, 
                    y_scores: Union[np.ndarray, Dict[str, np.ndarray]], 
                    title: str = "ROC Curves",
                    figsize: Tuple[int, int] = (8, 6),
                    colors: Optional[List[str]] = None,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for one or multiple models.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : Union[np.ndarray, Dict[str, np.ndarray]]
        Predicted probabilities or scores. If dict, keys are model names and values are scores.
    title : str, default="ROC Curves"
        Title of the plot
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    colors : Optional[List[str]], default=None
        Colors for the curves, if None, default colors are used
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    
    # Check if y_scores is a dictionary or a single array
    if isinstance(y_scores, dict):
        models = list(y_scores.keys())
        if colors is None:
            colors = [None] * len(models)
        
        # Plot ROC curve for each model
        for (model_name, scores), color in zip(y_scores.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {roc_auc:.3f})')
    else:
        # Plot ROC curve for a single model
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    
    # Set labels, title and legend
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, 
                               y_scores: Union[np.ndarray, Dict[str, np.ndarray]], 
                               title: str = "Precision-Recall Curve",
                               figsize: Tuple[int, int] = (8, 6),
                               colors: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curves for one or multiple models.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : Union[np.ndarray, Dict[str, np.ndarray]]
        Predicted probabilities or scores. If dict, keys are model names and values are scores.
    title : str, default="Precision-Recall Curve"
        Title of the plot
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    colors : Optional[List[str]], default=None
        Colors for the curves, if None, default colors are used
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    # Check if y_scores is a dictionary or a single array
    if isinstance(y_scores, dict):
        models = list(y_scores.keys())
        if colors is None:
            colors = [None] * len(models)
        
        # Plot P-R curve for each model
        for (model_name, scores), color in zip(y_scores.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=color, label=f'{model_name} (AUC = {pr_auc:.3f})')
    else:
        # Plot P-R curve for a single model
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    
    # Set labels, title and legend
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_metric_distribution(metric_values: List[float], 
                             metric_name: str = "Metric",
                             bins: int = 20,
                             color: str = "#1f77b4",
                             figsize: Tuple[int, int] = (8, 6),
                             show_mean: bool = True,
                             show_median: bool = True,
                             show_ci: bool = True,
                             confidence_level: float = 0.95,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of a metric (e.g., from cross-validation).
    
    Parameters
    ----------
    metric_values : List[float]
        List of metric values
    metric_name : str, default="Metric"
        Name of the metric
    bins : int, default=20
        Number of bins for histogram
    color : str, default="#1f77b4"
        Color for the histogram
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    show_mean : bool, default=True
        Whether to show mean line
    show_median : bool, default=True
        Whether to show median line
    show_ci : bool, default=True
        Whether to show confidence interval
    confidence_level : float, default=0.95
        Confidence level for CI
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    sns.histplot(metric_values, bins=bins, kde=True, color=color, ax=ax)
    
    # Calculate statistics
    mean_val = np.mean(metric_values)
    median_val = np.median(metric_values)
    
    # Calculate confidence interval
    if show_ci:
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        ci_lower = np.percentile(metric_values, lower_percentile)
        ci_upper = np.percentile(metric_values, upper_percentile)
        
        # Add CI to plot
        ax.axvline(x=ci_lower, color='red', linestyle='--', alpha=0.7, 
                   label=f'{confidence_level*100:.0f}% CI [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax.axvline(x=ci_upper, color='red', linestyle='--', alpha=0.7)
    
    # Add mean line
    if show_mean:
        ax.axvline(x=mean_val, color='green', linestyle='-', 
                   label=f'Mean: {mean_val:.3f}')
    
    # Add median line
    if show_median:
        ax.axvline(x=median_val, color='orange', linestyle='-', 
                   label=f'Median: {median_val:.3f}')
    
    # Set labels and title
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {metric_name}')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_correlation_matrix(data: pd.DataFrame,
                            method: str = 'pearson',
                            annot: bool = True,
                            cmap: str = 'coolwarm',
                            figsize: Tuple[int, int] = (10, 8),
                            title: str = "Feature Correlation Matrix",
                            mask_upper: bool = False,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix of features.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing features
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', 'spearman')
    annot : bool, default=True
        Whether to annotate cells
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    title : str, default="Feature Correlation Matrix"
        Title of the plot
    mask_upper : bool, default=False
        Whether to mask the upper triangle
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Calculate correlation matrix
    corr = data.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr, mask=mask, annot=annot, cmap=cmap, 
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={"size": 8})
    
    # Set title
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_temporal_trends(data: pd.DataFrame,
                         time_column: str,
                         value_columns: Union[str, List[str]],
                         title: str = "Temporal Trends",
                         figsize: Tuple[int, int] = (12, 6),
                         colors: Optional[List[str]] = None,
                         markers: Optional[List[str]] = None,
                         add_rolling_avg: bool = False,
                         window: int = 3,
                         date_format: str = '%Y-%m-%d',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot temporal trends of one or more variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing time series data
    time_column : str
        Name of the column containing time points
    value_columns : Union[str, List[str]]
        Name(s) of column(s) containing values to plot
    title : str, default="Temporal Trends"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    colors : Optional[List[str]], default=None
        Colors for the lines
    markers : Optional[List[str]], default=None
        Markers for the data points
    add_rolling_avg : bool, default=False
        Whether to add rolling average
    window : int, default=3
        Window size for rolling average
    date_format : str, default='%Y-%m-%d'
        Format for date ticks if time_column contains dates
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Ensure value_columns is a list
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    # Sort data by time if possible
    if pd.api.types.is_datetime64_any_dtype(data[time_column]):
        data = data.sort_values(by=time_column)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default colors and markers if not provided
    if colors is None:
        colors = [None] * len(value_columns)
    if markers is None:
        markers = [None] * len(value_columns)
    
    # Plot each value column
    for column, color, marker in zip(value_columns, colors, markers):
        ax.plot(data[time_column], data[column], label=column, 
                color=color, marker=marker)
        
        # Add rolling average if requested
        if add_rolling_avg:
            rolling_avg = data[column].rolling(window=window, center=True).mean()
            ax.plot(data[time_column], rolling_avg, 
                    color=color, linestyle='--', alpha=0.7,
                    label=f'{column} ({window}-point rolling avg)')
    
    # Format x-axis if time_column contains dates
    if pd.api.types.is_datetime64_any_dtype(data[time_column]):
        plt.gcf().autofmt_xdate()
        date_formatter = plt.matplotlib.dates.DateFormatter(date_format)
        ax.xaxis.set_major_formatter(date_formatter)
    
    # Set labels and title
    ax.set_xlabel(time_column)
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_calibration_curve(y_true: np.ndarray,
                          y_prob: Union[np.ndarray, Dict[str, np.ndarray]],
                          n_bins: int = 10,
                          title: str = "Calibration Curve",
                          figsize: Tuple[int, int] = (8, 6),
                          colors: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot calibration curve for binary classification models.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : Union[np.ndarray, Dict[str, np.ndarray]]
        Predicted probabilities. If dict, keys are model names and values are probabilities.
    n_bins : int, default=10
        Number of bins to use for calibration curve
    title : str, default="Calibration Curve"
        Title of the plot
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    colors : Optional[List[str]], default=None
        Colors for the curves
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Helper function to calculate calibration curve
    def calculate_calibration(y_true, y_prob, n_bins):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.minimum(bin_indices, n_bins - 1)  # Ensure within range
        
        bin_sums = np.bincount(bin_indices, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
        
        nonzero = bin_sums != 0
        prob_true = np.zeros(n_bins)
        prob_pred = np.zeros(n_bins)
        
        prob_true[nonzero] = bin_true[nonzero] / bin_sums[nonzero]
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                prob_pred[i] = np.mean(y_prob[bin_mask])
        
        return prob_true, prob_pred
    
    # Check if y_prob is a dictionary or a single array
    if isinstance(y_prob, dict):
        models = list(y_prob.keys())
        if colors is None:
            colors = [None] * len(models)
        
        # Plot calibration curve for each model
        for (model_name, probs), color in zip(y_prob.items(), colors):
            prob_true, prob_pred = calculate_calibration(y_true, probs, n_bins)
            ax.plot(prob_pred, prob_true, "s-", color=color, label=model_name)
    else:
        # Plot calibration curve for a single model
        prob_true, prob_pred = calculate_calibration(y_true, y_prob, n_bins)
        ax.plot(prob_pred, prob_true, "s-", label="Model")
    
    # Set labels, title and legend
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (Empirical probability)")
    ax.set_title(title)
    ax.legend(loc="best")
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_threshold_analysis(thresholds: List[float],
                           metrics: Dict[str, List[float]],
                           selected_threshold: Optional[float] = None,
                           title: str = "Threshold Analysis",
                           figsize: Tuple[int, int] = (10, 6),
                           colors: Optional[Dict[str, str]] = None,
                           log_scale: bool = False,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot metrics as a function of classification threshold.
    
    Parameters
    ----------
    thresholds : List[float]
        List of threshold values
    metrics : Dict[str, List[float]]
        Dictionary with metric names as keys and lists of values as values
    selected_threshold : Optional[float], default=None
        Threshold value to highlight, if None, no threshold is highlighted
    title : str, default="Threshold Analysis"
        Title of the plot
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    colors : Optional[Dict[str, str]], default=None
        Dictionary with metric names as keys and colors as values
    log_scale : bool, default=False
        Whether to use log scale for x-axis
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default colors if not provided
    if colors is None:
        colors = {}
    
    # Plot each metric
    for metric_name, metric_values in metrics.items():
        color = colors.get(metric_name, None)
        ax.plot(thresholds, metric_values, label=metric_name, color=color)
    
    # Highlight selected threshold if provided
    if selected_threshold is not None:
        ax.axvline(x=selected_threshold, color='red', linestyle='--', 
                  label=f'Selected threshold: {selected_threshold:.3f}')
    
    # Set x-axis to log scale if requested
    if log_scale:
        ax.set_xscale('log')
    
    # Set labels, title and legend
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

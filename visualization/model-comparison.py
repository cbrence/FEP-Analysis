"""
Model comparison visualization functions for FEP analysis.

This module provides specialized functions for visualizing model comparisons
and threshold effects in First Episode Psychosis prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
from ..evaluation.metrics import clinical_utility_score
import scipy.stats as stats


def plot_model_comparison(models_dict: Dict[str, Dict[str, float]],
                          metrics: Optional[List[str]] = None,
                          title: str = "Model Performance Comparison",
                          figsize: Tuple[int, int] = (12, 8),
                          color_palette: str = "Set2",
                          sort_by: Optional[str] = None,
                          ascending: bool = False,
                          error_bars: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple models across different metrics.
    
    Parameters
    ----------
    models_dict : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and dictionaries of metrics as values.
        Example: {'Model1': {'AUC': 0.85, 'F1': 0.76}, 'Model2': {'AUC': 0.82, 'F1': 0.78}}
    metrics : Optional[List[str]], default=None
        List of metrics to compare. If None, all metrics in the first model are used.
    title : str, default="Model Performance Comparison"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    color_palette : str, default="Set2"
        Color palette for the models
    sort_by : Optional[str], default=None
        Metric to sort models by. If None, original order is maintained.
    ascending : bool, default=False
        Whether to sort in ascending order
    error_bars : Optional[Dict[str, Dict[str, Tuple[float, float]]]], default=None
        Dictionary with model names as keys and dictionaries of metrics with error ranges.
        Example: {'Model1': {'AUC': (0.02, 0.02), 'F1': (0.03, 0.03)}}
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Convert to DataFrame for easier manipulation
    data = []
    for model_name, metrics_dict in models_dict.items():
        for metric_name, value in metrics_dict.items():
            data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value
            })
    df = pd.DataFrame(data)
    
    # Filter metrics if specified
    if metrics is not None:
        df = df[df['Metric'].isin(metrics)]
    
    # Sort if specified
    if sort_by is not None:
        # Create a mapping of model to sort value
        sort_values = {model: metrics_dict.get(sort_by, 0) 
                      for model, metrics_dict in models_dict.items()}
        
        # Create a sorted list of models
        sorted_models = sorted(sort_values.keys(), 
                              key=lambda x: sort_values[x],
                              reverse=not ascending)
        
        # Ensure consistent order by creating a categorical type
        df['Model'] = pd.Categorical(df['Model'], categories=sorted_models, ordered=True)
        df = df.sort_values('Model')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grouped bar chart
    sns.barplot(data=df, x='Metric', y='Value', hue='Model', palette=color_palette, ax=ax)
    
    # Add error bars if provided
    if error_bars is not None:
        # Find unique metrics and models in the correct order
        metrics_in_order = df['Metric'].unique()
        models_in_order = df['Model'].cat.categories if hasattr(df['Model'], 'cat') else df['Model'].unique()
        
        # Calculate bar positions
        width = 0.8 / len(models_in_order)  # width of one bar
        x_positions = []
        
        for i, metric in enumerate(metrics_in_order):
            for j, model in enumerate(models_in_order):
                # Calculate position of this bar
                pos = i + (-0.4 + width/2 + j*width)
                x_positions.append((model, metric, pos))
        
        # Add error bars
        for model, metric, pos in x_positions:
            if model in error_bars and metric in error_bars[model]:
                # Get value for this bar
                value = models_dict[model][metric]
                
                # Get error values
                yerr_neg, yerr_pos = error_bars[model][metric]
                
                # Plot error bar
                ax.errorbar(pos, value, yerr=[[yerr_neg], [yerr_pos]], 
                           fmt='none', color='black', capsize=5)
    
    # Set labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Adjust legend position
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_threshold_effects(thresholds: List[float],
                          metrics_by_model: Dict[str, Dict[str, List[float]]],
                          selected_thresholds: Optional[Dict[str, float]] = None,
                          title: str = "Threshold Effects on Model Performance",
                          figsize: Tuple[int, int] = (12, 8),
                          metrics_to_plot: Optional[List[str]] = None,
                          color_palette: str = "Set2",
                          linestyles: Optional[Dict[str, str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the effect of threshold changes on multiple metrics for different models.
    
    Parameters
    ----------
    thresholds : List[float]
        List of threshold values
    metrics_by_model : Dict[str, Dict[str, List[float]]]
        Dictionary with model names as keys and dictionaries of metrics as values.
        Each metric has a list of values corresponding to thresholds.
        Example: {'Model1': {'precision': [...], 'recall': [...]}}
    selected_thresholds : Optional[Dict[str, float]], default=None
        Dictionary with model names as keys and selected threshold values.
    title : str, default="Threshold Effects on Model Performance"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    metrics_to_plot : Optional[List[str]], default=None
        List of metrics to plot. If None, all metrics are plotted.
    color_palette : str, default="Set2"
        Color palette for the models
    linestyles : Optional[Dict[str, str]], default=None
        Dictionary with metric names as keys and linestyles as values.
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Determine metrics to plot
    if metrics_to_plot is None:
        # Find all metrics across all models
        metrics_to_plot = set()
        for model_metrics in metrics_by_model.values():
            metrics_to_plot.update(model_metrics.keys())
        metrics_to_plot = sorted(metrics_to_plot)
    
    # Determine number of subplots
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    
    # Flatten axes for easier indexing if multiple rows/columns
    if n_metrics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Set default linestyles if not provided
    if linestyles is None:
        linestyles = {metric: '-' for metric in metrics_to_plot}
    
    # Get color palette
    colors = sns.color_palette(color_palette, len(metrics_by_model))
    model_colors = {model: colors[i] for i, model in enumerate(metrics_by_model.keys())}
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Plot each model's values for this metric
        for j, (model, metrics) in enumerate(metrics_by_model.items()):
            if metric in metrics:
                # Plot line
                ax.plot(thresholds, metrics[metric], label=model if i == 0 else "",
                       color=model_colors[model], linestyle=linestyles.get(metric, '-'))
                
                # Add selected threshold if provided
                if selected_thresholds is not None and model in selected_thresholds:
                    thresh = selected_thresholds[model]
                    
                    # Find the closest threshold value index
                    idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - thresh))
                    value = metrics[metric][idx]
                    
                    # Plot point
                    ax.plot([thresh], [value], 'o', color=model_colors[model], 
                           markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        # Set labels and title
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Add single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(metrics_by_model)), title="Models")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_model_radar(models_dict: Dict[str, Dict[str, float]],
                    metrics: Optional[List[str]] = None,
                    title: str = "Model Comparison Radar Chart",
                    figsize: Tuple[int, int] = (10, 8),
                    min_values: Optional[Dict[str, float]] = None,
                    max_values: Optional[Dict[str, float]] = None,
                    color_palette: str = "Set2",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a radar chart (spider plot) to compare models across multiple metrics.
    
    Parameters
    ----------
    models_dict : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and dictionaries of metrics as values.
    metrics : Optional[List[str]], default=None
        List of metrics to compare. If None, all metrics in the first model are used.
    title : str, default="Model Comparison Radar Chart"
        Title of the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    min_values : Optional[Dict[str, float]], default=None
        Dictionary with minimum values for each metric for scaling.
        If None, minimum values are determined from the data.
    max_values : Optional[Dict[str, float]], default=None
        Dictionary with maximum values for each metric for scaling.
        If None, maximum values are determined from the data.
    color_palette : str, default="Set2"
        Color palette for the models
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Determine metrics to use
    if metrics is None:
        metrics = list(next(iter(models_dict.values())).keys())
    
    # Filter models_dict to only include specified metrics
    filtered_models = {model: {k: v for k, v in metrics_dict.items() if k in metrics}
                     for model, metrics_dict in models_dict.items()}
    
    # Determine min and max values for scaling
    if min_values is None:
        min_values = {metric: min(model[metric] for model in filtered_models.values() if metric in model)
                     for metric in metrics}
    
    if max_values is None:
        max_values = {metric: max(model[metric] for model in filtered_models.values() if metric in model)
                     for metric in metrics}
    
    # Function to scale values between 0 and 1
    def scale_value(value, metric):
        min_val = min_values[metric]
        max_val = max_values[metric]
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        return (value - min_val) / (max_val - min_val)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    
    # Set number of angles based on number of metrics
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    
    # Make the plot circular by appending the first angle to the end
    angles += angles[:1]
    
    # Get color palette
    colors = sns.color_palette(color_palette, len(filtered_models))
    
    # Plot each model
    for i, (model_name, model_metrics) in enumerate(filtered_models.items()):
        # Scale values between 0 and 1
        values = [scale_value(model_metrics.get(metric, 0), metric) for metric in metrics]
        
        # Close the polygon by appending the first value
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add value labels to the chart
    ax.set_rlabel_position(0)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    ax.set_title(title, y=1.1)
    
    # Add min/max labels for each metric
    for i, metric in enumerate(metrics):
        angle = angles[i]
        min_val = min_values[metric]
        max_val = max_values[metric]
        
        # Add text annotations at 0.0 and 1.0 positions
        ax.annotate(f"{min_val:.2f}",
                   xy=(angle, 0.1),
                   xytext=(angle, 0.1),
                   color='gray',
                   weight='light',
                   size=8,
                   ha="center")
        
        ax.annotate(f"{max_val:.2f}",
                   xy=(angle, 0.9),
                   xytext=(angle, 0.9),
                   color='gray',
                   weight='bold',
                   size=8,
                   ha="center")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_paired_metric_comparison(models_dict: Dict[str, Dict[str, float]],
                                 metric_pairs: List[Tuple[str, str]],
                                 title: str = "Trade-off Analysis",
                                 figsize: Tuple[int, int] = (12, 10),
                                 annotate: bool = True,
                                 color_palette: str = "Set2",
                                 ideal_regions: Optional[List[Tuple[str, str, str]]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create scatter plots for pairs of metrics to visualize trade-offs.
    
    Parameters
    ----------
    models_dict : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and dictionaries of metrics as values.
    metric_pairs : List[Tuple[str, str]]
        List of metric pairs to plot.
        Example: [('precision', 'recall'), ('auc', 'clinical_utility')]
    title : str, default="Trade-off Analysis"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    annotate : bool, default=True
        Whether to annotate points with model names
    color_palette : str, default="Set2"
        Color palette for the scatter points
    ideal_regions : Optional[List[Tuple[str, str, str]]], default=None
        List of tuples specifying ideal regions for each metric pair.
        Each tuple contains (x_direction, y_direction, region_label).
        Direction can be 'higher' or 'lower'.
        Example: [('higher', 'higher', 'Ideal Region')]
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Determine number of subplots
    n_pairs = len(metric_pairs)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier indexing if multiple rows/columns
    if n_pairs > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Get color palette
    colors = sns.color_palette(color_palette, len(models_dict))
    model_colors = {model: colors[i] for i, model in enumerate(models_dict.keys())}
    
    # Plot each metric pair
    for i, (metric_x, metric_y) in enumerate(metric_pairs):
        ax = axes[i]
        
        # Extract values for this pair
        x_values = []
        y_values = []
        models = []
        colors_list = []
        
        for model, metrics in models_dict.items():
            if metric_x in metrics and metric_y in metrics:
                x_values.append(metrics[metric_x])
                y_values.append(metrics[metric_y])
                models.append(model)
                colors_list.append(model_colors[model])
        
        # Create scatter plot
        scatter = ax.scatter(x_values, y_values, s=100, alpha=0.7, c=colors_list)
        
        # Annotate points if requested
        if annotate:
            for j, model in enumerate(models):
                ax.annotate(model, (x_values[j], y_values[j]),
                          xytext=(10, 5), textcoords='offset points',
                          fontsize=8, weight='bold')
        
        # Add ideal region shading if specified
        if ideal_regions is not None and i < len(ideal_regions):
            x_dir, y_dir, region_label = ideal_regions[i]
            
            # Determine the corner point to use as center for the shaded region
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            if x_dir == 'higher' and y_dir == 'higher':
                # Shade top-right region
                corner_x, corner_y = x_max, y_max
                rect_x, rect_y = x_min, y_min
                width, height = x_max - x_min, y_max - y_min
            elif x_dir == 'higher' and y_dir == 'lower':
                # Shade bottom-right region
                corner_x, corner_y = x_max, y_min
                rect_x, rect_y = x_min, y_min
                width, height = x_max - x_min, y_max - y_min
            elif x_dir == 'lower' and y_dir == 'higher':
                # Shade top-left region
                corner_x, corner_y = x_min, y_max
                rect_x, rect_y = x_min, y_min
                width, height = x_max - x_min, y_max - y_min
            else:  # 'lower' and 'lower'
                # Shade bottom-left region
                corner_x, corner_y = x_min, y_min
                rect_x, rect_y = x_min, y_min
                width, height = x_max - x_min, y_max - y_min
            
            # Add semi-transparent rectangle
            import matplotlib.patches as patches
            rect = patches.Rectangle((rect_x, rect_y), width, height, linewidth=0, 
                                    alpha=0.1, color='green')
            ax.add_patch(rect)
            
            # Add label to the ideal region
            ax.annotate(region_label, (corner_x, corner_y),
                      xytext=(-20 if x_dir == 'lower' else 20, 
                              -20 if y_dir == 'lower' else 20),
                      textcoords='offset points',
                      color='green', fontsize=10,
                      ha='right' if x_dir == 'lower' else 'left',
                      va='bottom' if y_dir == 'lower' else 'top')
        
        # Set labels
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())
        ax.set_title(f"{metric_x.capitalize()} vs {metric_y.capitalize()}")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes)):
        axes[i].set_visible(False)
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, alpha=0.7)
              for model, color in model_colors.items()]
    labels = list(model_colors.keys())
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(models_dict)), title="Models")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_model_ranking(models_dict: Dict[str, Dict[str, float]],
                      metrics: List[str],
                      weights: Optional[Dict[str, float]] = None,
                      title: str = "Model Ranking",
                      higher_is_better: Dict[str, bool] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      color_palette: str = "Set2",
                      show_scores: bool = True,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a horizontal bar chart showing model ranking based on multiple metrics.
    
    Parameters
    ----------
    models_dict : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and dictionaries of metrics as values.
    metrics : List[str]
        List of metrics to use for ranking
    weights : Optional[Dict[str, float]], default=None
        Dictionary with metric names as keys and weights as values.
        If None, all metrics are weighted equally.
    title : str, default="Model Ranking"
        Title of the plot
    higher_is_better : Dict[str, bool], default=None
        Dictionary specifying whether higher values are better for each metric.
        If None, higher is assumed better for all metrics.
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    color_palette : str, default="Set2"
        Color palette for the bars
    show_scores : bool, default=True
        Whether to show the scores on the bars
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set default values
    if weights is None:
        weights = {metric: 1.0 / len(metrics) for metric in metrics}
    
    if higher_is_better is None:
        higher_is_better = {metric: True for metric in metrics}
    
    # Normalize metrics to 0-1 scale
    normalized_models = {}
    
    for metric in metrics:
        # Get all values for this metric
        values = [models_dict[model].get(metric, 0) for model in models_dict]
        min_val = min(values)
        max_val = max(values)
        
        # Skip if min and max are the same to avoid division by zero
        if min_val == max_val:
            for model in models_dict:
                if model not in normalized_models:
                    normalized_models[model] = {}
                normalized_models[model][metric] = 0.5
            continue
        
        # Normalize each model's value
        for model in models_dict:
            if model not in normalized_models:
                normalized_models[model] = {}
            
            value = models_dict[model].get(metric, 0)
            norm_value = (value - min_val) / (max_val - min_val)
            
            # Invert if lower is better
            if not higher_is_better.get(metric, True):
                norm_value = 1 - norm_value
            
            normalized_models[model][metric] = norm_value
    
    # Calculate weighted scores
    scores = {}
    for model, norm_metrics in normalized_models.items():
        score = sum(norm_metrics.get(metric, 0) * weights.get(metric, 0) for metric in metrics)
        scores[model] = score
    
    # Sort models by score
    sorted_models = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color palette
    colors = sns.color_palette(color_palette, len(sorted_models))
    
    # Plot horizontal bars
    y_pos = range(len(sorted_models))
    ax.barh(y_pos, [scores[model] for model in sorted_models], color=colors)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel('Score')
    ax.set_title(title)
    
    # Add scores on bars if requested
    if show_scores:
        for i, model in enumerate(sorted_models):
            ax.text(scores[model] + 0.01, i, f'{scores[model]:.3f}', 
                   va='center', color='black')
    
    # Add a description of the metrics used
    metric_desc = ', '.join([f"{metric} ({weights[metric]:.0%})" for metric in metrics])
    plt.figtext(0.5, 0.01, f"Metrics used (with weights): {metric_desc}", 
               ha="center", fontsize=8, wrap=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

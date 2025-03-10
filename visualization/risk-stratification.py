"""
Risk stratification visualization functions for FEP analysis.

This module provides specialized functions for visualizing risk stratification
and risk distribution over time for First Episode Psychosis patients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


def plot_risk_distribution(risk_scores: Union[List[float], np.ndarray],
                          class_labels: Optional[Union[List[int], np.ndarray]] = None,
                          thresholds: Optional[List[float]] = None,
                          title: str = "Risk Score Distribution",
                          bins: int = 30,
                          figsize: Tuple[int, int] = (10, 6),
                          color_positive: str = "#ff7f0e",
                          color_negative: str = "#1f77b4",
                          show_metrics: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of risk scores with optional class separation and thresholds.
    
    Parameters
    ----------
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores or probabilities
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        True class labels (1 for positive, 0 for negative)
    thresholds : Optional[List[float]], default=None
        List of threshold values to show as vertical lines
    title : str, default="Risk Score Distribution"
        Title of the plot
    bins : int, default=30
        Number of bins for the histogram
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    color_positive : str, default="#ff7f0e"
        Color for positive class scores
    color_negative : str, default="#1f77b4"
        Color for negative class scores
    show_metrics : bool, default=True
        Whether to show metrics like separation statistics
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Ensure numpy arrays
    risk_scores = np.array(risk_scores)
    if class_labels is not None:
        class_labels = np.array(class_labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    if class_labels is not None:
        # Separate scores by class
        pos_scores = risk_scores[class_labels == 1]
        neg_scores = risk_scores[class_labels == 0]
        
        # Create histograms for each class
        sns.histplot(pos_scores, bins=bins, color=color_positive, 
                    alpha=0.6, label="Positive Class", ax=ax)
        sns.histplot(neg_scores, bins=bins, color=color_negative, 
                    alpha=0.6, label="Negative Class", ax=ax)
        
        # Add legend
        ax.legend()
        
        # Calculate and show separation metrics if requested
        if show_metrics and len(pos_scores) > 0 and len(neg_scores) > 0:
            # Calculate mean and standard deviation for each class
            pos_mean = np.mean(pos_scores)
            neg_mean = np.mean(neg_scores)
            pos_std = np.std(pos_scores)
            neg_std = np.std(neg_scores)
            
            # Calculate separation metrics
            separation = (pos_mean - neg_mean) / ((pos_std + neg_std) / 2)
            
            # Add metrics to the plot
            ax.text(0.02, 0.95, 
                   f"Pos Mean: {pos_mean:.3f}, Neg Mean: {neg_mean:.3f}\n"
                   f"Pos Std: {pos_std:.3f}, Neg Std: {neg_std:.3f}\n"
                   f"Separation: {separation:.3f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        # Create a single histogram
        sns.histplot(risk_scores, bins=bins, color="#1f77b4", ax=ax)
    
    # Add threshold lines if provided
    if thresholds is not None:
        ymax = ax.get_ylim()[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
        
        for i, threshold in enumerate(thresholds):
            ax.axvline(x=threshold, color=colors[i], linestyle='--', 
                      label=f"Threshold {i+1}: {threshold:.3f}")
        
        # Add legend for thresholds
        handles, labels = ax.get_legend_handles_labels()
        class_handles = handles[:2] if class_labels is not None else []
        threshold_handles = handles[2:] if class_labels is not None else handles
        
        if class_labels is not None:
            ax.legend(class_handles + threshold_handles, 
                     labels[:2] + [f"Threshold {i+1}: {t:.3f}" for i, t in enumerate(thresholds)],
                     loc='best')
        else:
            ax.legend([f"Threshold {i+1}: {t:.3f}" for i, t in enumerate(thresholds)],
                     loc='best')
    
    # Set labels and title
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_risk_timeline(timestamps: Union[List, np.ndarray],
                      risk_scores: Union[List[float], np.ndarray],
                      patient_ids: Optional[Union[List, np.ndarray]] = None,
                      events: Optional[List[Dict[str, Any]]] = None,
                      title: str = "Risk Score Timeline",
                      figsize: Tuple[int, int] = (12, 6),
                      color_map: str = "viridis",
                      threshold: Optional[float] = None,
                      highlight_high_risk: bool = True,
                      rolling_window: Optional[int] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot timeline of risk scores with optional event markers.
    
    Parameters
    ----------
    timestamps : Union[List, np.ndarray]
        List of time points (datetime or numeric)
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores at each time point
    patient_ids : Optional[Union[List, np.ndarray]], default=None
        Patient identifiers if multiple patients
    events : Optional[List[Dict[str, Any]]], default=None
        List of events to mark on the timeline.
        Each event should be a dict with at least 'time' and 'label' keys.
        Optional 'color' and 'marker' keys can be included.
    title : str, default="Risk Score Timeline"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    color_map : str, default="viridis"
        Colormap for multiple patients
    threshold : Optional[float], default=None
        Risk threshold to show as horizontal line
    highlight_high_risk : bool, default=True
        Whether to highlight periods above threshold
    rolling_window : Optional[int], default=None
        Window size for rolling average, if None, no smoothing is applied
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Ensure arrays
    timestamps = np.array(timestamps)
    risk_scores = np.array(risk_scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have multiple patients
    if patient_ids is not None:
        patient_ids = np.array(patient_ids)
        unique_patients = np.unique(patient_ids)
        
        # Create colormap for patients
        norm = Normalize(vmin=0, vmax=len(unique_patients)-1)
        cmap = plt.cm.get_cmap(color_map)
        
        # Plot each patient separately
        for i, patient in enumerate(unique_patients):
            mask = patient_ids == patient
            patient_times = timestamps[mask]
            patient_scores = risk_scores[mask]
            
            # Sort by time if necessary
            if len(patient_times) > 0:
                sort_idx = np.argsort(patient_times)
                patient_times = patient_times[sort_idx]
                patient_scores = patient_scores[sort_idx]
            
            # Apply rolling average if requested
            if rolling_window is not None and len(patient_scores) >= rolling_window:
                # Calculate rolling mean manually to handle datetime x-axis
                smooth_scores = np.convolve(patient_scores, 
                                          np.ones(rolling_window)/rolling_window, 
                                          mode='valid')
                # Adjust times to match the convolved data
                smooth_times = patient_times[rolling_window-1:]
                
                # Plot raw data as scatter and smoothed as line
            ax.scatter(times, risk_scores, color=cmap(i/len(unique_patients)), 
                      alpha=0.3, s=20)
            ax.plot(smooth_times, smooth_scores, color=cmap(i/len(unique_patients)), 
                   alpha=0.8, linewidth=2)
        else:
            # Plot without smoothing
            ax.plot(times, risk_scores, color=cmap(i/len(unique_patients)), 
                   marker='o', alpha=0.8, linewidth=2)
        
        # Add threshold line if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
            
            # Highlight high-risk periods if requested
            if highlight_events:
                # Find periods above threshold
                above_threshold = risk_scores > threshold
                
                if np.any(above_threshold):
                    # Create masked array for highlighting
                    masked_times = np.ma.masked_array(times, mask=~above_threshold)
                    masked_scores = np.ma.masked_array(risk_scores, mask=~above_threshold)
                    
                    # Highlight points
                    ax.plot(masked_times, masked_scores, 'o', color='red', 
                           markersize=8, alpha=0.6)
        
        # Highlight clinical events if available
        if event_col is not None and highlight_events:
            events = patient_df[patient_df[event_col] == 1]
            
            if len(events) > 0:
                event_times = events[time_col].values
                event_risks = events[risk_col].values
                
                # Add markers for events
                ax.plot(event_times, event_risks, 'D', color='black',
                       markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        # Set title with patient ID
        ax.set_title(f"Patient {patient_id}")
        
        # Format x-axis if times are datetime
        if pd.api.types.is_datetime64_any_dtype(patient_df[time_col]):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(unique_patients), len(axes)):
        axes[i].set_visible(False)
    
    # Add common labels
    fig.text(0.5, 0.04, "Time", ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, "Risk Score", ha='center', va='center', rotation='vertical', fontsize=12)
    
    # Create common legend
    legend_elements = []
    
    # Add line for risk score
    legend_elements.append(
        plt.Line2D([0], [0], color=cmap(0.5), marker='o', linestyle='-',
                 markersize=5, label="Risk Score")
    )
    
    # Add threshold line if provided
    if threshold is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linestyle='--',
                     markersize=0, label=f"Threshold ({threshold:.2f})")
        )
    
    # Add event marker if available
    if event_col is not None and highlight_events:
        legend_elements.append(
            plt.Line2D([0], [0], marker='D', color='white',
                     markerfacecolor='black', markersize=8,
                     markeredgecolor='red', markeredgewidth=2,
                     linestyle='None', label="Clinical Event")
        )
    
    # Add legend to figure
    fig.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.95])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_risk_composition(feature_contributions: pd.DataFrame,
                        risk_scores: Union[List[float], np.ndarray],
                        patient_ids: Optional[Union[List, np.ndarray]] = None,
                        title: str = "Risk Score Composition",
                        figsize: Tuple[int, int] = (12, 8),
                        n_features: int = 5,
                        sort_by: str = "mean",
                        normalize: bool = False,
                        color_map: str = "Set2",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot composition of risk scores by feature contributions.
    
    Parameters
    ----------
    feature_contributions : pd.DataFrame
        DataFrame with feature contributions (features as columns)
    risk_scores : Union[List[float], np.ndarray]
        Overall risk scores
    patient_ids : Optional[Union[List, np.ndarray]], default=None
        Patient identifiers if multiple patients
    title : str, default="Risk Score Composition"
        Title of the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    n_features : int, default=5
        Number of top features to show individually
    sort_by : str, default="mean"
        How to sort features ('mean', 'max', 'variance')
    normalize : bool, default=False
        Whether to normalize contributions to sum to 1
    color_map : str, default="Set2"
        Colormap for feature contributions
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Ensure feature_contributions is a DataFrame
    if not isinstance(feature_contributions, pd.DataFrame):
        raise ValueError("feature_contributions must be a pandas DataFrame")
    
    # Select top features
    if sort_by == "mean":
        feature_means = feature_contributions.abs().mean().sort_values(ascending=False)
        top_features = feature_means.index[:n_features].tolist()
    elif sort_by == "max":
        feature_max = feature_contributions.abs().max().sort_values(ascending=False)
        top_features = feature_max.index[:n_features].tolist()
    elif sort_by == "variance":
        feature_var = feature_contributions.var().sort_values(ascending=False)
        top_features = feature_var.index[:n_features].tolist()
    else:
        raise ValueError("sort_by must be 'mean', 'max', or 'variance'")
    
    # Group remaining features
    other_features = [col for col in feature_contributions.columns if col not in top_features]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={'width_ratios': [3, 1]})
    
    # Handle multiple patients or single patient
    if patient_ids is not None:
        patient_ids = np.array(patient_ids)
        unique_patients = np.unique(patient_ids)
        
        # Sort patients by risk score
        patient_mean_risks = {}
        for patient in unique_patients:
            mask = patient_ids == patient
            patient_mean_risks[patient] = np.mean(risk_scores[mask])
        
        sorted_patients = sorted(unique_patients, key=lambda p: patient_mean_risks[p], reverse=True)
        
        # Create a new DataFrame with top features and "Other" column
        plot_data = pd.DataFrame(index=sorted_patients)
        
        # Add top features
        for feature in top_features:
            plot_data[feature] = [np.mean(feature_contributions.loc[patient_ids == patient, feature]) 
                                for patient in sorted_patients]
        
        # Add "Other" as sum of remaining features
        plot_data["Other"] = [np.mean(feature_contributions.loc[patient_ids == patient, other_features].sum(axis=1)) 
                            for patient in sorted_patients]
        
        # Normalize if requested
        if normalize:
            # Ensure each row sums to 1
            row_sums = plot_data.abs().sum(axis=1)
            plot_data = plot_data.div(row_sums, axis=0)
        
        # Create stacked bar chart
        plot_data.plot(kind='barh', stacked=True, ax=ax1, colormap=color_map)
        
        # Set labels
        ax1.set_xlabel("Contribution to Risk Score")
        ax1.set_ylabel("Patient")
        ax1.set_title("Feature Contributions by Patient")
        
        # Create second plot with average contributions
        mean_contributions = plot_data.mean().sort_values(ascending=True)
        mean_contributions.plot(kind='barh', ax=ax2, colormap=color_map)
        
        ax2.set_xlabel("Mean Contribution")
        ax2.set_title("Average Feature Contributions")
    
    else:
        # Single patient or aggregated view
        
        # Create a new DataFrame with top features and "Other" column
        plot_data = pd.DataFrame(index=range(len(risk_scores)))
        
        # Add top features
        for feature in top_features:
            plot_data[feature] = feature_contributions[feature].values
        
        # Add "Other" as sum of remaining features
        plot_data["Other"] = feature_contributions[other_features].sum(axis=1).values
        
        # Normalize if requested
        if normalize:
            # Ensure each row sums to 1
            row_sums = plot_data.abs().sum(axis=1)
            plot_data = plot_data.div(row_sums, axis=0)
        
        # Sort by risk score
        plot_data['risk'] = risk_scores
        plot_data = plot_data.sort_values('risk', ascending=False)
        plot_data = plot_data.drop('risk', axis=1)
        
        # Limit to top 30 samples for readability
        if len(plot_data) > 30:
            # Take 10 from top, 10 from middle, 10 from bottom
            indices = list(range(10))
            middle_start = len(plot_data) // 2 - 5
            indices += list(range(middle_start, middle_start + 10))
            indices += list(range(len(plot_data) - 10, len(plot_data)))
            plot_data = plot_data.iloc[indices]
        
        # Create stacked bar chart
        plot_data.plot(kind='barh', stacked=True, ax=ax1, colormap=color_map)
        
        # Set labels
        ax1.set_xlabel("Contribution to Risk Score")
        ax1.set_ylabel("Sample")
        ax1.set_title("Feature Contributions by Sample")
        
        # Create second plot with average contributions
        mean_contributions = plot_data.mean().sort_values(ascending=True)
        mean_contributions.plot(kind='barh', ax=ax2, colormap=color_map)
        
        ax2.set_xlabel("Mean Contribution")
        ax2.set_title("Average Feature Contributions")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig
                ax.scatter(patient_times, patient_scores, color=cmap(norm(i)), 
                          alpha=0.3, s=20)
                ax.plot(smooth_times, smooth_scores, color=cmap(norm(i)), 
                       alpha=0.8, linewidth=2, label=f"Patient {patient}")
            else:
                # Plot without smoothing
                ax.plot(patient_times, patient_scores, color=cmap(norm(i)), 
                       marker='o', alpha=0.8, linewidth=2, 
                       label=f"Patient {patient}")
    else:
        # Sort by time if necessary
        if len(timestamps) > 0:
            sort_idx = np.argsort(timestamps)
            sorted_times = timestamps[sort_idx]
            sorted_scores = risk_scores[sort_idx]
        else:
            sorted_times = timestamps
            sorted_scores = risk_scores
        
        # Apply rolling average if requested
        if rolling_window is not None and len(sorted_scores) >= rolling_window:
            # Calculate rolling mean
            smooth_scores = np.convolve(sorted_scores, 
                                      np.ones(rolling_window)/rolling_window, 
                                      mode='valid')
            # Adjust times to match the convolved data
            smooth_times = sorted_times[rolling_window-1:]
            
            # Plot raw data as scatter and smoothed as line
            ax.scatter(sorted_times, sorted_scores, color="#1f77b4", 
                      alpha=0.3, s=20)
            ax.plot(smooth_times, smooth_scores, color="#1f77b4", 
                   alpha=0.8, linewidth=2, label=f"Risk Score (Rolling Window={rolling_window})")
        else:
            # Plot without smoothing
            ax.plot(sorted_times, sorted_scores, color="#1f77b4", 
                   marker='o', alpha=0.8, linewidth=2, label="Risk Score")
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  label=f"Threshold ({threshold:.2f})")
        
        # Highlight high-risk periods if requested
        if highlight_high_risk:
            if patient_ids is not None:
                for i, patient in enumerate(unique_patients):
                    mask = patient_ids == patient
                    patient_times = timestamps[mask]
                    patient_scores = risk_scores[mask]
                    
                    # Sort by time if necessary
                    if len(patient_times) > 0:
                        sort_idx = np.argsort(patient_times)
                        patient_times = patient_times[sort_idx]
                        patient_scores = patient_scores[sort_idx]
                    
                    # Find periods above threshold
                    above_threshold = patient_scores > threshold
                    
                    if np.any(above_threshold):
                        # Create masked array for highlighting
                        masked_times = np.ma.masked_array(patient_times, mask=~above_threshold)
                        masked_scores = np.ma.masked_array(patient_scores, mask=~above_threshold)
                        
                        # Highlight points
                        ax.plot(masked_times, masked_scores, 'o', color='red', 
                               markersize=8, alpha=0.6)
            else:
                # Find periods above threshold
                above_threshold = sorted_scores > threshold
                
                if np.any(above_threshold):
                    # Create masked array for highlighting
                    masked_times = np.ma.masked_array(sorted_times, mask=~above_threshold)
                    masked_scores = np.ma.masked_array(sorted_scores, mask=~above_threshold)
                    
                    # Highlight points
                    ax.plot(masked_times, masked_scores, 'o', color='red', 
                           markersize=8, alpha=0.6)
    
    # Add event markers if provided
    if events:
        for event in events:
            event_time = event['time']
            event_label = event['label']
            event_color = event.get('color', 'green')
            event_marker = event.get('marker', '^')
            
            # Add marker at the event time
            ax.axvline(x=event_time, color=event_color, linestyle=':', alpha=0.7)
            
            # Add label
            ymin, ymax = ax.get_ylim()
            y_pos = ymin + 0.1 * (ymax - ymin)
            ax.text(event_time, y_pos, event_label, rotation=90, 
                   color=event_color, verticalalignment='bottom',
                   horizontalalignment='right')
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Risk Score")
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Format x-axis if timestamps are datetime
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_risk_stratification(risk_scores: Union[List[float], np.ndarray],
                           thresholds: List[float],
                           class_labels: Optional[Union[List[int], np.ndarray]] = None,
                           class_weights: Optional[List[float]] = None,
                           title: str = "Risk Stratification Analysis",
                           stratum_labels: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 10),
                           show_counts: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot risk stratification analysis with multiple thresholds.
    
    Parameters
    ----------
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores or probabilities
    thresholds : List[float]
        List of threshold values defining risk strata
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        True class labels (1 for positive, 0 for negative)
    class_weights : Optional[List[float]], default=None
        Weights for different classes, useful for calculating weighted metrics
    title : str, default="Risk Stratification Analysis"
        Title of the plot
    stratum_labels : Optional[List[str]], default=None
        Labels for risk strata, should have length len(thresholds) + 1
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    show_counts : bool, default=True
        Whether to show counts in each stratum
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Ensure numpy arrays
    risk_scores = np.array(risk_scores)
    thresholds = sorted(thresholds)  # Ensure thresholds are sorted
    
    # Set default stratum labels if not provided
    if stratum_labels is None:
        if len(thresholds) == 1:
            stratum_labels = ["Low Risk", "High Risk"]
        elif len(thresholds) == 2:
            stratum_labels = ["Low Risk", "Medium Risk", "High Risk"]
        elif len(thresholds) == 3:
            stratum_labels = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk"]
        else:
            stratum_labels = [f"Stratum {i+1}" for i in range(len(thresholds) + 1)]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Create grid for subplots
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
    
    # Subplot 1: Risk score distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create histogram with density
    hist_kws = {'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 1}
    sns.histplot(risk_scores, bins=30, kde=True, ax=ax1, **hist_kws)
    
    # Add vertical lines for thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    ymax = ax1.get_ylim()[1]
    
    for i, threshold in enumerate(thresholds):
        ax1.axvline(x=threshold, color=colors[i], linestyle='--', linewidth=2,
                  label=f"Threshold {i+1}: {threshold:.3f}")
    
    # Add shaded regions for each stratum
    x_min, x_max = ax1.get_xlim()
    all_boundaries = [x_min] + thresholds + [x_max]
    
    for i in range(len(all_boundaries) - 1):
        left = all_boundaries[i]
        right = all_boundaries[i+1]
        color = plt.cm.viridis(i / (len(all_boundaries) - 1))
        ax1.axvspan(left, right, alpha=0.1, color=color, label=stratum_labels[i])
    
    # Set labels
    ax1.set_xlabel("Risk Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Risk Score Distribution with Strata")
    
    # Subplot 2: Stratum populations
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate population in each stratum
    stratum_counts = []
    for i in range(len(all_boundaries) - 1):
        lower = all_boundaries[i]
        upper = all_boundaries[i+1]
        count = np.sum((risk_scores >= lower) & (risk_scores < upper))
        stratum_counts.append(count)
    
    # Create barplot
    colors = plt.cm.viridis(np.linspace(0, 1, len(stratum_counts)))
    bars = ax2.barh(range(len(stratum_counts)), stratum_counts, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add count labels
    if show_counts:
        for i, bar in enumerate(bars):
            width = bar.get_width()
            percentage = 100 * width / len(risk_scores)
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f"{int(width)} ({percentage:.1f}%)", 
                   va='center', fontsize=9)
    
    # Set labels
    ax2.set_yticks(range(len(stratum_counts)))
    ax2.set_yticklabels(stratum_labels)
    ax2.set_xlabel("Count")
    ax2.set_title("Population by Risk Stratum")
    
    # Subplot 3: Calibration plot (if class labels are provided)
    ax3 = fig.add_subplot(gs[1, 0])
    
    if class_labels is not None:
        class_labels = np.array(class_labels)
        
        # Calculate observed rate in each bin
        bin_edges = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(risk_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        bin_sums = np.bincount(bin_indices, minlength=len(bin_edges)-1)
        bin_pos = np.bincount(bin_indices, weights=class_labels, minlength=len(bin_edges)-1)
        
        bin_rates = np.zeros(len(bin_edges)-1)
        mask = bin_sums > 0
        bin_rates[mask] = bin_pos[mask] / bin_sums[mask]
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot calibration curve
        ax3.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
        ax3.plot(bin_centers, bin_rates, 'o-', color='#1f77b4', label="Observed Rate")
        
        # Calculate calibration metrics
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(class_labels, risk_scores, n_bins=10)
        
        # Add threshold indicators on calibration curve
        for i, threshold in enumerate(thresholds):
            # Find the closest bin center
            idx = np.abs(bin_centers - threshold).argmin()
            ax3.plot(bin_centers[idx], bin_rates[idx], 'o', 
                   color=colors[i], markersize=10, alpha=0.8,
                   label=f"Threshold {i+1}: {threshold:.3f}")
        
        # Set labels
        ax3.set_xlabel("Predicted Risk")
        ax3.set_ylabel("Observed Rate")
        ax3.set_title("Calibration Plot")
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    
    else:
        # If no class labels, show empty plot with message
        ax3.text(0.5, 0.5, "Calibration plot requires class labels", 
               ha='center', va='center', fontsize=12)
        ax3.set_xlabel("Predicted Risk")
        ax3.set_ylabel("Observed Rate")
        ax3.set_title("Calibration Plot (Not Available)")
    
    # Subplot 4: Confusion matrix or risk metrics by stratum
    ax4 = fig.add_subplot(gs[1, 1])
    
    if class_labels is not None:
        # Calculate metrics for each stratum
        metrics_by_stratum = []
        
        for i in range(len(all_boundaries) - 1):
            lower = all_boundaries[i]
            upper = all_boundaries[i+1]
            
            # Get samples in this stratum
            mask = (risk_scores >= lower) & (risk_scores < upper)
            stratum_scores = risk_scores[mask]
            stratum_labels = class_labels[mask]
            
            # Calculate metrics
            if len(stratum_labels) > 0:
                pos_rate = np.mean(stratum_labels)
                count = len(stratum_labels)
                pos_count = np.sum(stratum_labels)
                
                metrics_by_stratum.append({
                    'Stratum': stratum_labels[i],
                    'Positive Rate': pos_rate,
                    'Count': count,
                    'Positive Count': pos_count
                })
        
        # Create table with metrics
        cell_text = []
        for metrics in metrics_by_stratum:
            cell_text.append([
                f"{metrics['Positive Rate']:.2f}",
                f"{metrics['Positive Count']}/{metrics['Count']}"
            ])
        
        # Create table
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=cell_text,
                        rowLabels=stratum_labels,
                        colLabels=["Positive Rate", "Pos/Total"],
                        cellLoc='center',
                        rowLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the table cells by positive rate
        for i, metrics in enumerate(metrics_by_stratum):
            table[(i+1, 0)].set_facecolor(plt.cm.RdYlGn(1 - metrics['Positive Rate']))
    
    else:
        # If no class labels, show empty table with message
        ax4.text(0.5, 0.5, "Metrics require class labels", 
               ha='center', va='center', fontsize=10)
        ax4.axis('off')
    
    # Set overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_longitudinal_risk(patient_data: pd.DataFrame,
                         patient_id_col: str,
                         time_col: str,
                         risk_col: str,
                         event_col: Optional[str] = None,
                         title: str = "Longitudinal Risk Analysis",
                         figsize: Tuple[int, int] = (15, 10),
                         n_patients: Optional[int] = None,
                         threshold: Optional[float] = None,
                         color_map: str = "viridis",
                         highlight_events: bool = True,
                         rolling_window: Optional[int] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot longitudinal risk profiles for multiple patients.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        DataFrame containing patient data
    patient_id_col : str
        Column name for patient identifiers
    time_col : str
        Column name for time points
    risk_col : str
        Column name for risk scores
    event_col : Optional[str], default=None
        Column name for clinical events (1 for event, 0 for no event)
    title : str, default="Longitudinal Risk Analysis"
        Title of the plot
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    n_patients : Optional[int], default=None
        Number of patients to display, if None, all patients are shown
    threshold : Optional[float], default=None
        Risk threshold to show as horizontal line
    color_map : str, default="viridis"
        Colormap for patient lines
    highlight_events : bool, default=True
        Whether to highlight clinical events
    rolling_window : Optional[int], default=None
        Window size for rolling average, if None, no smoothing is applied
    save_path : Optional[str], default=None
        Path to save the figure, if None, figure is not saved
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Get unique patients
    unique_patients = patient_data[patient_id_col].unique()
    
    # Limit number of patients if specified
    if n_patients is not None:
        unique_patients = unique_patients[:n_patients]
    
    # Determine number of rows and columns for subplots
    n_cols = min(3, len(unique_patients))
    n_rows = (len(unique_patients) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    
    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Create colormap
    cmap = plt.cm.get_cmap(color_map)
    
    # Plot each patient
    for i, patient_id in enumerate(unique_patients):
        ax = axes[i]
        
        # Get data for this patient
        patient_df = patient_data[patient_data[patient_id_col] == patient_id].copy()
        
        # Sort by time
        patient_df = patient_df.sort_values(by=time_col)
        
        # Get times and risk scores
        times = patient_df[time_col].values
        risk_scores = patient_df[risk_col].values
        
        # Apply rolling average if requested
        if rolling_window is not None and len(risk_scores) >= rolling_window:
            # Calculate rolling mean
            smooth_scores = np.convolve(risk_scores, 
                                      np.ones(rolling_window)/rolling_window, 
                                      mode='valid')
            # Adjust times to match the convolved data
            smooth_times = times[rolling_window-1:]
            
            # Plot raw data as scatter and smoothed as line
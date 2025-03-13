"""
Risk visualization components for the FEP analysis web application.

This module provides specialized Streamlit components for displaying
risk stratification and temporal risk visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import io
import base64
import matplotlib.cm as cm
from matplotlib.colors import Normalize


CLASS_COLORS = {
    0: '#b30000',  # Dark red
    1: '#e60000',  # Red
    2: '#ff1a1a',  # Light red
    3: '#ff8000',  # Dark orange
    4: '#ffa64d',  # Orange
    5: '#ffcc00',  # Yellow-orange
    6: '#67e667',  # Light green
    7: '#2eb82e'   # Dark green
}

def get_class_color(class_num):
    """Get the color for a specific class."""
    return CLASS_COLORS.get(class_num, '#1f77b4')  # Default to blue if class not found


def display_probability_bars(probabilities: Dict[str, float],
                           title: str = "Outcome Probabilities",
                           threshold: Optional[float] = None,
                           sorted_by_value: bool = True,
                           positive_color: str = "#1f77b4",
                           negative_color: str = "#ff7f0e") -> None:
    """
    Display probability bars for multiple outcomes.
    
    Parameters
    ----------
    probabilities : Dict[str, float]
        Dictionary mapping outcome names to probabilities
    title : str, default="Outcome Probabilities"
        Title for the visualization
    threshold : Optional[float], default=None
        Threshold to mark on the bars
    sorted_by_value : bool, default=True
        Whether to sort bars by probability values
    positive_color : str, default="#1f77b4"
        Color for probabilities above threshold
    negative_color : str, default="#ff7f0e"
        Color for probabilities below threshold
    """
    st.subheader(title)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({
        'Outcome': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    
    # Sort if requested
    if sorted_by_value:
        df = df.sort_values('Probability', ascending=False)
    
    # Create a horizontal bar chart with Altair
    base = alt.Chart(df).encode(
        y=alt.Y('Outcome:N', sort=None),  # Use the order in the DataFrame
        tooltip=['Outcome:N', 'Probability:Q']
    )
    
    if isinstance(df['Outcome'].iloc[0], (int, np.integer)) or df['Outcome'].iloc[0].isdigit():
        # If outcomes are class numbers, use class colors
        class_colors = [get_class_color(int(outcome)) for outcome in df['Outcome']]

        bars = base.mark_bar().encode(
        x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Outcome:N', scale=alt.Scale(domain=list(df['Outcome']), range=class_colors))
    )

    # Create the bars with conditional coloring if threshold is provided
    if threshold is not None:
        # Define color based on threshold
        color_condition = alt.condition(
            alt.datum.Probability >= threshold,
            alt.value(positive_color),
            alt.value(negative_color)
        )
        
        bars = base.mark_bar().encode(
            x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
            color=color_condition
        )
        
        # Add threshold line
        threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(
            color='black',
            strokeDash=[3, 3],
            size=2
        ).encode(
            x='threshold:Q'
        )
        
        # Combine the chart elements
        chart = (bars + threshold_line).properties(
            width=600,
            height=300
        )
    else:
        # Simple bars without threshold
        chart = base.mark_bar().encode(
            x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.value(positive_color)
        ).properties(
            width=600,
            height=300
        )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Add table with actual values
    st.caption("Probability Values")
    df['Probability'] = df['Probability'].apply(lambda x: f"{x:.2%}")
    st.dataframe(df, hide_index=True)
    
    # Add information about threshold if provided
    if threshold is not None:
        above_threshold = df[df['Probability'].apply(lambda x: float(x.strip('%'))/100 >= threshold)]
        if not above_threshold.empty:
            st.info(f"Outcomes above threshold ({threshold:.2%}): {', '.join(above_threshold['Outcome'].tolist())}")




def display_risk_timeline(timestamps: Union[List, np.ndarray],
                         risk_scores: Union[List[float], np.ndarray],
                         patient_ids: Optional[Union[List, np.ndarray]] = None,
                         class_labels: Optional[Union[List[int], np.ndarray]] = None,  
                         events: Optional[List[Dict[str, Any]]] = None,
                         title: str = "Risk Score Timeline",
                         threshold: Optional[float] = None,
                         show_raw_data: bool = True,
                         show_trend: bool = True,
                         window_size: int = 3,
                         color_map: str = "viridis") -> None:
    """
    Display a timeline of risk scores with optional event markers.
    
    Parameters
    ----------
    timestamps : Union[List, np.ndarray]
        List of time points (datetime or numeric)
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores at each time point
    patient_ids : Optional[Union[List, np.ndarray]], default=None
        Patient identifiers if multiple patients
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        Class labels (0-7) corresponding to each time point for color coding
    events : Optional[List[Dict[str, Any]]], default=None
        List of events to mark on the timeline.
        Each event should be a dict with at least 'time' and 'label' keys.
    title : str, default="Risk Score Timeline"
        Title for the visualization
    threshold : Optional[float], default=None
        Risk threshold to show as horizontal line
    show_raw_data : bool, default=True
        Whether to show raw data points
    show_trend : bool, default=True
        Whether to show trend line
    window_size : int, default=3
        Window size for rolling average trend
    color_map : str, default="viridis"
        Colormap for multiple patients
    """
    st.subheader(title)
    
    # Convert to numpy arrays for easier handling
    timestamps = np.array(timestamps)
    risk_scores = np.array(risk_scores)
    
    # Check if we have multiple patients
    if patient_ids is not None:
        patient_ids = np.array(patient_ids)
        unique_patients = np.unique(patient_ids)
        
        # Create interactive chart with Altair
        # Prepare data
        data = pd.DataFrame({
            'timestamp': timestamps,
            'risk_score': risk_scores,
            'patient_id': patient_ids
        })
        
        # Create base chart
        chart = alt.Chart(data).encode(
            x='timestamp:T',
            color='patient_id:N'
        )
        
        # Add points if requested
        if show_raw_data:
            points = chart.mark_circle(size=60).encode(
                y='risk_score:Q',
                tooltip=['timestamp:T', 'risk_score:Q', 'patient_id:N']
            )
        
        # Add trend line if requested
        if show_trend:
            line = chart.mark_line().encode(
                y='risk_score:Q'
            )
            
            # Combine charts
            if show_raw_data:
                combined_chart = (points + line)
            else:
                combined_chart = line
        else:
            combined_chart = points
        
        # Add threshold if provided
        if threshold is not None:
            threshold_line = alt.Chart(
                pd.DataFrame({'threshold': [threshold]})
            ).mark_rule(color='red', strokeDash=[3, 3]).encode(
                y='threshold:Q',
                tooltip=alt.Tooltip('threshold:Q', title='Threshold')
            )
            
            combined_chart = combined_chart + threshold_line
        
        # Display chart
        st.altair_chart(combined_chart, use_container_width=True)
    
    else:
        # Single patient - use matplotlib for more customization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw data if requested
        if show_raw_data:
            ax.scatter(timestamps, risk_scores, color='#1f77b4', alpha=0.7, s=50)
        
        # Add trend line if requested
        if show_trend and len(risk_scores) >= window_size:
            # Calculate rolling average
            rolling_avg = np.convolve(risk_scores, np.ones(window_size)/window_size, mode='valid')
            # Adjust timestamps to match rolling average
            rolling_timestamps = timestamps[window_size-1:]
            
            ax.plot(rolling_timestamps, rolling_avg, color='#1f77b4', linewidth=2, 
                   label=f'Trend (Window={window_size})')
        elif show_trend:
            ax.plot(timestamps, risk_scores, color='#1f77b4', linewidth=2, label='Trend')
        
        # Add threshold if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      label=f'Threshold ({threshold:.2f})')
            
            # Highlight high risk periods
            above_threshold = risk_scores > threshold
            
            if np.any(above_threshold):
                high_risk_times = [t for t, above in zip(timestamps, above_threshold) if above]
                high_risk_scores = [s for s, above in zip(risk_scores, above_threshold) if above]
                
                if 'class_labels' in locals() and class_labels is not None:
                    # Filter class labels for above-threshold points
                    high_risk_classes = [c for c, above in zip(class_labels, above_threshold) if above]
                    # Use class-specific colors for each point
                    for t, s, c in zip(high_risk_times, high_risk_scores, high_risk_classes):
                        ax.scatter(t, s, color=get_class_color(c), s=80, zorder=5, 
                                edgecolor='black')
                else:
                     # Default coloring when class labels aren't available
                    ax.scatter(high_risk_times, high_risk_scores, color='red', s=80, zorder=5, 
                            label='High Risk', edgecolor='darkred')
        
        # Add events if provided
        if events:
            event_times = [event['time'] for event in events]
            event_labels = [event['label'] for event in events]
            
            # Find y position for events (slightly above highest risk score)
            y_pos = max(risk_scores) * 1.05
            
            # Add event markers
            ax.scatter(event_times, [y_pos] * len(event_times), marker='^', color='green', 
                      s=100, zorder=10, label='Events')
            
            # Add event labels
            for time, label in zip(event_times, event_labels):
                ax.annotate(label, (time, y_pos), xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, color='darkgreen')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Score')
        ax.set_title(title)
        
        # Format x-axis if timestamps are datetime
        if pd.api.types.is_datetime64_any_dtype(timestamps) or isinstance(timestamps[0], datetime):
            fig.autofmt_xdate()
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Display chart
        st.pyplot(fig)
    
    # Create data table
    if patient_ids is not None:
        data = pd.DataFrame({
            'Timestamp': timestamps,
            'Risk Score': risk_scores,
            'Patient ID': patient_ids
        })
    else:
        data = pd.DataFrame({
            'Timestamp': timestamps,
            'Risk Score': risk_scores
        })
    
    # Display data table in expander
    with st.expander("View Data"):
        st.dataframe(data)
    
    # Add download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="risk_timeline.csv",
        mime="text/csv"
    )


def display_risk_distribution(risk_scores: Union[List[float], np.ndarray],
                            class_labels: Optional[Union[List[int], np.ndarray]] = None,
                            thresholds: Optional[List[float]] = None,
                            title: str = "Risk Score Distribution",
                            show_metrics: bool = True) -> None:
    """
    Display distribution of risk scores with optional class separation.
    
    Parameters
    ----------
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores or probabilities
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        True class labels (1 for positive, 0 for negative)
    thresholds : Optional[List[float]], default=None
        List of threshold values to show as vertical lines
    title : str, default="Risk Score Distribution"
        Title for the visualization
    show_metrics : bool, default=True
        Whether to show metrics like separation statistics
    """
    st.subheader(title)
    
    # Ensure numpy arrays
    risk_scores = np.array(risk_scores)
    if class_labels is not None:
        class_labels = np.array(class_labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create KDE plot
    if class_labels is not None:
        # Separate scores by class
        pos_scores = risk_scores[class_labels == 1]
        neg_scores = risk_scores[class_labels == 0]
        
        # Create histograms for each class
        sns.histplot(pos_scores, kde=True, color="#ff7f0e", alpha=0.6, 
                   label="Positive Class", ax=ax)
        sns.histplot(neg_scores, kde=True, color="#1f77b4", alpha=0.6, 
                   label="Negative Class", ax=ax)
        
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
            
            # Add metrics as text box
            textstr = (f"Pos Mean: {pos_mean:.3f}, Neg Mean: {neg_mean:.3f}\n"
                     f"Pos Std: {pos_std:.3f}, Neg Std: {neg_std:.3f}\n"
                     f"Separation: {separation:.3f}")
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top', bbox=props)
    else:
        # Create a single histogram
        sns.histplot(risk_scores, kde=True, color="#1f77b4", ax=ax)
    
    # Add threshold lines if provided
    if thresholds is not None:
        ymax = ax.get_ylim()[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
        
        for i, threshold in enumerate(thresholds):
            ax.axvline(x=threshold, color=colors[i], linestyle='--', 
                      label=f"Threshold {i+1}: {threshold:.3f}")
    
    # Set labels and title
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    # Display chart
    st.pyplot(fig)
    
    # Display summary statistics
    if show_metrics:
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean", f"{np.mean(risk_scores):.3f}")
        
        with col2:
            st.metric("Median", f"{np.median(risk_scores):.3f}")
        
        with col3:
            st.metric("Standard Deviation", f"{np.std(risk_scores):.3f}")
        
        # Display quantiles
        st.write("Quantiles:")
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = np.quantile(risk_scores, quantiles)
        
        # Create DataFrame for quantiles
        quantile_df = pd.DataFrame({
            'Quantile': [f"{q*100:.0f}%" for q in quantiles],
            'Value': [f"{v:.3f}" for v in quantile_values]
        })
        
        # Display as columns
        cols = st.columns(len(quantiles))
        for i, (_, row) in enumerate(quantile_df.iterrows()):
            with cols[i]:
                st.metric(row['Quantile'], row['Value'])


def display_risk_stratification(risk_scores: Union[List[float], np.ndarray],
                              thresholds: List[float],
                              class_labels: Optional[Union[List[int], np.ndarray]] = None,
                              stratum_labels: Optional[List[str]] = None,
                              title: str = "Risk Stratification Analysis") -> None:
    """
    Display risk stratification analysis with multiple thresholds.
    
    Parameters
    ----------
    risk_scores : Union[List[float], np.ndarray]
        Predicted risk scores or probabilities
    thresholds : List[float]
        List of threshold values defining risk strata
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        True class labels (1 for positive, 0 for negative)
    stratum_labels : Optional[List[str]], default=None
        Labels for risk strata, should have length len(thresholds) + 1
    title : str, default="Risk Stratification Analysis"
        Title for the visualization
    """
    st.subheader(title)
    
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
    
    # Define all boundaries including min and max
    all_boundaries = [0.0] + thresholds + [1.0]
    
    # Create DataFrame for display
    data = []
    for i in range(len(all_boundaries) - 1):
        lower = all_boundaries[i]
        upper = all_boundaries[i+1]
        
        # Count samples in this stratum
        stratum_mask = (risk_scores >= lower) & (risk_scores < upper)
        count = np.sum(stratum_mask)
        percentage = 100 * count / len(risk_scores)
        
        # Calculate positive rate if class labels are provided
        if class_labels is not None:
            class_labels_np = np.array(class_labels)
            stratum_labels_np = class_labels_np[stratum_mask]
            positive_count = np.sum(stratum_labels_np == 1)
            positive_rate = 100 * positive_count / count if count > 0 else 0
        else:
            positive_count = None
            positive_rate = None
        
        # Add to data
        data.append({
            'Stratum': stratum_labels[i],
            'Lower Bound': lower,
            'Upper Bound': upper,
            'Count': count,
            'Percentage': percentage,
            'Positive Count': positive_count,
            'Positive Rate': positive_rate
        })
    
    df = pd.DataFrame(data)
    
    # Create visualization
    # 1. Horizontal bar chart for population distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create colormap
    #colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    if 'class' in df.columns:
    # Use class-specific colors if class column exists
        colors = [get_class_color(cls) for cls in df['class']]
    else:
    # Create colors based on risk level names
        risk_colors = {
            'High Risk': '#ff1a1a',
            'Moderate Risk': '#ffa64d',
            'Low Risk': '#67e667'
        }
    colors = [risk_colors.get(stratum, plt.cm.viridis(i/len(df))) for i, stratum in enumerate(df['Stratum'])]
    # Plot horizontal bars
    bars = ax.barh(df['Stratum'], df['Count'], color=colors)
    
    # Add count and percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = df.iloc[i]['Percentage']
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
               f"{int(width)} ({percentage:.1f}%)", 
               va='center')
    
    # Set labels and title
    ax.set_xlabel('Count')
    ax.set_ylabel('Risk Stratum')
    ax.set_title('Population by Risk Stratum')
    
    # Display chart
    st.pyplot(fig)
    
    # If class labels are provided, create positive rate visualization
    if class_labels is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot positive rates
        bars = ax2.barh(df['Stratum'], df['Positive Rate'], color=colors)
        
        # Add positive rate labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pos_count = df.iloc[i]['Positive Count']
            count = df.iloc[i]['Count']
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"{width:.1f}% ({pos_count}/{count})", 
                   va='center')
        
        # Set labels and title
        ax2.set_xlabel('Positive Rate (%)')
        ax2.set_ylabel('Risk Stratum')
        ax2.set_title('Positive Rate by Risk Stratum')
        
        # Display chart
        st.pyplot(fig2)
    
    # Display data table
    st.subheader("Stratification Data")
    
    # Format table for display
    display_df = df.copy()
    display_df['Lower Bound'] = display_df['Lower Bound'].apply(lambda x: f"{x:.2f}")
    display_df['Upper Bound'] = display_df['Upper Bound'].apply(lambda x: f"{x:.2f}")
    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    if class_labels is not None:
        display_df['Positive Rate'] = display_df['Positive Rate'].apply(lambda x: f"{x:.1f}%" if x is not None else "N/A")
    else:
        display_df = display_df.drop(columns=['Positive Count', 'Positive Rate'])
    
    st.dataframe(display_df)
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="risk_stratification.csv",
        mime="text/csv"
    )


def display_risk_map(x_values: Union[List[float], np.ndarray],
                   y_values: Union[List[float], np.ndarray],
                   risk_scores: Union[List[float], np.ndarray],
                   x_label: str = "Feature 1",
                   y_label: str = "Feature 2",
                   title: str = "Risk Map",
                   point_size: int = 100,
                   class_labels: Optional[Union[List[int], np.ndarray]] = None,
                   threshold: Optional[float] = None) -> None:
    """
    Display a 2D risk map visualizing risk scores across two features.
    
    Parameters
    ----------
    x_values : Union[List[float], np.ndarray]
        Values for x-axis feature
    y_values : Union[List[float], np.ndarray]
        Values for y-axis feature
    risk_scores : Union[List[float], np.ndarray]
        Risk scores for each point
    x_label : str, default="Feature 1"
        Label for x-axis
    y_label : str, default="Feature 2"
        Label for y-axis
    title : str, default="Risk Map"
        Title for the visualization
    point_size : int, default=100
        Size of scatter points
    class_labels : Optional[Union[List[int], np.ndarray]], default=None
        True class labels (1 for positive, 0 for negative)
    threshold : Optional[float], default=None
        Risk threshold for coloring points
    """
    st.subheader(title)
    
    # Create DataFrame
    data = pd.DataFrame({
        'x': x_values,
        'y': y_values,
        'risk': risk_scores
    })
    
    # Add class labels if provided
    if class_labels is not None:
        data['class'] = class_labels
    
    # Create interactive scatter plot with Altair
    # Base chart
    if class_labels is not None:
        # Use class labels for shape
        chart = alt.Chart(data).mark_circle(size=point_size).encode(
            x=alt.X('x:Q', title=x_label),
            y=alt.Y('y:Q', title=y_label),
            color=alt.Color('class:N', scale=alt.Scale(domain=list(CLASS_COLORS.keys()),
            range=list(CLASS_COLORS.values())
            )),
            #color=alt.Color('risk:Q', title='Risk Score', scale=alt.Scale(scheme='viridis')),
            shape=alt.Shape('class:N', title='Class'),
            tooltip=['x:Q', 'y:Q', 'risk:Q', 'class:N']
        )
    else:
        # No class labels, just use risk for color
        chart = alt.Chart(data).mark_circle(size=point_size).encode(
            x=alt.X('x:Q', title=x_label),
            y=alt.Y('y:Q', title=y_label),
            color=alt.Color('risk:Q', title='Risk Score', scale=alt.Scale(scheme='viridis')),
            tooltip=['x:Q', 'y:Q', 'risk:Q']
        )
    
    # If threshold is provided, add a layer for high-risk points
    if threshold is not None:
        # Create a selection for high-risk points
        high_risk = data[data['risk'] >= threshold]
        
        if len(high_risk) > 0:
            # Create layer for high-risk points
            high_risk_layer = alt.Chart(high_risk).mark_circle(
                size=point_size*1.5,
                stroke='black',
                strokeWidth=2
            ).encode(
                x=alt.X('x:Q'),
                y=alt.Y('y:Q'),
                color=alt.value('red'),
                tooltip=['x:Q', 'y:Q', 'risk:Q']
            )
            
            # Combine layers
            chart = alt.layer(chart, high_risk_layer)
    
    # Set chart properties
    chart = chart.properties(
        width=600,
        height=400,
        title=title
    ).interactive()
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # Display data table in expander
    with st.expander("View Data"):
        st.dataframe(data)
    
    # Add download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="risk_map.csv",
        mime="text/csv"
    )

def display_risk_factors(risk_factors: List[Dict[str, Any]],
                       title: str = "Key Risk Factors",
                       show_weights: bool = True,
                       show_descriptions: bool = True) -> None:
    """
    Display key risk factors with their weights and descriptions.
    
    Parameters
    ----------
    risk_factors : List[Dict[str, Any]]
        List of risk factor dictionaries, each containing at least 'name' and 'weight' keys.
        Optional keys include 'description', 'category', and 'modifiable'.
    title : str, default="Key Risk Factors"
        Title for the visualization
    show_weights : bool, default=True
        Whether to show risk factor weights
    show_descriptions : bool, default=True
        Whether to show risk factor descriptions
    """
    st.subheader(title)
    
    # Sort risk factors by absolute weight (descending)
    sorted_factors = sorted(risk_factors, key=lambda x: abs(x.get('weight', 0)), reverse=True)
    
    # Prepare data for display
    risk_data = []
    protective_data = []
    
    for factor in sorted_factors:
        # Get factor properties
        name = factor.get('name', 'Unknown')
        weight = factor.get('weight', 0)
        description = factor.get('description', '')
        category = factor.get('category', 'Other')
        modifiable = factor.get('modifiable', False)
        
        # Create display dictionary
        display_dict = {
            'name': name,
            'weight': weight,
            'description': description,
            'category': category,
            'modifiable': modifiable
        }
        
        # Add to appropriate list based on weight
        if weight >= 0:
            risk_data.append(display_dict)
        else:
            protective_data.append(display_dict)
    
    # Create tabs for risk and protective factors
    if risk_data and protective_data:
        risk_tab, protective_tab = st.tabs(["Risk Factors", "Protective Factors"])
        
        with risk_tab:
            _display_factor_table(risk_data, "risk", show_weights, show_descriptions)
        
        with protective_tab:
            _display_factor_table(protective_data, "protective", show_weights, show_descriptions)
    elif risk_data:
        _display_factor_table(risk_data, "risk", show_weights, show_descriptions)
    elif protective_data:
        _display_factor_table(protective_data, "protective", show_weights, show_descriptions)
    else:
        st.info("No risk factors available for display.")


def _display_factor_table(factors: List[Dict[str, Any]], 
                        factor_type: str,
                        show_weights: bool,
                        show_descriptions: bool) -> None:
    """
    Helper function to display a table of risk or protective factors.
    
    Parameters
    ----------
    factors : List[Dict[str, Any]]
        List of factor dictionaries to display
    factor_type : str
        Type of factors ("risk" or "protective")
    show_weights : bool
        Whether to show weights
    show_descriptions : bool
        Whether to show descriptions
    """
    # Create DataFrame
    df = pd.DataFrame(factors)
    
    # Format DataFrame for display
    if 'weight' in df.columns:
        df['weight'] = df['weight'].apply(lambda x: abs(x))  # Use absolute value for display
    
    # Add modifiable indicator
    if 'modifiable' in df.columns:
        df['modifiable'] = df['modifiable'].apply(lambda x: "✓" if x else "")
    
    # Group by category if available
    if 'category' in df.columns and len(df['category'].unique()) > 1:
        categories = sorted(df['category'].unique())
        
        for category in categories:
            st.write(f"**{category}**")
            category_factors = df[df['category'] == category]
            
            for _, row in category_factors.iterrows():
                _display_factor_card(row, factor_type, show_weights, show_descriptions)
            
            st.markdown("---")
    else:
        # Display all factors without categorization
        for _, row in df.iterrows():
            _display_factor_card(row, factor_type, show_weights, show_descriptions)


def _display_factor_card(factor: pd.Series, 
                       factor_type: str,
                       show_weights: bool,
                       show_descriptions: bool) -> None:
    """
    Display a single factor as a card-like element.
    
    Parameters
    ----------
    factor : pd.Series
        Factor data as a pandas Series
    factor_type : str
        Type of factor ("risk" or "protective")
    show_weights : bool
        Whether to show weights
    show_descriptions : bool
        Whether to show descriptions
    """
    # Determine color based on factor type
    # Determine color based on factor type or class if available    
    if 'class' in factor and factor['class'] is not None:
        color = get_class_color(factor['class'])
    else:
        color = "red" if factor_type == "risk" else "green"
    
    # Create columns for layout
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Display factor name with optional modifiable indicator
        modifiable = factor.get('modifiable', False)
        modifiable_indicator = " 🔄" if modifiable else ""
        st.markdown(f"**{factor['name']}{modifiable_indicator}**")
        
        # Display description if available and requested
        if show_descriptions and 'description' in factor and factor['description']:
            st.caption(factor['description'])
    
    with col2:
        # Display weight if requested
        if show_weights and 'weight' in factor:
            weight = factor['weight']
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{weight:.2f}</span>", 
                       unsafe_allow_html=True)

def display_longitudinal_risk(patient_data: pd.DataFrame,
                            patient_id_col: str = "patient_id",
                            time_col: str = "timestamp",
                            risk_col: str = "risk_score",
                            event_col: Optional[str] = None,
                            title: str = "Longitudinal Risk Analysis",
                            n_patients: Optional[int] = None,
                            threshold: Optional[float] = None) -> None:
    """
    Display longitudinal risk profiles for multiple patients.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        DataFrame containing patient data
    patient_id_col : str, default="patient_id"
        Column name for patient identifiers
    time_col : str, default="timestamp"
        Column name for time points
    risk_col : str, default="risk_score"
        Column name for risk scores
    event_col : Optional[str], default=None
        Column name for clinical events (1 for event, 0 for no event)
    title : str, default="Longitudinal Risk Analysis"
        Title for the visualization
    n_patients : Optional[int], default=None
        Number of patients to display, if None, all patients are shown
    threshold : Optional[float], default=None
        Risk threshold to show as horizontal line
    """
    st.subheader(title)
    
    # Get unique patients
    unique_patients = patient_data[patient_id_col].unique()
    
    # Limit number of patients if specified
    if n_patients is not None and n_patients < len(unique_patients):
        unique_patients = unique_patients[:n_patients]
        st.caption(f"Showing {n_patients} of {len(unique_patients)} patients")
    
    # Patient selector
    selected_patient = st.selectbox(
        "Select Patient",
        options=["All Patients"] + unique_patients.tolist()
    )
    
    # Filter data based on selection
    if selected_patient == "All Patients":
        filtered_data = patient_data[patient_data[patient_id_col].isin(unique_patients)]
    else:
        filtered_data = patient_data[patient_data[patient_id_col] == selected_patient]
    
    # Create interactive visualization with Altair
    # Base chart
    chart = alt.Chart(filtered_data).encode(
        x=alt.X(f'{time_col}:T', title='Time')
    )
    
    # Add lines for risk scores
    lines = chart.mark_line().encode(
        y=alt.Y(f'{risk_col}:Q', title='Risk Score'),
        color=alt.Color(f'{patient_id_col}:N', title='Patient')
    )
    
    # Add points for risk scores
    points = chart.mark_circle(size=60).encode(
        y=alt.Y(f'{risk_col}:Q'),
        color=alt.Color(f'{patient_id_col}:N'),
        tooltip=[f'{time_col}:T', f'{risk_col}:Q', f'{patient_id_col}:N']
    )
    
    # Combine lines and points
    combined = lines + points
    
    # Add threshold if provided
    if threshold is not None:
        threshold_line = alt.Chart(
            pd.DataFrame({'threshold': [threshold]})
        ).mark_rule(color='red', strokeDash=[3, 3]).encode(
            y='threshold:Q',
            tooltip=alt.Tooltip('threshold:Q', title='Threshold')
        )
        
        combined = combined + threshold_line
    
    # Add events if available
    if event_col is not None:
        events = filtered_data[filtered_data[event_col] == 1]
        
        if len(events) > 0:
            event_markers = alt.Chart(events).mark_point(
                shape='triangle-up',
                size=150,
                color='green',
                filled=True
            ).encode(
                x=f'{time_col}:T',
                y=alt.Y(f'{risk_col}:Q'),
                tooltip=[f'{time_col}:T', f'{patient_id_col}:N', 'event_type:N'] 
                if 'event_type' in events.columns else [f'{time_col}:T', f'{patient_id_col}:N']
            )
            
            combined = combined + event_markers
    
    # Set chart properties
    combined = combined.properties(
        width=700,
        height=400,
        title=title
    ).interactive()
    
    # Display chart
    st.altair_chart(combined, use_container_width=True)
    
    # Display summary statistics if showing all patients
    if selected_patient == "All Patients":
        st.subheader("Patient Risk Summary")
        
        # Calculate summary statistics by patient
        summary = filtered_data.groupby(patient_id_col)[risk_col].agg(
            ['mean', 'min', 'max', 'std', 'count']
        ).reset_index()
        
        # Format for display
        summary.columns = [patient_id_col, 'Mean Risk', 'Min Risk', 'Max Risk', 'Std Dev', 'Observations']
        summary = summary.sort_values('Mean Risk', ascending=False)
        
        # Apply formatting
        for col in ['Mean Risk', 'Min Risk', 'Max Risk', 'Std Dev']:
            summary[col] = summary[col].apply(lambda x: f"{x:.3f}")
        
        # Display table
        st.dataframe(summary)
    
    # Display data table in expander
    with st.expander("View Data"):
        st.dataframe(filtered_data)
    
    # Add download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="longitudinal_risk.csv",
        mime="text/csv"
    )


def display_risk_composition(feature_contributions: pd.DataFrame,
                           risk_scores: Union[List[float], np.ndarray],
                           title: str = "Risk Score Composition",
                           n_features: int = 5,
                           normalize: bool = False) -> None:
    """
    Display composition of risk scores by feature contributions.
    
    Parameters
    ----------
    feature_contributions : pd.DataFrame
        DataFrame with feature contributions (features as columns)
    risk_scores : Union[List[float], np.ndarray]
        Overall risk scores
    title : str, default="Risk Score Composition"
        Title for the visualization
    n_features : int, default=5
        Number of top features to show individually
    normalize : bool, default=False
        Whether to normalize contributions to sum to 1
    """
    st.subheader(title)
    
    # Ensure feature_contributions is a DataFrame
    if not isinstance(feature_contributions, pd.DataFrame):
        st.error("Feature contributions must be a pandas DataFrame")
        return
    
    # Sort patients by risk score
    indices = np.argsort(risk_scores)[::-1]  # Descending order
    sorted_contribs = feature_contributions.iloc[indices]
    sorted_scores = np.array(risk_scores)[indices]
    
    # Add index column for sample IDs
    sorted_contribs = sorted_contribs.reset_index(drop=True)
    sorted_contribs.index = sorted_contribs.index.map(lambda x: f"Sample {x+1}")
    
    # Select top features by mean absolute contribution
    feature_means = sorted_contribs.abs().mean().sort_values(ascending=False)
    top_features = feature_means.index[:n_features].tolist()
    
    # Group remaining features
    other_features = [col for col in sorted_contribs.columns if col not in top_features]
    
    # Create a new DataFrame with top features and "Other" column
    plot_data = pd.DataFrame(index=sorted_contribs.index)
    
    # Add top features
    for feature in top_features:
        plot_data[feature] = sorted_contribs[feature]
    
    # Add "Other" as sum of remaining features
    if other_features:
        plot_data["Other"] = sorted_contribs[other_features].sum(axis=1)
    
    # Add risk score for reference
    plot_data["Risk Score"] = sorted_scores
    
    # Normalize if requested
    if normalize:
        # Calculate row sums excluding risk score
        row_sums = plot_data.drop(columns=["Risk Score"]).abs().sum(axis=1)
        # Normalize each row
        for col in plot_data.columns:
            if col != "Risk Score":
                plot_data[col] = plot_data[col].div(row_sums)
    
    # Limit to top 20 samples for readability
    if len(plot_data) > 20:
        plot_data = plot_data.head(20)
    
    # Transpose for better visualization
    plot_data_t = plot_data.drop(columns=["Risk Score"]).T
    
    # Create interactive heatmap with Altair
    # Prepare data for Altair (long format)
    heat_data = plot_data.drop(columns=["Risk Score"]).reset_index().melt(
        id_vars="index",
        var_name="Feature",
        value_name="Contribution"
    )
    
    # Create heatmap
    heatmap = alt.Chart(heat_data).mark_rect().encode(
        x=alt.X('index:N', title='Sample'),
        y=alt.Y('Feature:N', title='Feature'),
        color=alt.Color('Contribution:Q', scale=alt.Scale(scheme='blueorange', domain=[-1, 1])),
        tooltip=['index:N', 'Feature:N', 'Contribution:Q']
    ).properties(
        width=700,
        height=400,
        title=title
    )
    
    # Display chart
    st.altair_chart(heatmap, use_container_width=True)
    
    # Display average contributions
    st.subheader("Average Feature Contributions")
    
    # Calculate average contributions
    avg_contribs = plot_data.drop(columns=["Risk Score"]).mean().sort_values(ascending=False)
    
    # Create bar chart
    bar_data = pd.DataFrame({
        'Feature': avg_contribs.index,
        'Contribution': avg_contribs.values
    })
    
    bar_chart = alt.Chart(bar_data).mark_bar().encode(
        x=alt.X('Contribution:Q'),
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Contribution:Q', scale=alt.Scale(scheme='blueorange', domain=[-1, 1])),
        tooltip=['Feature:N', 'Contribution:Q']
    ).properties(
        width=600,
        height=300,
        title="Average Feature Contributions"
    )
    
    # Display chart
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Display data table in expander
    with st.expander("View Data"):
        st.dataframe(plot_data)
    
    # Add download button
    csv = plot_data.to_csv()
    st.download_button(
        label="Download Data",
        data=csv,
        file_name="risk_composition.csv",
        mime="text/csv"
    )

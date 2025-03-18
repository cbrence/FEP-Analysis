"""
Result display components for the FEP analysis web application.

This module provides reusable Streamlit components for displaying
analysis results, predictions, and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import altair as alt
from datetime import datetime, timedelta
import io
import base64


def display_prediction_results(predictions: Union[pd.DataFrame, Dict[str, Any]],
                              threshold: float = 0.5,
                              patient_id: Optional[str] = None,
                              show_gauge: bool = True,
                              show_features: bool = True,
                              show_explanation: bool = True,
                              explanation_method: str = "shap") -> None:
    """
    Display prediction results with visualizations.
    
    Parameters
    ----------
    predictions : Union[pd.DataFrame, Dict[str, Any]]
        Prediction results from the model
    threshold : float, default=0.5
        Threshold for binary predictions
    patient_id : Optional[str], default=None
        Patient identifier for display
    show_gauge : bool, default=True
        Whether to show the gauge visualization
    show_features : bool, default=True
        Whether to show feature contributions
    show_explanation : bool, default=True
        Whether to show prediction explanations
    explanation_method : str, default="shap"
        Method used for feature importance explanation
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(predictions, dict):
        predictions = pd.DataFrame([predictions])
    
    # Extract prediction score
    if 'score' in predictions.columns:
        score = predictions['score'].iloc[0]
    elif 'probability' in predictions.columns:
        score = predictions['probability'].iloc[0]
    elif 'risk_score' in predictions.columns:
        score = predictions['risk_score'].iloc[0]
    else:
        # Assume first numeric column is the score
        numeric_cols = predictions.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            score = predictions[numeric_cols[0]].iloc[0]
        else:
            st.error("No prediction score found in results")
            return
    
    # Determine risk level based on threshold
    risk_level = "High Risk" if score >= threshold else "Low Risk"
    risk_color = "red" if score >= threshold else "green"
    
    # Display header with patient ID if provided
    if patient_id:
        st.subheader(f"Prediction Results for Patient {patient_id}")
    else:
        st.subheader("Prediction Results")
    
    # Create columns layout
    col1, col2 = st.columns([1, 2])
    
    # Display gauge in first column if requested
    if show_gauge:
        with col1:
            create_gauge_chart(score, threshold)
    
    # Display summary metrics in second column
    with col2:
        # Show risk level and score
        st.markdown(
            f"<h3 style='color: {risk_color}; margin-bottom: 0;'>{risk_level}</h3>", 
            unsafe_allow_html=True
        )
        st.markdown(f"**Risk Score:** {score:.3f}")
        st.markdown(f"**Threshold:** {threshold:.2f}")
        
        # Display timestamp if available
        if 'timestamp' in predictions.columns:
            timestamp = predictions['timestamp'].iloc[0]
            st.markdown(f"**Prediction Time:** {timestamp}")
    
    # Display feature contributions if requested
    if show_features and 'feature_contributions' in predictions.columns:
        st.subheader("Feature Contributions")
        
        # Extract feature contributions
        feature_contribs = predictions['feature_contributions'].iloc[0]
        
        if isinstance(feature_contribs, (str, bytes)):
            # Parse from JSON if stored as string
            feature_contribs = pd.read_json(feature_contribs)
        
        # Plot feature contributions
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_contribs.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Feature Contributions to Prediction")
        ax.set_xlabel("Contribution")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
    
    # Display prediction explanation if requested
    if show_explanation:
        st.subheader("Prediction Explanation")
        
        if explanation_method == "shap" and 'shap_values' in predictions.columns:
            # Display SHAP visualization if available
            display_shap_explanation(predictions)
        else:
            # Create a simple explanation based on the score
            if score >= threshold:
                st.markdown(
                    f"This patient is classified as **{risk_level}** with a "
                    f"score of **{score:.3f}**, which is above the threshold of {threshold:.2f}."
                )
                
                if "top_risk_factors" in predictions.columns:
                    st.markdown("**Key risk factors identified:**")
                    risk_factors = predictions["top_risk_factors"].iloc[0]
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
            else:
                st.markdown(
                    f"This patient is classified as **{risk_level}** with a "
                    f"score of **{score:.3f}**, which is below the threshold of {threshold:.2f}."
                )
                
                if "protective_factors" in predictions.columns:
                    st.markdown("**Protective factors identified:**")
                    protective_factors = predictions["protective_factors"].iloc[0]
                    for factor in protective_factors:
                        st.markdown(f"- {factor}")
    
    # Display additional information if available
    if 'notes' in predictions.columns and not pd.isna(predictions['notes'].iloc[0]):
        st.subheader("Additional Notes")
        st.markdown(predictions['notes'].iloc[0])


def create_gauge_chart(value: float, threshold: float = 0.5) -> None:
    """
    Create a gauge chart visualization for risk score.
    
    Parameters
    ----------
    value : float
        Risk score value (0-1)
    threshold : float, default=0.5
        Threshold value for risk determination
    """
    # Create a gauge chart using Altair
    # Define the chart data
    df = pd.DataFrame({
        'value': [0, value, 1],
        'color': ['low', 'current', 'high']
    })
    
    # Create a colored background arc for the gauge
    arc_bg = alt.Chart(pd.DataFrame({
        'start': [0],
        'end': [1],
        'threshold': [threshold]
    })).mark_arc(
        innerRadius=80,
        outerRadius=100,
        startAngle=0,
        endAngle=180,
    ).encode(
        color=alt.Color(
            'threshold:N',
            legend=None,
            scale=alt.Scale(domain=['threshold'], range=['lightgray'])
        )
    )
    
    # Create the main gauge arc
    arc = alt.Chart(df.iloc[[0, 1]]).mark_arc(
        innerRadius=80,
        outerRadius=100,
        startAngle=0,
        endAngle=180 * value
    ).encode(
        color=alt.Color(
            'color:N',
            legend=None,
            scale=alt.Scale(
                domain=['low', 'current', 'high'],
                range=['green', 'red' if value >= threshold else 'green', 'red']
            )
        )
    )
    
    # Add threshold marker
    threshold_marker = alt.Chart(pd.DataFrame({
        'x': [threshold],
        'y': [0]
    })).mark_tick(
        thickness=2,
        color='black',
        size=120
    ).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), axis=None),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[-1, 1]), axis=None)
    ).transform_calculate(
        x=f'0.5 + {threshold}/2 * cos(180 * {threshold} * PI/180)',
        y=f'0.5 + sin(180 * {threshold} * PI/180)'
    )
    
    # Add text labels
    text = alt.Chart(df.iloc[[1]]).mark_text(
        align='center',
        baseline='middle',
        fontSize=28,
        fontWeight='bold'
    ).encode(
        text=alt.Text('value:Q', format='.2f'),
        color=alt.value('black')
    ).transform_calculate(
        x='0.5',
        y='0.7'
    )
    
    # Combine charts
    gauge = (arc_bg + arc + text).properties(
        width=200,
        height=150
    ).configure_view(
        strokeWidth=0
    )
    
    # Display the chart
    st.altair_chart(gauge, use_container_width=True)


def display_shap_explanation(predictions: pd.DataFrame) -> None:
    """
    Display SHAP-based explanation for predictions.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        Prediction results containing SHAP values
    """
    try:
        import shap
        
        # Extract SHAP values
        shap_values = predictions['shap_values'].iloc[0]
        
        if isinstance(shap_values, (str, bytes)):
            # Parse from JSON if stored as string
            shap_values = np.array(pd.read_json(shap_values))
        
        # Get feature names if available
        if 'feature_names' in predictions.columns:
            feature_names = predictions['feature_names'].iloc[0]
        else:
            # Assume column names without prediction columns
            exclude_cols = ['score', 'probability', 'risk_score', 'prediction', 
                          'timestamp', 'shap_values', 'feature_contributions',
                          'top_risk_factors', 'protective_factors', 'notes']
            feature_names = [col for col in predictions.columns if col not in exclude_cols]
        
        # Get feature values
        feature_values = predictions.iloc[0][feature_names].values
        
        # Create SHAP force plot
        st.write("SHAP Force Plot Explanation:")
        
        # Create figure using matplotlib since Streamlit doesn't directly support shap.plots
        plt.figure(figsize=(10, 3))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(plt)
        
    except ImportError:
        st.warning("SHAP package is required for detailed explanations. Please install it with `pip install shap`")
    except Exception as e:
        st.error(f"Error displaying SHAP explanation: {str(e)}")


def display_feature_importance(feature_names: List[str],
                              importance_values: List[float],
                              title: str = "Feature Importance",
                              top_n: int = 10,
                              method: str = "model",
                              color: str = "#1f77b4") -> None:
    """
    Display feature importance visualization.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importance_values : List[float]
        Importance values for each feature
    title : str, default="Feature Importance"
        Title for the visualization
    top_n : int, default=10
        Number of top features to display
    method : str, default="model"
        Method used to calculate feature importance
    color : str, default="#1f77b4"
        Color for the visualization
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Select top features if there are more features than top_n
    if len(df) > top_n:
        df = df.head(top_n)
    
    # Create visualization
    st.subheader(title)
    
    # Add method description
    if method == "model":
        st.caption("Based on model's internal feature importance")
    elif method == "shap":
        st.caption("Based on SHAP values (impact on model output)")
    elif method == "permutation":
        st.caption("Based on permutation importance (decrease in performance when feature is shuffled)")
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.sort_values('Importance').plot.barh(x='Feature', y='Importance', ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    
    # Display the chart
    st.pyplot(fig)
    
    # Display data table
    with st.expander("View Data"):
        st.dataframe(df)
    
    # Add download button
    csv = df.to_csv(index=False)
    download_button(csv, "feature_importance.csv", "Download Data")

def display_model_comparison(models_metrics: Dict[str, Dict[str, float]],
                           metrics_to_show: Optional[List[str]] = None,
                           higher_is_better: Optional[Dict[str, bool]] = None,
                           sort_by: Optional[str] = None,
                           title: str = "Model Comparison",
                           show_as_table: bool = True,
                           show_as_chart: bool = True) -> None:
    """
    Display comparison of multiple models across different metrics.
    
    Parameters
    ----------
    models_metrics : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and dictionaries of metrics as values
    metrics_to_show : Optional[List[str]], default=None
        List of metrics to include in comparison. If None, all metrics are shown.
    higher_is_better : Optional[Dict[str, bool]], default=None
        Dictionary specifying whether higher values are better for each metric
    sort_by : Optional[str], default=None
        Metric to sort models by
    title : str, default="Model Comparison"
        Title for the visualization
    show_as_table : bool, default=True
        Whether to show results as a table
    show_as_chart : bool, default=True
        Whether to show results as a chart
    """
    st.subheader(title)
    
    # Set default higher_is_better if not provided
    if higher_is_better is None:
        # Common metrics and whether higher values are better
        higher_is_better = {
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True,
            'auc': True,
            'roc_auc': True,
            'pr_auc': True,
            'mse': False,
            'rmse': False,
            'mae': False,
            'log_loss': False,
            'clinical_utility': True
        }
    
    # Filter metrics if specified
    if metrics_to_show is not None:
        models_metrics = {
            model: {k: v for k, v in metrics.items() if k in metrics_to_show}
            for model, metrics in models_metrics.items()
        }
    
    # Create DataFrame for easier manipulation
    data = []
    for model, metrics in models_metrics.items():
        row = {'Model': model}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort if specified
    if sort_by is not None and sort_by in df.columns:
        is_better = higher_is_better.get(sort_by, True)
        df = df.sort_values(sort_by, ascending=not is_better)
    
    # Format numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_display = df.copy()
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
    
    # Display as table if requested
    if show_as_table:
        st.dataframe(df_display)
    
    # Display as chart if requested
    if show_as_chart and len(df) > 0:
        # Prepare data for chart
        chart_data = df.melt(id_vars=['Model'], var_name='Metric', value_name='Value')
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=chart_data, x='Metric', y='Value', hue='Model', ax=ax)
        
        # Customize chart
        ax.set_title(title)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Display chart
        st.pyplot(fig)
    
    # Add download button
    csv = df.to_csv(index=False)
    download_button(csv, "model_comparison.csv", "Download Data")

def display_confidence_intervals(point_estimate: float,
                               lower_bound: float,
                               upper_bound: float,
                               method: str = "Bootstrap",
                               confidence_level: float = 0.95,
                               metric_name: str = "Performance",
                               color: str = "#1f77b4") -> None:
    """
    Display confidence intervals for model metrics.
    
    Parameters
    ----------
    point_estimate : float
        Point estimate (central value)
    lower_bound : float
        Lower bound of confidence interval
    upper_bound : float
        Upper bound of confidence interval
    method : str, default="Bootstrap"
        Method used to calculate confidence intervals
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence)
    metric_name : str, default="Performance"
        Name of the metric
    color : str, default="#1f77b4"
        Color for the visualization
    """
    # Create layout
    st.subheader(f"{metric_name} with Confidence Interval")
    st.caption(f"{confidence_level*100:.0f}% Confidence Interval ({method})")
    
    # Display values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Lower Bound", value=f"{lower_bound:.3f}")
    
    with col2:
        st.metric(label="Point Estimate", value=f"{point_estimate:.3f}")
    
    with col3:
        st.metric(label="Upper Bound", value=f"{upper_bound:.3f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Plot point estimate and CI
    ax.plot([point_estimate], [0], 'o', color=color, markersize=10)
    ax.plot([lower_bound, upper_bound], [0, 0], '-', color=color, linewidth=2)
    
    # Add labels
    ax.text(lower_bound, 0.1, f"{lower_bound:.3f}", ha='center', va='bottom')
    ax.text(point_estimate, 0.1, f"{point_estimate:.3f}", ha='center', va='bottom')
    ax.text(upper_bound, 0.1, f"{upper_bound:.3f}", ha='center', va='bottom')
    
    # Set axis limits and hide y-axis
    buffer = (upper_bound - lower_bound) * 0.1
    ax.set_xlim(lower_bound - buffer, upper_bound + buffer)
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Display the chart
    st.pyplot(fig)


def display_threshold_analysis(thresholds: List[float],
                             metrics: Dict[str, List[float]],
                             optimal_threshold: Optional[float] = None,
                             title: str = "Threshold Analysis",
                             metrics_to_show: Optional[List[str]] = None) -> None:
    """
    Display analysis of how different thresholds affect model metrics.
    
    Parameters
    ----------
    thresholds : List[float]
        List of threshold values
    metrics : Dict[str, List[float]]
        Dictionary with metric names as keys and lists of values as values
    optimal_threshold : Optional[float], default=None
        Optimal threshold value to highlight
    title : str, default="Threshold Analysis"
        Title for the visualization
    metrics_to_show : Optional[List[str]], default=None
        List of metrics to include. If None, all metrics are shown.
    """
    st.subheader(title)
    
    # Filter metrics if specified
    if metrics_to_show is not None:
        metrics = {k: v for k, v in metrics.items() if k in metrics_to_show}
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({'threshold': thresholds})
    for metric, values in metrics.items():
        df[metric] = values
    
    # Display interactive chart
    chart_data = df.melt(id_vars=['threshold'], var_name='metric', value_name='value')
    
    # Create line chart using Altair
    base = alt.Chart(chart_data).encode(
        x=alt.X('threshold:Q', title='Threshold'),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('metric:N', title='Metric')
    )
    
    line = base.mark_line().encode(
        tooltip=['threshold:Q', 'metric:N', 'value:Q']
    )
    
    # Add vertical line for optimal threshold if provided
    if optimal_threshold is not None:
        threshold_line = alt.Chart(
            pd.DataFrame({'threshold': [optimal_threshold]})
        ).mark_rule(color='red', strokeDash=[3, 3]).encode(
            x='threshold:Q',
            tooltip=alt.Tooltip('threshold:Q', title='Optimal Threshold')
        )
        
        # Combine charts
        chart = (line + threshold_line).properties(
            width=700,
            height=400,
            title=title
        ).interactive()
    else:
        chart = line.properties(
            width=700,
            height=400,
            title=title
        ).interactive()
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # Display data table
    with st.expander("View Data"):
        st.dataframe(df)
    
    # Display optimal threshold information if provided
    if optimal_threshold is not None:
        st.info(f"Optimal Threshold: {optimal_threshold:.3f}")
        
        # Find metrics at optimal threshold
        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - optimal_threshold))
        st.write("Metrics at optimal threshold:")
        
        # Display metric values
        metric_cols = st.columns(len(metrics))
        for i, (metric, values) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(label=metric, value=f"{values[idx]:.3f}")
    
    # Add download button
    csv = df.to_csv(index=False)
    download_button(csv, "threshold_analysis.csv", "Download Data")


def display_metrics_table(metrics: Dict[str, float],
                        comparison_metrics: Optional[Dict[str, float]] = None,
                        title: str = "Model Performance Metrics",
                        higher_is_better: Optional[Dict[str, bool]] = None) -> None:
    """
    Display a table of model metrics with optional comparison.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    comparison_metrics : Optional[Dict[str, float]], default=None
        Dictionary of metric names and values for comparison
    title : str, default="Model Performance Metrics"
        Title for the table
    higher_is_better : Optional[Dict[str, bool]], default=None
        Dictionary specifying whether higher values are better for each metric
    """
    st.subheader(title)
    
    # Set default higher_is_better if not provided
    if higher_is_better is None:
        # Common metrics and whether higher values are better
        higher_is_better = {
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True,
            'auc': True,
            'roc_auc': True,
            'pr_auc': True,
            'mse': False,
            'rmse': False,
            'mae': False,
            'log_loss': False,
            'clinical_utility': True
        }
    
    # Create DataFrame from metrics
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # Add comparison if provided
    if comparison_metrics is not None:
        # Add comparison values
        df['Comparison'] = df['Metric'].map(
            lambda x: comparison_metrics.get(x, None)
        )
        
        # Calculate difference
        df['Difference'] = df.apply(
            lambda row: row['Value'] - row['Comparison'] 
            if not pd.isna(row['Comparison']) else None,
            axis=1
        )
        
        # Determine if difference is improvement
        df['Improvement'] = df.apply(
            lambda row: higher_is_better.get(row['Metric'], True) == (row['Difference'] > 0)
            if not pd.isna(row['Difference']) else None,
            axis=1
        )
    
    # Format values
    df['Value'] = df['Value'].apply(lambda x: f"{x:.3f}")
    
    if comparison_metrics is not None:
        df['Comparison'] = df['Comparison'].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else None
        )
        
        df['Difference'] = df['Difference'].apply(
            lambda x: f"{x:+.3f}" if not pd.isna(x) else None
        )
    
    # Display table
    if comparison_metrics is not None:
        # Create a more visual display with comparison
        for i, row in df.iterrows():
            metric = row['Metric']
            value = row['Value']
            comparison = row['Comparison']
            difference = row['Difference']
            improvement = row['Improvement']
            
            # Create row with columns
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{metric}**")
            
            with col2:
                st.write(value)
            
            with col3:
                if comparison is not None:
                    st.write(comparison)
                else:
                    st.write("—")
            
            with col4:
                if difference is not None:
                    color = "green" if improvement else "red"
                    st.markdown(f"<span style='color:{color}'>{difference}</span>", unsafe_allow_html=True)
                else:
                    st.write("—")
            
            # Add separator
            st.markdown("---")
    else:
        # Simple table without comparison
        for i, row in df.iterrows():
            metric = row['Metric']
            value = row['Value']
            
            # Create row with columns
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write(f"**{metric}**")
            
            with col2:
                st.write(value)
            
            # Add separator
            st.markdown("---")
    
    # Add download button
    csv = df.to_csv(index=False)
    download_button(csv, "metrics.csv", "Download Data")


def download_button(object_to_download: Union[str, bytes, pd.DataFrame],
                  download_filename: str,
                  button_text: str) -> None:
    """
    Create a button to download data.
    
    Parameters
    ----------
    object_to_download : Union[str, bytes, pd.DataFrame]
        Object to be downloaded
    download_filename : str
        Filename for the download
    button_text : str
        Text to display on the button
    """
    # Convert DataFrame to CSV string
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Convert to bytes if string
    if isinstance(object_to_download, str):
        object_to_download = object_to_download.encode()
    
    # Convert to base64
    b64 = base64.b64encode(object_to_download).decode()
    
    # Determine MIME type
    if download_filename.endswith('.csv'):
        file_type = 'text/csv'
    elif download_filename.endswith('.json'):
        file_type = 'application/json'
    else:
        file_type = 'application/octet-stream'
    
    # Create download link
    href = f'<a href="data:{file_type};base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    # Display button
    st.markdown(href, unsafe_allow_html=True)
   # : str = "Feature Importance",
  #                           top_n: int = 10,
  #                           method: str = "model",
  #                           color: str = "#1f77b4") -> None:
    """
    Display feature importance visualization.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importance_values : List[float]
        Importance values for each feature
    title : str, default="Feature Importance"
        Title for the visualization
    top_n : int, default=10
        Number of top features to display
    method : str, default="model"
        Method used to calculate feature importance
    color : str, default="#1f77b4"
        Color for the visualization
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Select top features
    df = df.head(top_n)
    
    # Create visualization
    st.subheader(title)
    
    # Add method description
    if method == "model":
        st.caption("Based on model's internal feature importance")
    elif method == "shap":
        st.caption("Based on SHAP values (impact on model output)")
    elif method == "permutation":
        st.caption("Based on permutation importance (decrease in performance when feature is shuffled)")
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.sort_values('Importance').plot.barh(x='Feature', y='Importance', ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    
    # Display the chart
    st.pyplot(fig)
    
    # Display data table
    with st.expander("View Data"):
        st.dataframe(df)
    
    # Add download button
    csv = df.to_csv(index=False)
    download_button(csv, "feature_importance.csv", "Download Data")

    
def download_button(object_to_download: Union[str, bytes, pd.DataFrame],
                  download_filename: str,
                  button_text: str) -> None:
    """
    Create a button to download data.
    
    Parameters
    ----------
    object_to_download : Union[str, bytes, pd.DataFrame]
        Object to be downloaded
    download_filename : str
        Filename for the download
    button_text : str
        Text to display on the button
    """
    # Convert DataFrame to CSV string
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Convert to bytes if string
    if isinstance(object_to_download, str):
        object_to_download = object_to_download.encode()
    
    # Convert to base64
    b64 = base64.b64encode(object_to_download).decode()
    
    # Determine MIME type
    if download_filename.endswith('.csv'):
        file_type = 'text/csv'
    elif download_filename.endswith('.json'):
        file_type = 'application/json'
    else:
        file_type = 'application/octet-stream'
    
    # Create download link
    href = f'<a href="data:{file_type};base64,{b64}" download="{download_filename}">{button_text}</a>'
    
    # Display button
    st.markdown(href, unsafe_allow_html=True)

def display_confusion_matrix(y_true: Union[List[int], np.ndarray], 
                          y_pred: Union[List[int], np.ndarray],
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          normalize: bool = False,
                          cmap: str = "Blues",
                          include_metrics: bool = True,
                          figsize: Tuple[int, int] = (8, 6)) -> None: 
    """
    Display a confusion matrix with optional metrics.
        
    Parameters
    ----------
    y_true : Union[List[int], np.ndarray]
        True class labels
        y_pred : Union[List[int], np.ndarray]
            Predicted class labels
        class_names : Optional[List[str]], default=None
            Names for classes (e.g., ["Negative", "Positive"])
        title : str, default="Confusion Matrix"
            Title for the visualization
        normalize : bool, default=False
            Whether to normalize the confusion matrix
        cmap : str, default="Blues"
            Colormap for the confusion matrix
        include_metrics : bool, default=True
            Whether to include accuracy, precision, recall, and F1 score
        figsize : Tuple[int, int], default=(8, 6)
            Figure size
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set class names if not provided
    if class_names is None:
        # Default to binary classification names
        if cm.shape[0] == 2:
            class_names = ["Negative", "Positive"]
        else:
            class_names = [str(i) for i in range(cm.shape[0])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Plot confusion matrix
    im = sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Fix for matplotlib issues with tight layout
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display metrics if requested
    if include_metrics and len(np.unique(y_true)) == 2:  # Only for binary classification
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        
        with col4:
            st.metric("F1 Score", f"{f1:.3f}")
        
        # Add explanation of the metrics
        with st.expander("Metrics Explanation"):
            st.markdown("""
            - **Accuracy**: The proportion of correctly classified instances (TP + TN) / (TP + TN + FP + FN)
            - **Precision**: The proportion of true positives among instances predicted as positive TP / (TP + FP)
            - **Recall**: The proportion of true positives identified correctly TP / (TP + FN)
            - **F1 Score**: The harmonic mean of precision and recall 2 * (Precision * Recall) / (Precision + Recall)
            
            Where:
            - TP = True Positives
            - TN = True Negatives
            - FP = False Positives
            - FN = False Negatives
            """)
        
        # Calculate and display additional information about the confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        st.write("Confusion Matrix Details:")
        matrix_details = pd.DataFrame({
            "Metric": ["True Positives (TP)", "False Positives (FP)", "False Negatives (FN)", "True Negatives (TN)"],
            "Count": [tp, fp, fn, tn],
            "Description": [
                "Correctly predicted positive cases",
                "Incorrectly predicted positive cases (Type I error)",
                "Incorrectly predicted negative cases (Type II error)",
                "Correctly predicted negative cases"
            ]
        })
        
        st.dataframe(matrix_details, hide_index=True)
    
    # For multiclass provide different metrics
    elif include_metrics and len(np.unique(y_true)) > 2:
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            st.metric("Precision (macro)", f"{precision:.3f}")
        
        with col3:
            st.metric("Recall (macro)", f"{recall:.3f}")
        
        with col4:
            st.metric("F1 Score (macro)", f"{f1:.3f}")
        
        # Add note about macro averaging
        st.caption("Note: Precision, Recall, and F1 are calculated using macro averaging (unweighted mean per class)")
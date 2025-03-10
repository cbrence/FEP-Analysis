"""
Temporal feature engineering for early warning signs.

This module provides functions to create features that capture symptom
trends over time to detect early warning signs of relapse.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings, clinical_weights

# Setup logging
logger = logging.getLogger(__name__)

class SymptomTrendExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal trends from longitudinal symptom data.
    
    This transformer creates features that capture the trajectory of symptoms
    over time, which can help identify early warning signs of relapse.
    """
    
    def __init__(self, symptom_cols=None, id_col='patient_id', time_col='days_since_baseline'):
        """
        Initialize the SymptomTrendExtractor.
        
        Parameters:
        -----------
        symptom_cols : list of str, default=None
            List of symptom columns to analyze. If None, infer from data.
        id_col : str, default='patient_id'
            Column containing patient identifiers.
        time_col : str, default='days_since_baseline'
            Column containing time information.
        """
        self.symptom_cols = symptom_cols
        self.id_col = id_col
        self.time_col = time_col
        
    def fit(self, X, y=None):
        """
        Fit the transformer (identify symptom columns if needed).
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
        y : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # If symptom columns not specified, infer them
        if self.symptom_cols is None:
            # Look for PANSS columns (P, N, G prefixes)
            panss_cols = [col for col in X.columns if 
                         (any(col.startswith(prefix) for prefix in ['P', 'N', 'G']) and ':' in col) or
                         any(col.startswith(prefix) for prefix in ['M0_PANSS_P', 'M0_PANSS_N', 'M0_PANSS_G'])]
            
            self.symptom_cols = panss_cols
            logger.info(f"Identified {len(panss_cols)} symptom columns")
        
        return self
    
    def transform(self, X):
        """
        Transform the data to extract temporal trends.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with temporal trend features.
        """
        # Check that required columns exist
        required_cols = [self.id_col, self.time_col] + self.symptom_cols
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Filter to only include available columns
            symptom_cols = [col for col in self.symptom_cols if col in X.columns]
            if not symptom_cols:
                logger.error("No valid symptom columns found")
                return pd.DataFrame(index=X[self.id_col].unique())
        else:
            symptom_cols = self.symptom_cols
        
        # Create an output dataframe with one row per patient
        patient_ids = X[self.id_col].unique()
        trend_features = pd.DataFrame(index=patient_ids)
        
        # For each symptom, calculate trends
        for symptom in symptom_cols:
            # Initialize arrays for each trend metric
            slopes = []
            volatilities = []
            recent_changes = []
            max_increases = []
            last_values = []
            
            for patient_id in patient_ids:
                # Get patient data sorted by time
                patient_data = X[X[self.id_col] == patient_id].sort_values(self.time_col)
                
                # Extract symptom values and times
                symptom_values = patient_data[symptom].values
                times = patient_data[self.time_col].values
                
                # Calculate trend metrics if we have enough data points
                if len(symptom_values) > 1:
                    # Calculate slope using linear regression
                    slope, _, _, _, _ = stats.linregress(times, symptom_values)
                    slopes.append(slope)
                    
                    # Calculate volatility (standard deviation of changes)
                    changes = np.diff(symptom_values)
                    volatility = np.std(changes) if len(changes) > 0 else 0
                    volatilities.append(volatility)
                    
                    # Calculate recent change (last two observations)
                    recent_change = changes[-1] if len(changes) > 0 else 0
                    recent_changes.append(recent_change)
                    
                    # Calculate maximum increase
                    max_increase = np.max(changes) if len(changes) > 0 else 0
                    max_increases.append(max_increase)
                    
                    # Store last value
                    last_values.append(symptom_values[-1])
                else:
                    # Not enough data points for trends
                    slopes.append(0)
                    volatilities.append(0)
                    recent_changes.append(0)
                    max_increases.append(0)
                    last_values.append(symptom_values[0] if len(symptom_values) > 0 else np.nan)
            
            # Add trend features to the output dataframe
            trend_features[f'{symptom}_slope'] = slopes
            trend_features[f'{symptom}_volatility'] = volatilities
            trend_features[f'{symptom}_recent_change'] = recent_changes
            trend_features[f'{symptom}_max_increase'] = max_increases
            trend_features[f'{symptom}_last_value'] = last_values
        
        return trend_features

class EarlyWarningSigns(BaseEstimator, TransformerMixin):
    """
    Extract early warning signs based on clinical knowledge.
    
    This transformer creates features that identify specific patterns known
    to precede relapse in FEP patients.
    """
    
    def __init__(self, warning_signs=None):
        """
        Initialize the EarlyWarningSigns transformer.
        
        Parameters:
        -----------
        warning_signs : dict, default=None
            Dictionary mapping warning sign names to conditions.
            If None, use default warning signs from settings.
        """
        self.warning_signs = warning_signs or settings.EARLY_WARNING_SIGNS
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no actual fitting needed).
        
        Parameters:
        -----------
        X : pandas DataFrame
            Data with symptom measurements.
        y : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        return self
    
    def transform(self, X):
        """
        Transform the data to extract early warning signs.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Data with symptom measurements and trend features.
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with early warning sign features.
        """
        # Create output dataframe
        warning_features = pd.DataFrame(index=X.index)
        
        # Process each warning sign
        for sign_name, related_symptoms in self.warning_signs.items():
            # Check which symptoms we have data for
            available_symptoms = [s for s in related_symptoms if s in X.columns]
            
            if not available_symptoms:
                logger.warning(f"No symptoms available for warning sign: {sign_name}")
                continue
            
            # Calculate warning sign features
            
            # 1. Average symptom level
            symptom_avg = X[available_symptoms].mean(axis=1)
            warning_features[f'{sign_name}_level'] = symptom_avg
            
            # 2. Check for recent increases in any related symptom
            increase_cols = [f'{s}_recent_change' for s in available_symptoms 
                            if f'{s}_recent_change' in X.columns]
            
            if increase_cols:
                # Maximum recent increase in any related symptom
                max_increase = X[increase_cols].max(axis=1)
                warning_features[f'{sign_name}_increase'] = max_increase
                
                # Binary flag for significant increase (>= 2 points on scale)
                warning_features[f'{sign_name}_alert'] = (max_increase >= 2).astype(int)
            
            # 3. Trend slope for related symptoms
            slope_cols = [f'{s}_slope' for s in available_symptoms 
                         if f'{s}_slope' in X.columns]
            
            if slope_cols:
                # Average slope across related symptoms
                avg_slope = X[slope_cols].mean(axis=1)
                warning_features[f'{sign_name}_trend'] = avg_slope
                
                # Binary flag for concerning trend (slope > 0.1 points per time unit)
                warning_features[f'{sign_name}_trend_alert'] = (avg_slope > 0.1).astype(int)
        
        # Create overall early warning score
        alert_cols = [col for col in warning_features.columns if 'alert' in col]
        if alert_cols:
            warning_features['early_warning_score'] = warning_features[alert_cols].sum(axis=1)
        
        return warning_features

class SymptomStabilityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features that measure symptom stability over time.
    
    This transformer quantifies the stability of symptoms, which can help identify
    patients with fluctuating symptoms who may be at risk of relapse.
    """
    
    def __init__(self, symptom_cols=None, id_col='patient_id', time_col='days_since_baseline',
                 min_observations=3):
        """
        Initialize the SymptomStabilityExtractor.
        
        Parameters:
        -----------
        symptom_cols : list of str, default=None
            List of symptom columns to analyze. If None, infer from data.
        id_col : str, default='patient_id'
            Column containing patient identifiers.
        time_col : str, default='days_since_baseline'
            Column containing time information.
        min_observations : int, default=3
            Minimum number of observations required to calculate stability.
        """
        self.symptom_cols = symptom_cols
        self.id_col = id_col
        self.time_col = time_col
        self.min_observations = min_observations
        
    def fit(self, X, y=None):
        """
        Fit the transformer (identify symptom columns if needed).
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
        y : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # If symptom columns not specified, infer them
        if self.symptom_cols is None:
            # Look for PANSS columns (P, N, G prefixes)
            panss_cols = [col for col in X.columns if 
                         (any(col.startswith(prefix) for prefix in ['P', 'N', 'G']) and ':' in col) or
                         any(col.startswith(prefix) for prefix in ['M0_PANSS_P', 'M0_PANSS_N', 'M0_PANSS_G'])]
            
            self.symptom_cols = panss_cols
        
        return self
    
    def transform(self, X):
        """
        Transform the data to extract stability features.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with stability features.
        """
        # Check required columns
        required_cols = [self.id_col, self.time_col] + self.symptom_cols
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            symptom_cols = [col for col in self.symptom_cols if col in X.columns]
        else:
            symptom_cols = self.symptom_cols
        
        # Create output dataframe
        patient_ids = X[self.id_col].unique()
        stability_features = pd.DataFrame(index=patient_ids)
        
        # Calculate stability metrics for each patient and symptom
        for patient_id in patient_ids:
            # Get patient data
            patient_data = X[X[self.id_col] == patient_id].sort_values(self.time_col)
            
            # Skip if not enough observations
            if len(patient_data) < self.min_observations:
                continue
            
            # Calculate stability metrics for each symptom
            for symptom in symptom_cols:
                values = patient_data[symptom].values
                
                # Calculate coefficient of variation (normalized measure of dispersion)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else np.nan
                
                # Calculate autocorrelation (measure of sequential dependence)
                if len(values) > 1:
                    # Use lag-1 autocorrelation
                    autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                else:
                    autocorr = np.nan
                
                # Calculate entropy (measure of unpredictability)
                if len(values) > 2:
                    # Convert to discrete bins for entropy calculation
                    bins = min(len(values), 5)  # Use at most 5 bins
                    hist, _ = np.histogram(values, bins=bins)
                    hist = hist / len(values)  # Normalize
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small constant to avoid log(0)
                else:
                    entropy = np.nan
                
                # Store metrics
                stability_features.loc[patient_id, f'{symptom}_cv'] = cv
                stability_features.loc[patient_id, f'{symptom}_autocorr'] = autocorr
                stability_features.loc[patient_id, f'{symptom}_entropy'] = entropy
        
        # Create aggregate stability metrics across all symptoms
        for metric in ['cv', 'autocorr', 'entropy']:
            cols = [col for col in stability_features.columns if col.endswith(f'_{metric}')]
            if cols:
                stability_features[f'overall_{metric}'] = stability_features[cols].mean(axis=1)
        
        return stability_features

class TemporalPatternDetector(BaseEstimator, TransformerMixin):
    """
    Detect specific temporal patterns that may indicate risk of relapse.
    
    This transformer identifies patterns such as sudden increases in symptoms,
    sustained worsening, or irregular fluctuations.
    """
    
    def __init__(self, symptom_cols=None, id_col='patient_id', time_col='days_since_baseline',
                 sudden_increase_threshold=2.0, sustained_increase_min_periods=2):
        """
        Initialize the TemporalPatternDetector.
        
        Parameters:
        -----------
        symptom_cols : list of str, default=None
            List of symptom columns to analyze. If None, infer from data.
        id_col : str, default='patient_id'
            Column containing patient identifiers.
        time_col : str, default='days_since_baseline'
            Column containing time information.
        sudden_increase_threshold : float, default=2.0
            Threshold for detecting sudden increases in symptoms.
        sustained_increase_min_periods : int, default=2
            Minimum number of consecutive periods with increases to be considered sustained.
        """
        self.symptom_cols = symptom_cols
        self.id_col = id_col
        self.time_col = time_col
        self.sudden_increase_threshold = sudden_increase_threshold
        self.sustained_increase_min_periods = sustained_increase_min_periods
        
    def fit(self, X, y=None):
        """
        Fit the transformer (identify symptom columns if needed).
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
        y : array-like, default=None
            Not used, included for API compatibility.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # If symptom columns not specified, infer them
        if self.symptom_cols is None:
            # Look for PANSS columns
            panss_cols = [col for col in X.columns if 
                         (any(col.startswith(prefix) for prefix in ['P', 'N', 'G']) and ':' in col) or
                         any(col.startswith(prefix) for prefix in ['M0_PANSS_P', 'M0_PANSS_N', 'M0_PANSS_G'])]
            
            self.symptom_cols = panss_cols
        
        return self
    
    def transform(self, X):
        """
        Transform the data to detect temporal patterns.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Longitudinal data with multiple observations per patient.
            
        Returns:
        --------
        pandas DataFrame
            DataFrame with pattern detection features.
        """
        # Check required columns
        required_cols = [self.id_col, self.time_col] + self.symptom_cols
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            symptom_cols = [col for col in self.symptom_cols if col in X.columns]
        else:
            symptom_cols = self.symptom_cols
        
        # Create output dataframe
        patient_ids = X[self.id_col].unique()
        pattern_features = pd.DataFrame(index=patient_ids)
        
        # Detect patterns for each patient
        for patient_id in patient_ids:
            # Get patient data
            patient_data = X[X[self.id_col] == patient_id].sort_values(self.time_col)
            
            # Skip if not enough observations
            if len(patient_data) < 2:
                continue
            
            # Create aggregated pattern features
            sudden_increases = []
            sustained_increases = []
            fluctuations = []
            
            # Analyze each symptom
            for symptom in symptom_cols:
                values = patient_data[symptom].values
                times = patient_data[self.time_col].values
                
                # Calculate changes between consecutive measurements
                changes = np.diff(values)
                
                # 1. Detect sudden increases
                sudden_increase = np.any(changes >= self.sudden_increase_threshold)
                if sudden_increase:
                    sudden_increases.append(symptom)
                
                # 2. Detect sustained increases
                # Count consecutive increases
                consecutive_increases = 0
                has_sustained_increase = False
                
                for change in changes:
                    if change > 0:
                        consecutive_increases += 1
                    else:
                        consecutive_increases = 0
                        
                    if consecutive_increases >= self.sustained_increase_min_periods:
                        has_sustained_increase = True
                        break
                
                if has_sustained_increase:
                    sustained_increases.append(symptom)
                
                # 3. Detect fluctuations
                # Count direction changes
                direction_changes = 0
                prev_direction = None
                
                for change in changes:
                    if change == 0:
                        continue
                        
                    direction = 1 if change > 0 else -1
                    
                    if prev_direction is not None and direction != prev_direction:
                        direction_changes += 1
                        
                    prev_direction = direction
                
                # Calculate fluctuation ratio (direction changes / total changes)
                fluctuation_ratio = direction_changes / (len(changes) - 1) if len(changes) > 1 else 0
                
                # Consider it fluctuating if ratio is high (> 0.5)
                if fluctuation_ratio > 0.5 and len(changes) >= 3:
                    fluctuations.append(symptom)
                
                # Store individual symptom patterns
                pattern_features.loc[patient_id, f'{symptom}_sudden_increase'] = int(sudden_increase)
                pattern_features.loc[patient_id, f'{symptom}_sustained_increase'] = int(has_sustained_increase)
                pattern_features.loc[patient_id, f'{symptom}_fluctuation_ratio'] = fluctuation_ratio
            
            # Store aggregated pattern features
            pattern_features.loc[patient_id, 'n_sudden_increases'] = len(sudden_increases)
            pattern_features.loc[patient_id, 'n_sustained_increases'] = len(sustained_increases)
            pattern_features.loc[patient_id, 'n_fluctuating_symptoms'] = len(fluctuations)
            
            # Create binary flags for any occurrences
            pattern_features.loc[patient_id, 'has_sudden_increases'] = int(len(sudden_increases) > 0)
            pattern_features.loc[patient_id, 'has_sustained_increases'] = int(len(sustained_increases) > 0)
            pattern_features.loc[patient_id, 'has_fluctuating_symptoms'] = int(len(fluctuations) > 0)
            
            # Store symptom lists
            if len(sudden_increases) > 0:
                pattern_features.loc[patient_id, 'sudden_increase_symptoms'] = ','.join(sudden_increases)
            
            if len(sustained_increases) > 0:
                pattern_features.loc[patient_id, 'sustained_increase_symptoms'] = ','.join(sustained_increases)
            
            if len(fluctuations) > 0:
                pattern_features.loc[patient_id, 'fluctuating_symptoms'] = ','.join(fluctuations)
        
        return pattern_features

def create_trend_features(longitudinal_data, id_col='patient_id', time_col='days_since_baseline'):
    """
    Create features that capture symptom trends over time.
    
    Parameters:
    -----------
    longitudinal_data : DataFrame with multiple time points
        Longitudinal data with symptom measurements.
    id_col : str, default='patient_id'
        Column containing patient identifiers.
    time_col : str, default='days_since_baseline'
        Column containing time information.
    
    Returns:
    --------
    trend_features : DataFrame with trend features
        One row per patient with trend features.
    """
    # Create and apply the transformers
    trend_extractor = SymptomTrendExtractor(id_col=id_col, time_col=time_col)
    trend_features = trend_extractor.fit_transform(longitudinal_data)
    
    warning_extractor = EarlyWarningSigns()
    # Combine original most recent data with trend features
    most_recent = longitudinal_data.sort_values([id_col, time_col]).groupby(id_col).last()
    combined_data = pd.concat([most_recent, trend_features], axis=1)
    
    warning_features = warning_extractor.fit_transform(combined_data)
    
    # Add stability features if enough data points
    stability_extractor = SymptomStabilityExtractor(id_col=id_col, time_col=time_col)
    stability_features = stability_extractor.fit_transform(longitudinal_data)
    
    # Add pattern detection features
    pattern_detector = TemporalPatternDetector(id_col=id_col, time_col=time_col)
    pattern_features = pattern_detector.fit_transform(longitudinal_data)
    
    # Combine all features
    all_features = pd.concat([
        trend_features, 
        warning_features, 
        stability_features, 
        pattern_features
    ], axis=1)
    
    return all_features

def filter_longitudinal_data(data, id_col, time_col, time_window=None, min_observations=2):
    """
    Filter longitudinal data to include only patients with sufficient observations.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Longitudinal data with multiple observations per patient.
    id_col : str
        Column containing patient identifiers.
    time_col : str
        Column containing time information.
    time_window : tuple, default=None
        (min_time, max_time) window to include. If None, uses all data.
    min_observations : int, default=2
        Minimum number of observations required per patient.
    
    Returns:
    --------
    pandas DataFrame : Filtered data.
    """
    # Make a copy to avoid modifying the original
    filtered_data = data.copy()
    
    # Apply time window if specified
    if time_window is not None:
        min_time, max_time = time_window
        filtered_data = filtered_data[(filtered_data[time_col] >= min_time) & 
                                     (filtered_data[time_col] <= max_time)]
    
    # Count observations per patient
    observation_counts = filtered_data[id_col].value_counts()
    
    # Get patients with sufficient observations
    valid_patients = observation_counts[observation_counts >= min_observations].index
    
    # Filter to only include those patients
    filtered_data = filtered_data[filtered_data[id_col].isin(valid_patients)]
    
    return filtered_data

def create_early_warning_features(longitudinal_data, recent_data=None, id_col='patient_id', 
                                 time_col='days_since_baseline', warning_signs=None):
    """
    Create features specifically focused on early warning signs of relapse.
    
    Parameters:
    -----------
    longitudinal_data : pandas DataFrame
        Historical longitudinal data.
    recent_data : pandas DataFrame, default=None
        Most recent observations. If None, uses last observations from longitudinal_data.
    id_col : str, default='patient_id'
        Column containing patient identifiers.
    time_col : str, default='days_since_baseline'
        Column containing time information.
    warning_signs : dict, default=None
        Dictionary mapping warning sign names to related symptoms.
        If None, uses settings.EARLY_WARNING_SIGNS.
    
    Returns:
    --------
    pandas DataFrame : Early warning features with risk scores.
    """
    # Use configured warning signs if not provided
    if warning_signs is None:
        warning_signs = settings.EARLY_WARNING_SIGNS
    
    # Extract trend features
    trend_features = create_trend_features(
        longitudinal_data,
        id_col=id_col,
        time_col=time_col
    )
    
    # Get most recent data if not provided
    if recent_data is None:
        recent_data = longitudinal_data.sort_values([id_col, time_col]).groupby(id_col).last()
    
    # Combine recent data with trend features
    combined_data = pd.concat([recent_data, trend_features], axis=1)
    
    # Create early warning features
    warning_extractor = EarlyWarningSigns(warning_signs=warning_signs)
    warning_features = warning_extractor.fit_transform(combined_data)
    
    # Add pattern detection focused on warning signs
    pattern_detector = TemporalPatternDetector(
        symptom_cols=[symptom for symptoms in warning_signs.values() for symptom in symptoms],
        id_col=id_col,
        time_col=time_col,
        sudden_increase_threshold=1.5  # Lower threshold for warning signs
    )
    pattern_features = pattern_detector.fit_transform(longitudinal_data)
    
    # Combine warning features with pattern features
    early_warning_features = pd.concat([warning_features, pattern_features], axis=1)
    
    # Calculate overall risk score
    risk_indicators = [
        'early_warning_score',
        'has_sudden_increases',
        'has_sustained_increases'
    ]
    
    available_indicators = [col for col in risk_indicators if col in early_warning_features.columns]
    
    if available_indicators:
        # Weight the indicators based on clinical priority
        weights = {
            'early_warning_score': 2.0,
            'has_sudden_increases': 1.5,
            'has_sustained_increases': 1.0
        }
        
        # Calculate weighted sum
        early_warning_features['relapse_risk_score'] = sum(
            early_warning_features[indicator] * weights.get(indicator, 1.0)
            for indicator in available_indicators
        )
        
        # Normalize to 0-10 scale
        max_possible = sum(weights.get(indicator, 1.0) * 
                          (5 if indicator == 'early_warning_score' else 1)
                          for indicator in available_indicators)
        
        early_warning_features['relapse_risk_score'] = early_warning_features['relapse_risk_score'] / max_possible * 10
    
    return early_warning_features

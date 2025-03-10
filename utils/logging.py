"""
Logging utilities for FEP analysis.

This module provides consistent logging configuration and utilities
for the FEP analysis package.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Union, Dict, Any, List


def setup_logger(name: str, 
                level: int = logging.INFO, 
                log_file: Optional[str] = None,
                console: bool = True,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Parameters
    ----------
    name : str
        Name of the logger
    level : int, default=logging.INFO
        Logging level
    log_file : Optional[str], default=None
        Path to log file, if None, no file logging is set up
    console : bool, default=True
        Whether to log to console
    format_string : Optional[str], default=None
        Format string for logging. If None, a default format is used.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Define format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # Add file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, 
                         experiment_id: Optional[str] = None,
                         log_dir: str = './logs',
                         level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger for an experiment with appropriate file naming.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    experiment_id : Optional[str], default=None
        ID of the experiment. If None, current timestamp is used.
    log_dir : str, default='./logs'
        Directory to store log files
    level : int, default=logging.INFO
        Logging level
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Generate experiment_id if not provided
    if experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{experiment_name}_{timestamp}"
    
    # Create log filename
    log_file = os.path.join(log_dir, f"{experiment_id}.log")
    
    # Set up logger
    logger = setup_logger(experiment_id, level=level, log_file=log_file)
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name} (ID: {experiment_id})")
    
    return logger


class TimerContext:
    """
    Context manager for timing code blocks and logging the execution time.
    
    Example
    -------
    >>> logger = setup_logger('example')
    >>> with TimerContext('data_processing', logger):
    ...     # Code to be timed
    ...     process_data()
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        """
        Initialize timer context.
        
        Parameters
        ----------
        name : str
            Name of the operation being timed
        logger : Optional[logging.Logger], default=None
            Logger to use. If None, a new logger is created.
        level : int, default=logging.INFO
            Logging level for the timer messages
        """
        self.name = name
        
        if logger is None:
            self.logger = setup_logger('timer')
        else:
            self.logger = logger
            
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        """Start the timer when entering the context."""
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the execution time when exiting the context."""
        end_time = time.time()
        elapsed = end_time - self.start_time
        
        if exc_type is not None:
            # An exception occurred
            self.logger.log(self.level, f"{self.name} failed after {elapsed:.2f} seconds")
        else:
            # No exception
            self.logger.log(self.level, f"{self.name} completed in {elapsed:.2f} seconds")


def log_model_params(logger: logging.Logger, 
                    model_name: str, 
                    params: Dict[str, Any],
                    level: int = logging.INFO) -> None:
    """
    Log model parameters in a structured format.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    model_name : str
        Name of the model
    params : Dict[str, Any]
        Dictionary of model parameters
    level : int, default=logging.INFO
        Logging level
    """
    logger.log(level, f"Model: {model_name}")
    logger.log(level, "Parameters:")
    
    # Format parameters as string with indentation
    for key, value in params.items():
        logger.log(level, f"  {key}: {value}")


def log_metrics(logger: logging.Logger,
               metrics: Dict[str, float],
               prefix: str = "",
               level: int = logging.INFO) -> None:
    """
    Log evaluation metrics in a structured format.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    metrics : Dict[str, float]
        Dictionary of metric names and values
    prefix : str, default=""
        Prefix to add to the log message (e.g., "Training", "Validation")
    level : int, default=logging.INFO
        Logging level
    """
    if prefix:
        logger.log(level, f"{prefix} Metrics:")
    else:
        logger.log(level, "Metrics:")
    
    # Format metrics as string with indentation
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.log(level, f"  {metric_name}: {value:.4f}")
        else:
            logger.log(level, f"  {metric_name}: {value}")


def log_config(logger: logging.Logger,
              config: Dict[str, Any],
              header: str = "Configuration",
              level: int = logging.INFO) -> None:
    """
    Log configuration settings.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    config : Dict[str, Any]
        Dictionary of configuration settings
    header : str, default="Configuration"
        Header for the configuration section
    level : int, default=logging.INFO
        Logging level
    """
    logger.log(level, f"{header}:")
    
    # Helper function to format nested dictionaries
    def _log_dict(d, indent=2):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.log(level, " " * indent + f"{key}:")
                _log_dict(value, indent + 2)
            else:
                logger.log(level, " " * indent + f"{key}: {value}")
    
    _log_dict(config)


def create_experiment_summary(metrics: Dict[str, float],
                             params: Dict[str, Any],
                             runtime: float,
                             experiment_name: str,
                             timestamp: Optional[str] = None,
                             additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a structured summary of an experiment for easy logging or storage.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics
    params : Dict[str, Any]
        Dictionary of model parameters
    runtime : float
        Runtime in seconds
    experiment_name : str
        Name of the experiment
    timestamp : Optional[str], default=None
        Timestamp for the experiment. If None, current time is used.
    additional_info : Optional[Dict[str, Any]], default=None
        Additional information to include in the summary
        
    Returns
    -------
    Dict[str, Any]
        Structured summary of the experiment
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "runtime_seconds": runtime,
        "metrics": metrics,
        "parameters": params
    }
    
    if additional_info is not None:
        summary["additional_info"] = additional_info
    
    return summary


class MultiLevelAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds hierarchical context to log messages.
    
    This adapter allows adding multiple levels of context (e.g., component,
    operation, step) to log messages for better organization.
    
    Example
    -------
    >>> logger = setup_logger('example')
    >>> adapter = MultiLevelAdapter(logger, {'component': 'DataProcessor'})
    >>> adapter.info('Loading data')  # Logs "[DataProcessor] Loading data"
    >>> adapter.add_context('operation', 'Normalization')  # Adds second level
    >>> adapter.info('Normalizing features')  # Logs "[DataProcessor][Normalization] Normalizing features"
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, str] = None):
        """
        Initialize adapter with context.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance to adapt
        extra : Dict[str, str], default=None
            Initial context dictionary
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process log message by adding context prefixes."""
        if self.extra:
            # Create a prefix from all context levels
            prefix = ''.join(f"[{v}]" for k, v in self.extra.items())
            msg = f"{prefix} {msg}"
        
        return msg, kwargs
    
    def add_context(self, key: str, value: str) -> None:
        """
        Add a new context level.
        
        Parameters
        ----------
        key : str
            Context key (e.g., 'component', 'operation')
        value : str
            Context value (e.g., 'DataProcessor', 'Normalization')
        """
        self.extra[key] = value
    
    def remove_context(self, key: str) -> None:
        """
        Remove a context level.
        
        Parameters
        ----------
        key : str
            Context key to remove
        """
        if key in self.extra:
            del self.extra[key]
    
    def clear_context(self) -> None:
        """Remove all context levels."""
        self.extra.clear()

"""
Results loading utilities for QLLM evaluation visualization.

This module provides functions for loading and preprocessing
evaluation results for visualization.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a file.
    
    Args:
        results_path: Path to the evaluation results JSON file
        
    Returns:
        Dictionary containing evaluation results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def find_latest_results(base_dir: str, pattern: str = "evaluation_results_*.json") -> str:
    """
    Find the latest evaluation results file in a directory.
    
    Args:
        base_dir: Base directory to search
        pattern: Filename pattern to match
        
    Returns:
        Path to the latest results file, or None if not found
    """
    import glob
    import os
    
    # Get all matching files
    import fnmatch
    
    result_files = []
    for file in os.listdir(base_dir):
        if fnmatch.fnmatch(file, pattern):
            result_files.append(os.path.join(base_dir, file))
    
    if not result_files:
        return None
    
    # Find the latest file by modification time
    latest_file = max(result_files, key=os.path.getmtime)
    
    return latest_file


def extract_metric_data(
    results: Dict[str, Any],
    metric_name: str
) -> Tuple[List[str], List[float], List[float]]:
    """
    Extract data for a specific metric from results.
    
    Args:
        results: Evaluation results dictionary
        metric_name: Name of the metric to extract
        
    Returns:
        Tuple of (config_names, values, errors)
    """
    configs = []
    values = []
    errors = []
    
    for config_name, config_results in results.items():
        # Skip non-configuration entries
        if config_name == 'ablation_studies':
            continue
            
        if "metrics" in config_results and metric_name in config_results["metrics"]:
            metric_data = config_results["metrics"][metric_name]
            
            # Handle different metric formats
            if isinstance(metric_data, dict) and "mean" in metric_data:
                value = metric_data["mean"]
                error = metric_data.get("std", 0)
            elif isinstance(metric_data, dict) and "value" in metric_data:
                value = metric_data["value"]
                error = 0
            elif isinstance(metric_data, (int, float)):
                value = metric_data
                error = 0
            else:
                # Skip complex metrics that can't be easily plotted
                continue
                
            configs.append(config_name)
            values.append(value)
            errors.append(error)
    
    return configs, values, errors


def extract_ablation_data(
    results: Dict[str, Any],
    metric_name: str
) -> Tuple[List[str], List[float]]:
    """
    Extract ablation study data for a specific metric.
    
    Args:
        results: Evaluation results dictionary
        metric_name: Name of the metric to extract
        
    Returns:
        Tuple of (config_names, values)
    """
    if 'ablation_studies' not in results:
        return [], []
    
    ablation_results = results['ablation_studies']
    
    configs = []
    values = []
    
    for config_name, config_results in ablation_results.items():
        if "metrics" in config_results and metric_name in config_results["metrics"]:
            metric_data = config_results["metrics"][metric_name]
            
            # Handle different metric formats
            if isinstance(metric_data, dict) and "mean" in metric_data:
                value = metric_data["mean"]
            elif isinstance(metric_data, dict) and "value" in metric_data:
                value = metric_data["value"]
            elif isinstance(metric_data, (int, float)):
                value = metric_data
            else:
                # Skip complex metrics that can't be easily plotted
                continue
                
            configs.append(config_name)
            values.append(value)
    
    return configs, values


def convert_to_dataframe(
    results: Dict[str, Any],
    metrics: List[str]
) -> pd.DataFrame:
    """
    Convert results to a DataFrame for easier analysis.
    
    Args:
        results: Evaluation results dictionary
        metrics: List of metrics to include
        
    Returns:
        DataFrame with results
    """
    data = []
    
    for config_name, config_results in results.items():
        # Skip non-configuration entries
        if config_name == 'ablation_studies':
            continue
            
        if "metrics" not in config_results:
            continue
            
        row = {"config": config_name}
        
        for metric in metrics:
            if metric in config_results["metrics"]:
                metric_data = config_results["metrics"][metric]
                
                # Handle different metric formats
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    row[metric] = metric_data["mean"]
                elif isinstance(metric_data, dict) and "value" in metric_data:
                    row[metric] = metric_data["value"]
                elif isinstance(metric_data, (int, float)):
                    row[metric] = metric_data
            
        data.append(row)
    
    return pd.DataFrame(data)


def get_extension_metrics(
    results: Dict[str, Any],
    extension: str
) -> Dict[str, Any]:
    """
    Extract extension-specific metrics.
    
    Args:
        results: Evaluation results dictionary
        extension: Name of the extension
        
    Returns:
        Dictionary of extension-specific metrics
    """
    if extension not in results:
        return {}
    
    ext_results = results[extension]
    ext_key = f"{extension}_metrics"
    
    if ext_key not in ext_results:
        return {}
    
    return ext_results[ext_key]
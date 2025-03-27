"""
Basic metric plotting utilities for QLLM evaluation visualization.

This module provides functions for creating basic visualizations
of evaluation metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union

from src.evaluation.visualization.results_loader import extract_metric_data


def plot_metric_comparison(
    results: Dict[str, Any],
    metric: str,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
) -> None:
    """
    Create a bar chart comparing a metric across different configurations.
    
    Args:
        results: Evaluation results dictionary
        metric: Name of the metric to plot
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    plt.figure(figsize=figsize)
    
    # Extract data for the metric
    configs, values, errors = extract_metric_data(results, metric)
    
    if not configs:
        print(f"No data found for metric: {metric}")
        return
    
    # Create bar chart
    bars = plt.bar(configs, values, yerr=errors)
    
    # Customize appearance
    plt.title(title or f"{metric.replace('_', ' ').title()} Comparison")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(values),
            f'{height:.2f}',
            ha='center', va='bottom', 
            rotation=0
        )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_parameter_breakdown(
    results: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> None:
    """
    Create a visualization of parameter breakdown by extension.
    
    Args:
        results: Evaluation results dictionary
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    # Find configurations with parameter breakdowns
    param_data = {}
    
    for config_name, config_results in results.items():
        if "metrics" in config_results and "parameter_efficiency" in config_results["metrics"]:
            efficiency_data = config_results["metrics"]["parameter_efficiency"]
            
            if "extension_params" in efficiency_data:
                param_data[config_name] = efficiency_data["extension_params"]
    
    if not param_data:
        print("No parameter breakdown data found")
        return
    
    # Create a DataFrame for easier plotting
    extensions = set()
    for params in param_data.values():
        extensions.update(params.keys())
    
    extensions = sorted(list(extensions))
    
    data = []
    for config_name, params in param_data.items():
        row = [config_name]
        for ext in extensions:
            row.append(params.get(ext, 0))
        data.append(row)
    
    df = pd.DataFrame(data, columns=["config"] + extensions)
    df = df.set_index("config")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    ax = df.plot(kind="bar", stacked=True, figsize=figsize)
    
    plt.title(title or "Parameter Breakdown by Extension")
    plt.ylabel("Number of Parameters")
    plt.xlabel("Configuration")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.legend(title="Extension", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add total parameter count on top of each bar
    for i, total in enumerate(df.sum(axis=1)):
        ax.text(
            i, 
            total + 0.02 * df.sum(axis=1).max(),
            f'{total:,.0f}',
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_time_series(
    results_list: List[Dict[str, Any]],
    timestamps: List[str],
    metric: str,
    config: str,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
) -> None:
    """
    Create a time series plot of a metric across multiple evaluation runs.
    
    Args:
        results_list: List of evaluation results dictionaries
        timestamps: List of timestamps corresponding to results
        metric: Name of the metric to plot
        config: Configuration to plot
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    plt.figure(figsize=figsize)
    
    # Extract values for the metric and configuration
    values = []
    
    for results in results_list:
        if config in results and "metrics" in results[config] and metric in results[config]["metrics"]:
            metric_data = results[config]["metrics"][metric]
            
            # Handle different metric formats
            if isinstance(metric_data, dict) and "mean" in metric_data:
                value = metric_data["mean"]
            elif isinstance(metric_data, dict) and "value" in metric_data:
                value = metric_data["value"]
            elif isinstance(metric_data, (int, float)):
                value = metric_data
            else:
                value = None
            
            values.append(value)
        else:
            values.append(None)
    
    # Filter out None values
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_values = [values[i] for i in valid_indices]
    valid_timestamps = [timestamps[i] for i in valid_indices]
    
    if not valid_values:
        print(f"No data found for metric: {metric}, config: {config}")
        return
    
    # Create line plot
    plt.plot(valid_timestamps, valid_values, marker='o', linestyle='-')
    
    # Customize appearance
    plt.title(title or f"{metric.replace('_', ' ').title()} Over Time - {config}")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel("Timestamp")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on points
    for i, (timestamp, value) in enumerate(zip(valid_timestamps, valid_values)):
        plt.text(
            timestamp,
            value + 0.02 * max(valid_values),
            f'{value:.2f}',
            ha='center', va='bottom',
            rotation=0
        )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_metric_heatmap(
    results: Dict[str, Any],
    metrics: List[str],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None
) -> None:
    """
    Create a heatmap showing multiple metrics across configurations.
    
    Args:
        results: Evaluation results dictionary
        metrics: List of metrics to include
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    # Prepare data for heatmap
    configs = []
    data = []
    
    for config_name, config_results in results.items():
        if "metrics" not in config_results:
            continue
            
        # Skip non-configuration entries
        if config_name == 'ablation_studies':
            continue
            
        configs.append(config_name)
        
        row = []
        for metric in metrics:
            if metric in config_results["metrics"]:
                metric_data = config_results["metrics"][metric]
                
                # Handle different metric formats
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    value = metric_data["mean"]
                elif isinstance(metric_data, dict) and "value" in metric_data:
                    value = metric_data["value"]
                elif isinstance(metric_data, (int, float)):
                    value = metric_data
                else:
                    value = np.nan
            else:
                value = np.nan
                
            row.append(value)
        
        data.append(row)
    
    if not data:
        print("No data found for the specified metrics")
        return
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Normalize each metric column separately
    normalized_data = np.zeros_like(data_array)
    for i in range(data_array.shape[1]):
        col = data_array[:, i]
        valid_indices = ~np.isnan(col)
        if np.any(valid_indices):
            min_val = np.min(col[valid_indices])
            max_val = np.max(col[valid_indices])
            if max_val > min_val:
                normalized_data[:, i][valid_indices] = (col[valid_indices] - min_val) / (max_val - min_val)
            else:
                normalized_data[:, i][valid_indices] = 0.5
    
    # Create heatmap
    ax = plt.gca()
    im = ax.imshow(normalized_data, cmap='viridis')
    
    # Set labels
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticklabels(configs)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized Value')
    
    # Add values to cells
    for i in range(len(configs)):
        for j in range(len(metrics)):
            if not np.isnan(data_array[i, j]):
                text = f"{data_array[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="w")
    
    plt.title(title or "Metrics Comparison Across Configurations")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
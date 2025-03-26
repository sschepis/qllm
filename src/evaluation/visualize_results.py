"""
Visualization utilities for QLLM evaluation results.

This module provides functions for visualizing and analyzing results
from the comprehensive evaluation suite.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    configs = []
    values = []
    errors = []
    
    for config_name, config_results in results.items():
        # Skip non-configuration entries
        if config_name == 'ablation_studies':
            continue
            
        if "metrics" in config_results and metric in config_results["metrics"]:
            metric_data = config_results["metrics"][metric]
            
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


def plot_ablation_study(
    results: Dict[str, Any],
    metric: str,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7),
    title: Optional[str] = None
) -> None:
    """
    Create a visualization of ablation study results for a specific metric.
    
    Args:
        results: Evaluation results dictionary
        metric: Name of the metric to visualize
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    if 'ablation_studies' not in results:
        print("No ablation studies found in results")
        return
    
    ablation_results = results['ablation_studies']
    
    # Extract data for the metric
    configs = []
    values = []
    
    for config_name, config_results in ablation_results.items():
        if "metrics" in config_results and metric in config_results["metrics"]:
            metric_data = config_results["metrics"][metric]
            
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
    
    # Sort configurations in a meaningful order
    # Order: none, individual extensions, combinations, all
    individual_exts = ["multimodal", "memory", "quantum"]
    pairs = [f"{ext1}+{ext2}" for i, ext1 in enumerate(individual_exts) 
             for ext2 in individual_exts[i+1:]]
    
    ordered_configs = []
    # First 'none'
    if "none" in configs:
        ordered_configs.append("none")
    # Then individual extensions
    for ext in individual_exts:
        if ext in configs:
            ordered_configs.append(ext)
    # Then pairs
    for pair in pairs:
        if pair in configs:
            ordered_configs.append(pair)
    # Finally 'all'
    if "all" in configs:
        ordered_configs.append("all")
    
    # Add any remaining configs
    for config in configs:
        if config not in ordered_configs:
            ordered_configs.append(config)
    
    # Reorder data
    ordered_values = []
    for config in ordered_configs:
        idx = configs.index(config)
        ordered_values.append(values[idx])
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Use different colors for different types of configurations
    colors = ['lightgray']  # none
    colors += ['lightblue', 'lightgreen', 'salmon']  # individual exts
    colors += ['orange', 'purple', 'teal']  # pairs
    colors += ['darkred']  # all
    
    # Ensure we have enough colors
    while len(colors) < len(ordered_configs):
        colors.append('gray')
    
    bars = plt.bar(ordered_configs, ordered_values, color=colors[:len(ordered_configs)])
    
    # Customize appearance
    plt.title(title or f"Ablation Study: {metric.replace('_', ' ').title()}")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(ordered_values),
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


def plot_extension_metrics(
    results: Dict[str, Any],
    extension: str,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None
) -> None:
    """
    Create a visualization of extension-specific metrics.
    
    Args:
        results: Evaluation results dictionary
        extension: Name of the extension to visualize
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    if extension not in results:
        print(f"No results found for {extension} extension")
        return
    
    ext_results = results[extension]
    ext_key = f"{extension}_metrics"
    
    if ext_key not in ext_results:
        print(f"No extension-specific metrics found for {extension}")
        return
    
    ext_metrics = ext_results[ext_key]
    
    # Filter out non-numeric or complex metrics
    plotable_metrics = {}
    
    for metric_name, metric_value in ext_metrics.items():
        if isinstance(metric_value, (int, float)):
            plotable_metrics[metric_name] = metric_value
        elif isinstance(metric_value, dict) and any(key in metric_value for key in ["mean", "value"]):
            value = metric_value.get("mean", metric_value.get("value"))
            if isinstance(value, (int, float)):
                plotable_metrics[metric_name] = value
    
    if not plotable_metrics:
        print(f"No plottable metrics found for {extension}")
        return
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    metrics = list(plotable_metrics.keys())
    values = list(plotable_metrics.values())
    
    # Normalize values to 0-1 scale for better visualization
    min_val = min(values)
    max_val = max(values)
    if max_val > min_val:
        norm_values = [(v - min_val) / (max_val - min_val) for v in values]
    else:
        norm_values = [0.5 for _ in values]
    
    # Create a polar plot for extension metrics
    ax = plt.subplot(111, polar=True)
    
    # Compute angles for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    # Add values and close the plot
    norm_values += norm_values[:1]
    
    # Plot data
    ax.plot(angles, norm_values, 'o-', linewidth=2)
    ax.fill(angles, norm_values, alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=8)
    
    # Add original values as annotations
    for i, (angle, value, norm) in enumerate(zip(angles[:-1], values, norm_values[:-1])):
        ha = 'left' if angle < np.pi else 'right'
        ax.annotate(
            f"{value:.2f}",
            xy=(angle, norm + 0.1),
            ha=ha,
            va='center'
        )
    
    plt.title(title or f"{extension.title()} Extension Metrics")
    
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


def plot_efficiency_metrics(
    results: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None
) -> None:
    """
    Create a multi-faceted visualization of efficiency metrics.
    
    Args:
        results: Evaluation results dictionary
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height)
        title: Optional custom title
    """
    # Extract efficiency metrics from results
    configs = []
    memory_usage = []
    inference_speed = []
    parameter_counts = []
    
    for config_name, config_results in results.items():
        if "metrics" not in config_results:
            continue
            
        metrics = config_results["metrics"]
        
        # Skip if we don't have all the metrics
        if not all(m in metrics for m in ["memory_usage", "inference_speed", "parameter_efficiency"]):
            continue
            
        configs.append(config_name)
        
        # Extract memory usage
        if "model_size_mb" in metrics["memory_usage"]:
            memory_usage.append(metrics["memory_usage"]["model_size_mb"])
        else:
            memory_usage.append(0)
            
        # Extract inference speed
        if "tokens_per_second" in metrics["inference_speed"]:
            inference_speed.append(metrics["inference_speed"]["tokens_per_second"])
        else:
            inference_speed.append(0)
            
        # Extract parameter count
        if "effective_params" in metrics["parameter_efficiency"]:
            parameter_counts.append(metrics["parameter_efficiency"]["effective_params"])
        else:
            parameter_counts.append(0)
    
    if not configs:
        print("Insufficient data for efficiency visualization")
        return
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot memory usage
    axes[0].bar(configs, memory_usage)
    axes[0].set_title("Memory Usage (MB)")
    axes[0].set_ylabel("Memory (MB)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot inference speed
    axes[1].bar(configs, inference_speed, color='orange')
    axes[1].set_title("Inference Speed (Tokens/Second)")
    axes[1].set_ylabel("Tokens/Second")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot parameter count
    axes[2].bar(configs, parameter_counts, color='green')
    axes[2].set_title("Effective Parameter Count")
    axes[2].set_ylabel("Parameters")
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Set overall title
    fig.suptitle(title or "Efficiency Metrics Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def create_summary_dashboard(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = "dashboard"
) -> None:
    """
    Create a comprehensive dashboard of visualizations.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save dashboard visualizations
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List of metrics to visualize
    metrics = [
        "perplexity",
        "parameter_efficiency",
        "memory_usage",
        "inference_speed",
        "generation_diversity"
    ]
    
    # Generate metric comparison plots
    for metric in metrics:
        output_file = os.path.join(output_dir, f"{prefix}_{metric}_comparison.png")
        try:
            plot_metric_comparison(results, metric, output_file)
        except Exception as e:
            print(f"Error plotting {metric} comparison: {e}")
    
    # Generate extension-specific plots
    for extension in ["multimodal", "memory", "quantum"]:
        if extension in results:
            output_file = os.path.join(output_dir, f"{prefix}_{extension}_metrics.png")
            try:
                plot_extension_metrics(results, extension, output_file)
            except Exception as e:
                print(f"Error plotting {extension} metrics: {e}")
    
    # Generate ablation study plot
    if "ablation_studies" in results:
        for metric in metrics:
            output_file = os.path.join(output_dir, f"{prefix}_ablation_{metric}.png")
            try:
                plot_ablation_study(results, metric, output_file)
            except Exception as e:
                print(f"Error plotting ablation study for {metric}: {e}")
    
    # Generate parameter breakdown plot
    output_file = os.path.join(output_dir, f"{prefix}_parameter_breakdown.png")
    try:
        plot_parameter_breakdown(results, output_file)
    except Exception as e:
        print(f"Error plotting parameter breakdown: {e}")
    
    # Generate efficiency metrics plot
    output_file = os.path.join(output_dir, f"{prefix}_efficiency_metrics.png")
    try:
        plot_efficiency_metrics(results, output_file)
    except Exception as e:
        print(f"Error plotting efficiency metrics: {e}")
    
    print(f"Dashboard visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize QLLM evaluation results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_visualizations", help="Output directory")
    parser.add_argument("--prefix", type=str, default="dashboard", help="Filename prefix for visualizations")
    args = parser.parse_args()
    
    # Load results
    results = load_evaluation_results(args.results)
    
    # Create dashboard
    create_summary_dashboard(results, args.output_dir, args.prefix)
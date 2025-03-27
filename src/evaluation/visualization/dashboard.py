"""
Dashboard creation for QLLM evaluation visualization.

This module provides functionality for creating comprehensive dashboards
summarizing evaluation results.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from src.evaluation.visualization.metric_plots import (
    plot_metric_comparison,
    plot_parameter_breakdown
)
from src.evaluation.visualization.comparison_plots import (
    plot_ablation_study,
    plot_efficiency_metrics,
    plot_extension_metrics
)


def create_summary_dashboard(
    results: Dict[str, Any],
    output_dir: str,
    basic_metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> None:
    """
    Create a comprehensive summary dashboard with multiple plots.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save dashboard plots
        basic_metrics: List of basic metrics to include in the dashboard
        figsize: Figure size as (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use default metrics if none provided
    if basic_metrics is None:
        basic_metrics = ["perplexity", "parameter_efficiency", "memory_usage", "inference_speed"]
    
    # Create individual plot files
    metric_files = []
    
    for metric in basic_metrics:
        output_file = os.path.join(output_dir, f"{metric}.png")
        try:
            plot_metric_comparison(results, metric, output_file)
            metric_files.append(output_file)
        except Exception as e:
            print(f"Error creating {metric} plot: {str(e)}")
    
    # Create parameter breakdown
    param_file = os.path.join(output_dir, "parameter_breakdown.png")
    try:
        plot_parameter_breakdown(results, param_file)
    except Exception as e:
        print(f"Error creating parameter breakdown plot: {str(e)}")
        param_file = None
    
    # Create ablation study plots if available
    ablation_files = []
    if 'ablation_studies' in results:
        for metric in basic_metrics:
            output_file = os.path.join(output_dir, f"ablation_{metric}.png")
            try:
                plot_ablation_study(results, metric, output_file)
                ablation_files.append(output_file)
            except Exception as e:
                print(f"Error creating ablation plot for {metric}: {str(e)}")
    
    # Create efficiency metrics plot
    efficiency_file = os.path.join(output_dir, "efficiency_metrics.png")
    try:
        plot_efficiency_metrics(results, efficiency_file)
    except Exception as e:
        print(f"Error creating efficiency metrics plot: {str(e)}")
        efficiency_file = None
    
    # Create extension-specific plots
    extension_files = []
    for ext in ["multimodal", "memory", "quantum"]:
        if ext in results:
            output_file = os.path.join(output_dir, f"{ext}_metrics.png")
            try:
                plot_extension_metrics(results, ext, output_file)
                extension_files.append((ext, output_file))
            except Exception as e:
                print(f"Error creating {ext} extension plot: {str(e)}")
    
    # Create summary markdown file
    create_summary_markdown(
        results,
        metric_files,
        ablation_files,
        param_file,
        efficiency_file,
        extension_files,
        os.path.join(output_dir, "summary.md")
    )
    
    print(f"Dashboard created in {output_dir}")


def create_summary_markdown(
    results: Dict[str, Any],
    metric_files: List[str],
    ablation_files: List[str],
    param_file: Optional[str],
    efficiency_file: Optional[str],
    extension_files: List[Tuple[str, str]],
    output_file: str
) -> None:
    """
    Create a summary markdown file with links to all plots.
    
    Args:
        results: Evaluation results dictionary
        metric_files: List of metric plot files
        ablation_files: List of ablation study plot files
        param_file: Path to parameter breakdown plot file
        efficiency_file: Path to efficiency metrics plot file
        extension_files: List of (extension_name, plot_file) tuples
        output_file: Path to save markdown file
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("# QLLM Evaluation Summary\n\n")
        
        # Add timestamp if available
        if "timestamp" in results:
            f.write(f"**Evaluation Time:** {results['timestamp']}\n\n")
        
        # Basic metrics
        if metric_files:
            f.write("## Basic Metrics\n\n")
            for file_path in metric_files:
                metric_name = os.path.basename(file_path).replace(".png", "")
                f.write(f"### {metric_name.replace('_', ' ').title()}\n\n")
                f.write(f"![{metric_name}]({os.path.basename(file_path)})\n\n")
        
        # Parameter breakdown
        if param_file:
            f.write("## Parameter Breakdown\n\n")
            f.write(f"![Parameter Breakdown]({os.path.basename(param_file)})\n\n")
        
        # Efficiency metrics
        if efficiency_file:
            f.write("## Efficiency Metrics\n\n")
            f.write(f"![Efficiency Metrics]({os.path.basename(efficiency_file)})\n\n")
        
        # Ablation studies
        if ablation_files:
            f.write("## Ablation Studies\n\n")
            for file_path in ablation_files:
                metric_name = os.path.basename(file_path).replace("ablation_", "").replace(".png", "")
                f.write(f"### {metric_name.replace('_', ' ').title()} Ablation\n\n")
                f.write(f"![{metric_name} Ablation]({os.path.basename(file_path)})\n\n")
        
        # Extension-specific metrics
        if extension_files:
            f.write("## Extension-Specific Metrics\n\n")
            for ext_name, file_path in extension_files:
                f.write(f"### {ext_name.title()} Extension\n\n")
                f.write(f"![{ext_name.title()} Metrics]({os.path.basename(file_path)})\n\n")
        
        # Add numerical summary
        f.write("## Numerical Summary\n\n")
        f.write("| Configuration | ")
        
        # Get all metrics
        all_metrics = set()
        for config_name, config_results in results.items():
            if config_name == 'ablation_studies' or "metrics" not in config_results:
                continue
                
            for metric in config_results["metrics"].keys():
                all_metrics.add(metric)
        
        # Sort metrics
        sorted_metrics = sorted(list(all_metrics))
        
        # Write header row
        for metric in sorted_metrics:
            f.write(f"{metric.replace('_', ' ').title()} | ")
        f.write("\n")
        
        # Write separator row
        f.write("| --- |")
        for _ in sorted_metrics:
            f.write(" --- |")
        f.write("\n")
        
        # Write data rows
        for config_name, config_results in results.items():
            if config_name == 'ablation_studies' or "metrics" not in config_results:
                continue
                
            f.write(f"| {config_name} | ")
            
            for metric in sorted_metrics:
                if metric in config_results["metrics"]:
                    metric_data = config_results["metrics"][metric]
                    
                    # Handle different metric formats
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        value = f"{metric_data['mean']:.3f}"
                    elif isinstance(metric_data, dict) and "value" in metric_data:
                        value = f"{metric_data['value']:.3f}"
                    elif isinstance(metric_data, (int, float)):
                        value = f"{metric_data:.3f}"
                    else:
                        value = "N/A"
                else:
                    value = "N/A"
                    
                f.write(f"{value} | ")
            f.write("\n")


def create_interactive_dashboard(
    results: Dict[str, Any],
    output_file: str,
    basic_metrics: Optional[List[str]] = None
) -> None:
    """
    Create an interactive HTML dashboard with all evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_file: Path to save HTML dashboard
        basic_metrics: List of basic metrics to include in the dashboard
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("Error: plotly package not found. Cannot create interactive dashboard.")
        print("Install with: pip install plotly")
        return
    
    # Use default metrics if none provided
    if basic_metrics is None:
        basic_metrics = ["perplexity", "parameter_efficiency", "memory_usage", "inference_speed"]
    
    # Create HTML content
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>QLLM Evaluation Dashboard</title>",
        "    <meta charset=\"utf-8\">",
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        .dashboard-container { display: flex; flex-wrap: wrap; }",
        "        .plot-container { width: 800px; height: 500px; margin: 10px; }",
        "        h1, h2 { color: #333; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>QLLM Evaluation Dashboard</h1>"
    ]
    
    # Add timestamp if available
    if "timestamp" in results:
        html_content.append(f"    <p><strong>Evaluation Time:</strong> {results['timestamp']}</p>")
    
    # Create metric plots
    for metric in basic_metrics:
        # Extract data
        configs = []
        values = []
        
        for config_name, config_results in results.items():
            if config_name == 'ablation_studies':
                continue
                
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
                    continue
                    
                configs.append(config_name)
                values.append(value)
        
        if configs:
            # Create plot
            fig = px.bar(
                x=configs,
                y=values,
                labels={"x": "Configuration", "y": metric.replace("_", " ").title()},
                title=f"{metric.replace('_', ' ').title()} Comparison"
            )
            
            # Add to HTML
            plot_div = pio.to_html(fig, full_html=False)
            html_content.append(f"    <h2>{metric.replace('_', ' ').title()}</h2>")
            html_content.append(f"    <div class=\"plot-container\">{plot_div}</div>")
    
    # Add numerical summary table
    html_content.append("    <h2>Numerical Summary</h2>")
    html_content.append("    <table>")
    html_content.append("        <tr><th>Configuration</th>")
    
    # Get all metrics
    all_metrics = set()
    for config_name, config_results in results.items():
        if config_name == 'ablation_studies' or "metrics" not in config_results:
            continue
            
        for metric in config_results["metrics"].keys():
            all_metrics.add(metric)
    
    # Sort metrics
    sorted_metrics = sorted(list(all_metrics))
    
    # Write header row
    for metric in sorted_metrics:
        html_content.append(f"            <th>{metric.replace('_', ' ').title()}</th>")
    html_content.append("        </tr>")
    
    # Write data rows
    for config_name, config_results in results.items():
        if config_name == 'ablation_studies' or "metrics" not in config_results:
            continue
            
        html_content.append(f"        <tr><td>{config_name}</td>")
        
        for metric in sorted_metrics:
            if metric in config_results["metrics"]:
                metric_data = config_results["metrics"][metric]
                
                # Handle different metric formats
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    value = f"{metric_data['mean']:.3f}"
                elif isinstance(metric_data, dict) and "value" in metric_data:
                    value = f"{metric_data['value']:.3f}"
                elif isinstance(metric_data, (int, float)):
                    value = f"{metric_data:.3f}"
                else:
                    value = "N/A"
            else:
                value = "N/A"
                
            html_content.append(f"            <td>{value}</td>")
        html_content.append("        </tr>")
    
    html_content.append("    </table>")
    
    # Close HTML
    html_content.append("</body>")
    html_content.append("</html>")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(html_content))
    
    print(f"Interactive dashboard saved to {output_file}")
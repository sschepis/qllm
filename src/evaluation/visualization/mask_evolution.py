"""
Visualization tools for quantum mask evolution patterns in QLLM models.

This module provides specialized visualizations for tracking and analyzing
how quantum masks evolve over time, helping to understand the dynamics of
the structured sparsity patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union


def plot_mask_evolution_heatmap(
    mask_history: List[Dict[str, np.ndarray]],
    layer_name: str,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a heatmap visualization of how a specific layer's mask evolves over time.
    
    Args:
        mask_history: List of mask snapshots at different evolution steps
        layer_name: Name of the layer to visualize
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not mask_history or layer_name not in mask_history[0]:
        print(f"No mask history data found for layer: {layer_name}")
        return
    
    # Extract masks for the specified layer
    layer_masks = [snapshot[layer_name] for snapshot in mask_history if layer_name in snapshot]
    
    if not layer_masks:
        print(f"No masks found for layer: {layer_name}")
        return
    
    # Calculate mask changes between steps
    mask_changes = []
    for i in range(1, len(layer_masks)):
        # Element-wise XOR to get changes (1 where masks differ)
        change = (layer_masks[i] != layer_masks[i-1]).astype(np.float32)
        mask_changes.append(change)
    
    if not mask_changes:
        print("Not enough history to visualize evolution")
        return
    
    # Create heatmap of cumulative changes
    cumulative_changes = np.sum(mask_changes, axis=0)
    
    plt.figure(figsize=figsize)
    
    # Plot the heatmap
    ax = sns.heatmap(
        cumulative_changes,
        cmap="viridis",
        cbar_kws={"label": "Number of Changes"}
    )
    
    # Set title and labels
    plt.title(title or f"Mask Evolution Heatmap for {layer_name}")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Mask evolution heatmap saved to {output_file}")
    else:
        plt.show()


def create_mask_evolution_animation(
    mask_history: List[Dict[str, np.ndarray]],
    layer_name: str,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    interval: int = 200  # ms between frames
) -> None:
    """
    Create an animation showing how a mask evolves over time.
    
    Args:
        mask_history: List of mask snapshots at different evolution steps
        layer_name: Name of the layer to visualize
        output_file: Optional path to save the animation (as .gif or .mp4)
        title: Optional custom title
        figsize: Figure size as (width, height)
        interval: Time interval between frames in milliseconds
    """
    if not mask_history or layer_name not in mask_history[0]:
        print(f"No mask history data found for layer: {layer_name}")
        return
    
    # Extract masks for the specified layer
    layer_masks = [snapshot[layer_name] for snapshot in mask_history if layer_name in snapshot]
    
    if not layer_masks:
        print(f"No masks found for layer: {layer_name}")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # First frame
    im = ax.imshow(
        layer_masks[0],
        cmap='viridis',
        interpolation='nearest',
        aspect='auto'
    )
    
    # Title and colorbar
    plt.title(title or f"Mask Evolution for {layer_name}")
    plt.colorbar(im, ax=ax, label="Mask Value")
    
    # Updated function for animation
    def update(frame):
        im.set_array(layer_masks[frame])
        ax.set_title(f"{title or f'Mask Evolution for {layer_name}'} - Step {frame+1}/{len(layer_masks)}")
        return [im]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(layer_masks),
        interval=interval,
        blit=True
    )
    
    plt.tight_layout()
    
    if output_file:
        if output_file.endswith('.gif'):
            ani.save(output_file, writer='pillow', fps=1000/interval)
        elif output_file.endswith('.mp4'):
            ani.save(output_file, writer='ffmpeg', fps=1000/interval)
        else:
            print("Unsupported output format. Use .gif or .mp4")
            return
        print(f"Mask evolution animation saved to {output_file}")
    else:
        plt.show()


def plot_sparsity_evolution(
    mask_history: List[Dict[str, np.ndarray]],
    layer_names: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot how sparsity evolves over time for multiple layers.
    
    Args:
        mask_history: List of mask snapshots at different evolution steps
        layer_names: Optional list of layer names to include (all if None)
        output_file: Optional path to save the visualization
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    if not mask_history:
        print("No mask history data found")
        return
    
    # If no layer names specified, use all from first snapshot
    if layer_names is None:
        layer_names = list(mask_history[0].keys())
    
    # Calculate sparsity for each layer at each step
    sparsity_data = {layer: [] for layer in layer_names if layer in mask_history[0]}
    
    for snapshot in mask_history:
        for layer in sparsity_data:
            if layer in snapshot:
                # Sparsity = proportion of zeros
                sparsity = 1.0 - np.mean(snapshot[layer])
                sparsity_data[layer].append(sparsity)
    
    # Remove layers with no data
    sparsity_data = {layer: values for layer, values in sparsity_data.items() if values}
    
    if not sparsity_data:
        print("No sparsity data to plot")
        return
    
    # Create line plot
    plt.figure(figsize=figsize)
    
    for layer, sparsity_values in sparsity_data.items():
        # Truncate layer name if too long
        display_name = layer
        if len(display_name) > 30:
            display_name = layer[:15] + "..." + layer[-12:]
        
        plt.plot(
            range(len(sparsity_values)),
            sparsity_values,
            marker='o',
            label=display_name
        )
    
    # Set title and labels
    plt.title(title or "Sparsity Evolution Over Time")
    plt.xlabel("Evolution Step")
    plt.ylabel("Sparsity (% of zeros)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if len(sparsity_data) > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Sparsity evolution plot saved to {output_file}")
    else:
        plt.show()


def plot_mask_stability_metrics(
    stability_metrics: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Create a visualization of mask stability metrics.
    
    Args:
        stability_metrics: Dictionary containing mask stability metrics
        output_file: Optional path to save the visualization
        figsize: Figure size as (width, height)
    """
    if not stability_metrics:
        print("No stability metrics data provided")
        return
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Mask changes over time
    if "tracking_steps" in stability_metrics and "mask_changes" in stability_metrics:
        steps = stability_metrics["tracking_steps"]
        changes = stability_metrics["mask_changes"]
        
        if len(steps) == len(changes) and len(steps) > 0:
            axes[0, 0].plot(steps, changes, marker='o', linestyle='-')
            axes[0, 0].set_title("Mask Change Rate Over Time")
            axes[0, 0].set_xlabel("Training Step")
            axes[0, 0].set_ylabel("Change Rate")
            axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Performance metrics
    if "tracking_steps" in stability_metrics and "performance_metrics" in stability_metrics:
        steps = stability_metrics["tracking_steps"]
        performance = stability_metrics["performance_metrics"]
        
        if len(steps) == len(performance) and len(steps) > 0:
            perf_changes = [p.get("performance_change", 0) for p in performance]
            
            axes[0, 1].plot(steps, perf_changes, marker='o', linestyle='-', color='green')
            axes[0, 1].set_title("Performance Change Over Evolution")
            axes[0, 1].set_xlabel("Training Step")
            axes[0, 1].set_ylabel("Performance Change")
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Convergence visualization
    if "mask_changes" in stability_metrics and len(stability_metrics["mask_changes"]) > 1:
        changes = stability_metrics["mask_changes"]
        
        # Calculate rate of change in mask changes
        diffs = np.diff(changes)
        step_indices = range(1, len(changes))
        
        axes[1, 0].bar(step_indices, diffs, color=['green' if d < 0 else 'red' for d in diffs])
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title("Change in Mask Evolution Rate")
        axes[1, 0].set_xlabel("Step Transition")
        axes[1, 0].set_ylabel("Δ Change Rate")
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Overall stability metrics
    metrics_to_plot = [
        ("stability_score", "Stability Score"),
        ("convergence", "Convergence"),
        ("perf_stability", "Performance Stability"),
        ("final_change_rate", "Final Change Rate")
    ]
    
    metric_values = []
    metric_labels = []
    
    for key, label in metrics_to_plot:
        if key in stability_metrics:
            metric_values.append(stability_metrics[key])
            metric_labels.append(label)
    
    if metric_values:
        axes[1, 1].bar(metric_labels, metric_values, color='purple')
        axes[1, 1].set_title("Stability Metrics")
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    # Add overall title
    fig.suptitle("Mask Evolution Stability Analysis", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Mask stability metrics plot saved to {output_file}")
    else:
        plt.show()


def visualize_mask_pattern_comparison(
    pattern_results: Dict[str, Dict[str, Any]],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Create a visualization comparing different mask patterns.
    
    Args:
        pattern_results: Dictionary mapping pattern types to their evaluation results
        output_file: Optional path to save the visualization
        figsize: Figure size as (width, height)
    """
    if not pattern_results:
        print("No pattern results data provided")
        return
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract pattern names and metrics
    pattern_names = list(pattern_results.keys())
    sparsities = [pattern_results[p].get("sparsity", 0) for p in pattern_names]
    inference_times = [pattern_results[p].get("inference_time", {}).get("mean", 0) for p in pattern_names]
    
    # Normalize inference time (lower is better)
    max_time = max(inference_times) if inference_times else 1.0
    norm_times = [max_time - t for t in inference_times] if max_time > 0 else inference_times
    
    # Plot 1: Sparsity by pattern
    axes[0, 0].bar(pattern_names, sparsities)
    axes[0, 0].set_title("Sparsity by Pattern Type")
    axes[0, 0].set_xlabel("Pattern Type")
    axes[0, 0].set_ylabel("Sparsity")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Inference time (lower is better)
    axes[0, 1].bar(pattern_names, inference_times, color='green')
    axes[0, 1].set_title("Inference Time by Pattern Type")
    axes[0, 1].set_xlabel("Pattern Type")
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Overall pattern effectiveness (higher is better)
    # Weighted average of sparsity and normalized inference time
    effectiveness = [0.7 * s + 0.3 * nt for s, nt in zip(sparsities, norm_times)]
    
    axes[1, 0].bar(pattern_names, effectiveness, color='purple')
    axes[1, 0].set_title("Overall Pattern Effectiveness")
    axes[1, 0].set_xlabel("Pattern Type")
    axes[1, 0].set_ylabel("Effectiveness Score")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Radar chart comparing multiple metrics
    # Extract additional metrics if available
    pattern_metrics = {
        "Sparsity": sparsities,
        "Speed": norm_times
    }
    
    # Check for additional metrics in first pattern and add if present for all
    first_pattern = pattern_results[pattern_names[0]] if pattern_names else {}
    for metric_name in first_pattern.keys():
        if metric_name not in ["sparsity", "inference_time"] and isinstance(first_pattern[metric_name], (int, float)):
            metric_values = [pattern_results[p].get(metric_name, 0) for p in pattern_names]
            pattern_metrics[metric_name] = metric_values
    
    # Use radar chart if we have more than 2 metrics
    if len(pattern_metrics) > 2:
        # Clear the last subplot and create a polar axis
        axes[1, 1].remove()
        ax_radar = fig.add_subplot(2, 2, 4, polar=True)
        
        # Set up the radar chart
        metrics = list(pattern_metrics.keys())
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for i, pattern in enumerate(pattern_names):
            values = [pattern_metrics[metric][i] for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax_radar.plot(angles, values, linewidth=2, label=pattern)
            ax_radar.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax_radar.set_title("Pattern Comparison")
        ax_radar.grid(True)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    else:
        # Create a bar chart comparison if we don't have enough metrics for radar
        axes[1, 1].bar(pattern_names, effectiveness, color='purple')
        axes[1, 1].set_title("Pattern Effectiveness (Alternative View)")
        axes[1, 1].set_xlabel("Pattern Type")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle("Mask Pattern Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Pattern comparison plot saved to {output_file}")
    else:
        plt.show()
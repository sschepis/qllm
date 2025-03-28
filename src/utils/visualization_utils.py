"""
Visualization utilities for QLLM.

This module provides utilities for creating visualizations for various aspects
of the QLLM system, including training progress, model outputs, and evaluation results.
"""

import os
import math
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger("qllm.utils.visualization")


def setup_plotting(
    style: str = "seaborn-v0_8-darkgrid",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    font_scale: float = 1.0
) -> None:
    """
    Set up matplotlib for consistent plotting.
    
    Args:
        style: Matplotlib style to use
        figsize: Default figure size
        dpi: Default DPI
        font_scale: Scale factor for fonts
    """
    # Set style
    plt.style.use(style)
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    
    # Set fonts
    plt.rcParams["font.size"] = 12 * font_scale
    plt.rcParams["axes.labelsize"] = 12 * font_scale
    plt.rcParams["axes.titlesize"] = 14 * font_scale
    plt.rcParams["xtick.labelsize"] = 10 * font_scale
    plt.rcParams["ytick.labelsize"] = 10 * font_scale
    plt.rcParams["legend.fontsize"] = 10 * font_scale


def create_figure(
    figsize: Optional[Tuple[int, int]] = None,
    dpi: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    tight_layout: bool = True
) -> Tuple[Figure, Axes]:
    """
    Create a matplotlib figure and axes with common settings.
    
    Args:
        figsize: Figure size in inches
        dpi: Figure resolution
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        tight_layout: Whether to use tight layout
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Set title and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Apply tight layout
    if tight_layout:
        fig.tight_layout()
    
    return fig, ax


def save_figure(
    fig: Figure,
    filename: str,
    directory: str = "figures",
    formats: List[str] = ["png", "pdf"],
    dpi: int = 300,
    close_figure: bool = True
) -> List[str]:
    """
    Save a figure to disk in multiple formats.
    
    Args:
        fig: Figure to save
        filename: Base filename (without extension)
        directory: Directory to save figure in
        formats: List of formats to save as
        dpi: Resolution for raster formats
        close_figure: Whether to close figure after saving
        
    Returns:
        List of saved file paths
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save in each format
    saved_paths = []
    for fmt in formats:
        # Create full path
        path = os.path.join(directory, f"{filename}.{fmt}")
        
        # Save figure
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
        
        logger.info(f"Saved figure to {path}")
    
    # Close figure if requested
    if close_figure:
        plt.close(fig)
    
    return saved_paths


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    include_legend: bool = True,
    grid: bool = True,
    smooth_factor: int = 0
) -> Optional[Tuple[Figure, List[Axes]]]:
    """
    Plot training curves from metrics history.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure (if None, figure is not saved)
        include_legend: Whether to include legend
        grid: Whether to show grid
        smooth_factor: Window size for smoothing (0 for no smoothing)
        
    Returns:
        Tuple of (figure, list of axes) if output_path is None, otherwise None
    """
    # Determine number of metrics to plot
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        logger.warning("No metrics to plot")
        return None
    
    # Create figure
    n_cols = min(2, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes if needed
    if n_metrics == 1:
        axes = [axes]
    elif n_rows * n_cols > 1:
        axes = axes.flatten()
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Apply smoothing if requested
        if smooth_factor > 0 and len(values) > smooth_factor:
            smoothed_values = []
            for j in range(len(values)):
                window_start = max(0, j - smooth_factor)
                window_end = min(len(values), j + smooth_factor + 1)
                window = values[window_start:window_end]
                smoothed_values.append(sum(window) / len(window))
            
            # Plot both original and smoothed
            steps = list(range(1, len(values) + 1))
            ax.plot(steps, values, alpha=0.3, label=f"{metric_name} (raw)")
            ax.plot(steps, smoothed_values, label=f"{metric_name} (smoothed)")
        else:
            # Plot original only
            steps = list(range(1, len(values) + 1))
            ax.plot(steps, values, label=metric_name)
        
        # Set title and labels
        if metric_name.lower() == "loss":
            ax.set_title(f"Training Loss")
            ax.set_ylabel("Loss")
        else:
            ax.set_title(f"{metric_name}")
            ax.set_ylabel("Value")
        
        ax.set_xlabel("Steps")
        
        # Set grid and legend
        if grid:
            ax.grid(True, alpha=0.3)
        if include_legend:
            ax.legend()
    
    # Remove empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        fig.delaxes(axes[i])
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save figure if requested
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return None
    
    return fig, axes


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> Optional[Tuple[Figure, Axes]]:
    """
    Plot a confusion matrix.
    
    Args:
        matrix: Confusion matrix (shape [n_classes, n_classes])
        class_names: List of class names
        title: Plot title
        cmap: Colormap name
        normalize: Whether to normalize by row
        figsize: Figure size
        output_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Tuple of (figure, axes) if output_path is None, otherwise None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        matrix = matrix.astype(float)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
    # Plot confusion matrix
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Counts" if not normalize else "Proportion", rotation=-90, va="bottom")
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure if requested
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return None
    
    return fig, ax


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None
) -> Optional[Tuple[Figure, Axes]]:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weight matrix (shape [n_tokens, n_tokens])
        tokens: List of token strings
        title: Plot title
        cmap: Colormap name
        figsize: Figure size (computed based on number of tokens if None)
        output_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Tuple of (figure, axes) if output_path is None, otherwise None
    """
    # Compute figure size if not provided
    if figsize is None:
        size_factor = max(len(tokens) / 10, 1)
        figsize = (size_factor * 8, size_factor * 6)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap=plt.cm.get_cmap(cmap))
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    
    # Set token labels, handling potentially long token lists
    if len(tokens) > 30:
        # Show a subset of ticks
        tick_interval = max(1, len(tokens) // 30)
        ax.set_xticks(np.arange(0, len(tokens), tick_interval))
        ax.set_yticks(np.arange(0, len(tokens), tick_interval))
        
        # Set labels for the shown ticks
        x_labels = [tokens[i] for i in range(0, len(tokens), tick_interval)]
        y_labels = [tokens[i] for i in range(0, len(tokens), tick_interval)]
    else:
        x_labels = tokens
        y_labels = tokens
    
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel('Target Token')
    ax.set_xlabel('Source Token')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure if requested
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return None
    
    return fig, ax


def plot_embedding_clusters(
    embeddings: np.ndarray,
    labels: List[Any],
    title: str = "Embedding Clusters",
    method: str = "tsne",
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None,
    **kwargs
) -> Optional[Tuple[Figure, Axes]]:
    """
    Visualize embedding clusters using dimensionality reduction.
    
    Args:
        embeddings: Embedding matrix (shape [n_samples, n_dimensions])
        labels: List of labels for each sample
        title: Plot title
        method: Dimensionality reduction method ("tsne", "pca", or "umap")
        figsize: Figure size
        output_path: Path to save the figure (if None, figure is not saved)
        **kwargs: Additional arguments for the dimensionality reduction method
        
    Returns:
        Tuple of (figure, axes) if output_path is None, otherwise None
    """
    # Check if required packages are installed
    try:
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, **kwargs)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, **kwargs)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        return None
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert labels to numeric if they're not already
    if not isinstance(labels[0], (int, float, np.integer, np.floating)):
        unique_labels = sorted(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_map[label] for label in labels]
    else:
        numeric_labels = labels
    
    # Create scatter plot
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=numeric_labels,
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add legend if there aren't too many classes
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 20:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=scatter.cmap(scatter.norm(i)), 
                          markersize=10, label=str(label))
                  for i, label in enumerate(unique_labels)]
        ax.legend(handles=legend_elements, loc='best')
    
    # Set title and labels
    method_names = {"tsne": "t-SNE", "pca": "PCA", "umap": "UMAP"}
    ax.set_title(f"{title}\n({method_names.get(method, method)} Projection)")
    ax.set_xlabel(f"Dimension 1")
    ax.set_ylabel(f"Dimension 2")
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure if requested
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return None
    
    return fig, ax


def create_interactive_dashboard(
    metrics: Dict[str, List[float]],
    output_path: str = "dashboard.html",
    title: str = "Model Training Dashboard"
) -> str:
    """
    Create an interactive HTML dashboard for visualizing training metrics.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        output_path: Path to save the HTML file
        title: Dashboard title
        
    Returns:
        Path to the generated HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly is required for interactive dashboards. Install with: pip install plotly")
        return ""
    
    # Determine layout
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[metric_name for metric_name in metrics.keys()],
        vertical_spacing=0.1
    )
    
    # Add traces for each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        steps = list(range(1, len(values) + 1))
        
        # Add raw values
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode='lines',
                name=f"{metric_name}",
                line=dict(width=2)
            ),
            row=row,
            col=col
        )
        
        # Add smoothed values
        if len(values) > 10:
            window_size = max(5, len(values) // 20)
            smoothed_values = []
            for j in range(len(values)):
                window_start = max(0, j - window_size)
                window_end = min(len(values), j + window_size + 1)
                window = values[window_start:window_end]
                smoothed_values.append(sum(window) / len(window))
            
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=smoothed_values,
                    mode='lines',
                    name=f"{metric_name} (smoothed)",
                    line=dict(width=2, dash='dash')
                ),
                row=row,
                col=col
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300 * n_rows,
        width=600 * n_cols,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Steps")
    
    for i, metric_name in enumerate(metrics.keys()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if metric_name.lower() == "loss":
            fig.update_yaxes(title_text="Loss", row=row, col=col)
        else:
            fig.update_yaxes(title_text="Value", row=row, col=col)
    
    # Create directory if needed
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save HTML file
    fig.write_html(output_path)
    logger.info(f"Interactive dashboard saved to {output_path}")
    
    return output_path
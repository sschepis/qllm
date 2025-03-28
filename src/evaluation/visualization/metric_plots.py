"""
Metric Plotter for QLLM evaluation.

This module provides specialized visualization components for metric results,
extending the base plotter with functionality specific to plotting evaluation
metrics and category scores.
"""

import os
import math
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.evaluation.visualization.base_plotter import BasePlotter


logger = logging.getLogger("qllm.evaluation")


class MetricPlotter(BasePlotter):
    """
    Plotter for evaluation metrics.
    
    This class extends the base plotter with functionality specific to
    plotting evaluation metrics, including bar charts, radar plots,
    and category comparisons.
    """
    
    def __init__(
        self,
        metric_order: Optional[List[str]] = None,
        category_order: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the metric plotter.
        
        Args:
            metric_order: Optional ordered list of metrics to plot
            category_order: Optional ordered list of categories to plot
            **kwargs: Additional parameters for the base plotter
        """
        # Initialize base plotter
        super().__init__(**kwargs)
        
        # Store metric and category orders
        self.metric_order = metric_order
        self.category_order = category_order
    
    def plot_metrics(
        self,
        metric_results: Dict[str, Any],
        output_path: Optional[str] = None,
        title: str = "Evaluation Metrics",
        plot_type: str = "bar",
        sort_metrics: bool = True,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Plot evaluation metrics.
        
        Args:
            metric_results: Dictionary of metric results
            output_path: Optional path to save the plot to
            title: Title for the plot
            plot_type: Type of plot ('bar', 'radar', 'heatmap')
            sort_metrics: Whether to sort metrics by value
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Extract metric values
        metrics = {}
        for metric_name, result in metric_results.items():
            # Skip metrics with errors
            if "error" in result:
                continue
            
            # Extract value
            if "value" in result and isinstance(result["value"], (int, float)):
                metrics[metric_name] = result["value"]
        
        # Use metric order if provided
        if self.metric_order is not None:
            # Filter to metrics in both results and order
            ordered_metrics = [
                (metric, metrics[metric])
                for metric in self.metric_order
                if metric in metrics
            ]
            # Add any remaining metrics not in order
            for metric, value in metrics.items():
                if metric not in self.metric_order:
                    ordered_metrics.append((metric, value))
        else:
            # Sort metrics by value if requested
            if sort_metrics:
                ordered_metrics = sorted(
                    metrics.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            else:
                ordered_metrics = list(metrics.items())
        
        # Create appropriate plot based on type
        if plot_type == "radar":
            return self._plot_radar_metrics(
                ordered_metrics,
                output_path=output_path,
                title=title,
                **kwargs
            )
        elif plot_type == "heatmap":
            return self._plot_heatmap_metrics(
                ordered_metrics,
                output_path=output_path,
                title=title,
                **kwargs
            )
        else:
            # Default to bar plot
            return self._plot_bar_metrics(
                ordered_metrics,
                output_path=output_path,
                title=title,
                **kwargs
            )
    
    def _plot_bar_metrics(
        self,
        ordered_metrics: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Evaluation Metrics",
        color: str = "dodgerblue",
        threshold_line: Optional[float] = None,
        show_values: bool = True,
        horizontal: bool = False,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a bar chart of metrics.
        
        Args:
            ordered_metrics: List of (metric_name, value) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            color: Bar color
            threshold_line: Optional threshold line value
            show_values: Whether to show values on bars
            horizontal: Whether to create a horizontal bar chart
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Create figure
        fig, ax = self.create_figure(**kwargs)
        
        # Extract metric names and values
        metric_names = [m[0] for m in ordered_metrics]
        values = [m[1] for m in ordered_metrics]
        
        # Create bar plot
        if horizontal:
            # Horizontal bar chart (metrics on y-axis)
            bars = ax.barh(metric_names, values, color=color)
            
            # Set axis labels
            xlabel = "Score"
            ylabel = ""
            
            # Set label positions
            if show_values:
                for bar in bars:
                    width = bar.get_width()
                    label_position = width + 0.01
                    ax.text(
                        label_position,
                        bar.get_y() + bar.get_height() / 2,
                        self.format_value(width),
                        va='center',
                        fontsize=self.tick_fontsize
                    )
            
            # Add threshold line if requested
            if threshold_line is not None:
                ax.axvline(
                    x=threshold_line,
                    color='red',
                    linestyle='--',
                    alpha=0.7,
                    label=f"Threshold ({threshold_line})"
                )
                ax.legend()
        else:
            # Vertical bar chart (metrics on x-axis)
            bars = ax.bar(metric_names, values, color=color)
            
            # Set axis labels
            xlabel = ""
            ylabel = "Score"
            
            # Set label positions
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    label_position = height + 0.01
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_position,
                        self.format_value(height),
                        ha='center',
                        va='bottom',
                        fontsize=self.tick_fontsize
                    )
            
            # Add threshold line if requested
            if threshold_line is not None:
                ax.axhline(
                    y=threshold_line,
                    color='red',
                    linestyle='--',
                    alpha=0.7,
                    label=f"Threshold ({threshold_line})"
                )
                ax.legend()
        
        # Apply styling
        self.apply_styling(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs
        )
        
        # Adjust layout for tick labels
        plt.tight_layout()
        
        # Rotate x-tick labels for vertical bar chart
        if not horizontal and len(metric_names) > 0:
            plt.xticks(rotation=45, ha='right')
        
        # Save and return
        if output_path:
            self.save_figure(fig, output_path)
            self.close_figure(fig)
            return output_path
        else:
            return fig, ax
    
    def _plot_radar_metrics(
        self,
        ordered_metrics: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Evaluation Metrics",
        color: str = "dodgerblue",
        fill: bool = True,
        fill_alpha: float = 0.2,
        show_values: bool = True,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a radar chart of metrics.
        
        Args:
            ordered_metrics: List of (metric_name, value) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            color: Line color
            fill: Whether to fill the radar chart
            fill_alpha: Alpha value for fill
            show_values: Whether to show values on the chart
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Create figure
        fig, ax = self.create_figure(**kwargs)
        
        # Extract metric names and values
        metric_names = [m[0] for m in ordered_metrics]
        values = [m[1] for m in ordered_metrics]
        
        # Number of metrics
        n = len(metric_names)
        
        # Empty radar chart
        if n == 0:
            ax.text(
                0.5, 0.5,
                "No metrics to display",
                ha='center',
                va='center',
                fontsize=self.label_fontsize
            )
            
            # Apply styling
            self.apply_styling(
                ax,
                title=title,
                **kwargs
            )
            
            # Save and return
            if output_path:
                self.save_figure(fig, output_path)
                self.close_figure(fig)
                return output_path
            else:
                return fig, ax
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        
        # Close the plot
        values.append(values[0])
        angles.append(angles[0])
        metric_names.append(metric_names[0])
        
        # Set the angle axis
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw lines and points
        ax.plot(angles, values, color=color, marker='o', linewidth=2)
        
        # Fill the area if requested
        if fill:
            ax.fill(angles, values, color=color, alpha=fill_alpha)
        
        # Set labels on the radial axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names[:-1])
        
        # Show values if requested
        if show_values:
            for i, (angle, value, name) in enumerate(zip(angles[:-1], values[:-1], metric_names[:-1])):
                ha = 'center'
                if angle < np.pi / 2 or angle > 3 * np.pi / 2:
                    ha = 'left'
                elif angle > np.pi / 2 and angle < 3 * np.pi / 2:
                    ha = 'right'
                
                value_x = (value + 0.1) * np.cos(angle)
                value_y = (value + 0.1) * np.sin(angle)
                
                ax.text(
                    value_x, value_y,
                    self.format_value(value),
                    ha=ha,
                    va='center',
                    fontsize=self.tick_fontsize
                )
        
        # Apply styling
        self.apply_styling(
            ax,
            title=title,
            grid=True,
            **kwargs
        )
        
        # Save and return
        if output_path:
            self.save_figure(fig, output_path)
            self.close_figure(fig)
            return output_path
        else:
            return fig, ax
    
    def _plot_heatmap_metrics(
        self,
        ordered_metrics: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Evaluation Metrics",
        cmap: str = "viridis",
        show_values: bool = True,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a heatmap of metrics.
        
        Args:
            ordered_metrics: List of (metric_name, value) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            cmap: Colormap to use
            show_values: Whether to show values on the heatmap
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Create figure
        fig, ax = self.create_figure(**kwargs)
        
        # Extract metric names and values
        metric_names = [m[0] for m in ordered_metrics]
        values = [m[1] for m in ordered_metrics]
        
        # Number of metrics
        n = len(metric_names)
        
        # Calculate grid dimensions
        grid_size = math.ceil(math.sqrt(n))
        
        # Create grid of values
        grid = np.zeros((grid_size, grid_size))
        for i, value in enumerate(values):
            row = i // grid_size
            col = i % grid_size
            grid[row, col] = value
        
        # Create heatmap
        im = ax.imshow(grid, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Score", fontsize=self.label_fontsize)
        
        # Set axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Show values and labels if requested
        if show_values:
            for i, (name, value) in enumerate(zip(metric_names, values)):
                row = i // grid_size
                col = i % grid_size
                
                # Determine text color based on value
                if value > 0.5:
                    text_color = "white"
                else:
                    text_color = "black"
                
                # Show value
                ax.text(
                    col, row,
                    f"{name}\n{self.format_value(value)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=self.tick_fontsize
                )
        
        # Apply styling
        self.apply_styling(
            ax,
            title=title,
            grid=False,
            **kwargs
        )
        
        # Save and return
        if output_path:
            self.save_figure(fig, output_path)
            self.close_figure(fig)
            return output_path
        else:
            return fig, ax
    
    def plot_categories(
        self,
        category_scores: Dict[str, float],
        output_path: Optional[str] = None,
        title: str = "Category Scores",
        plot_type: str = "bar",
        sort_categories: bool = True,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Plot category scores.
        
        Args:
            category_scores: Dictionary mapping categories to scores
            output_path: Optional path to save the plot to
            title: Title for the plot
            plot_type: Type of plot ('bar', 'radar', 'pie')
            sort_categories: Whether to sort categories by value
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Use category order if provided
        if self.category_order is not None:
            # Filter to categories in both scores and order
            ordered_categories = [
                (category, category_scores[category])
                for category in self.category_order
                if category in category_scores
            ]
            # Add any remaining categories not in order
            for category, score in category_scores.items():
                if category not in self.category_order:
                    ordered_categories.append((category, score))
        else:
            # Sort categories by value if requested
            if sort_categories:
                ordered_categories = sorted(
                    category_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            else:
                ordered_categories = list(category_scores.items())
        
        # Create appropriate plot based on type
        if plot_type == "radar":
            return self._plot_radar_categories(
                ordered_categories,
                output_path=output_path,
                title=title,
                **kwargs
            )
        elif plot_type == "pie":
            return self._plot_pie_categories(
                ordered_categories,
                output_path=output_path,
                title=title,
                **kwargs
            )
        else:
            # Default to bar plot
            return self._plot_bar_categories(
                ordered_categories,
                output_path=output_path,
                title=title,
                **kwargs
            )
    
    def _plot_bar_categories(
        self,
        ordered_categories: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Category Scores",
        use_colors: bool = True,
        show_values: bool = True,
        horizontal: bool = False,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a bar chart of category scores.
        
        Args:
            ordered_categories: List of (category, score) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            use_colors: Whether to use different colors for categories
            show_values: Whether to show values on bars
            horizontal: Whether to create a horizontal bar chart
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Create figure
        fig, ax = self.create_figure(**kwargs)
        
        # Extract category names and scores
        category_names = [c[0] for c in ordered_categories]
        scores = [c[1] for c in ordered_categories]
        
        # Get colors if requested
        if use_colors:
            # Get categorical colors
            colors = list(self.get_categorical_colors(
                category_names,
                palette=self.palette
            ).values())
        else:
            # Use single color
            colors = "dodgerblue"
        
        # Create bar plot
        if horizontal:
            # Horizontal bar chart (categories on y-axis)
            bars = ax.barh(category_names, scores, color=colors)
            
            # Set axis labels
            xlabel = "Score"
            ylabel = ""
            
            # Set label positions
            if show_values:
                for bar in bars:
                    width = bar.get_width()
                    label_position = width + 0.01
                    ax.text(
                        label_position,
                        bar.get_y() + bar.get_height() / 2,
                        self.format_value(width),
                        va='center',
                        fontsize=self.tick_fontsize
                    )
        else:
            # Vertical bar chart (categories on x-axis)
            bars = ax.bar(category_names, scores, color=colors)
            
            # Set axis labels
            xlabel = ""
            ylabel = "Score"
            
            # Set label positions
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    label_position = height + 0.01
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_position,
                        self.format_value(height),
                        ha='center',
                        va='bottom',
                        fontsize=self.tick_fontsize
                    )
        
        # Apply styling
        self.apply_styling(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs
        )
        
        # Rotate x-tick labels for vertical bar chart
        if not horizontal and len(category_names) > 0:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout for tick labels
        plt.tight_layout()
        
        # Save and return
        if output_path:
            self.save_figure(fig, output_path)
            self.close_figure(fig)
            return output_path
        else:
            return fig, ax
    
    def _plot_radar_categories(
        self,
        ordered_categories: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Category Scores",
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a radar chart of category scores.
        
        Args:
            ordered_categories: List of (category, score) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Reuse the radar metric plotting function
        return self._plot_radar_metrics(
            ordered_categories,
            output_path=output_path,
            title=title,
            **kwargs
        )
    
    def _plot_pie_categories(
        self,
        ordered_categories: List[Tuple[str, float]],
        output_path: Optional[str] = None,
        title: str = "Category Scores",
        show_values: bool = True,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create a pie chart of category scores.
        
        Args:
            ordered_categories: List of (category, score) tuples
            output_path: Optional path to save the plot to
            title: Title for the plot
            show_values: Whether to show values in labels
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
        """
        # Create figure
        fig, ax = self.create_figure(**kwargs)
        
        # Extract category names and scores
        category_names = [c[0] for c in ordered_categories]
        scores = [c[1] for c in ordered_categories]
        
        # Normalize scores for pie chart
        total_score = sum(scores)
        if total_score > 0:
            normalized_scores = [score / total_score for score in scores]
        else:
            normalized_scores = [1.0 / len(scores) for _ in scores]
        
        # Create labels
        if show_values:
            labels = [
                f"{category} ({self.format_value(score)})"
                for category, score in zip(category_names, scores)
            ]
        else:
            labels = category_names
        
        # Create pie chart
        ax.pie(
            normalized_scores,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )
        
        # Equal aspect ratio ensures circular pie
        ax.axis('equal')
        
        # Apply styling
        self.apply_styling(
            ax,
            title=title,
            grid=False,
            **kwargs
        )
        
        # Save and return
        if output_path:
            self.save_figure(fig, output_path)
            self.close_figure(fig)
            return output_path
        else:
            return fig, ax
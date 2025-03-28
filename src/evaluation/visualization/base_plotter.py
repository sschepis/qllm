"""
Base Plotter for QLLM evaluation visualization.

This module provides a base class for visualization components,
defining common functionality and interfaces that specialized
visualization components can extend.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


logger = logging.getLogger("qllm.evaluation")


class BasePlotter:
    """
    Base class for visualization components.
    
    This class provides common functionality for visualization components,
    including figure creation, styling, and saving, as well as defining
    a consistent interface that specialized plotters can extend.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        style: str = "seaborn-v0_8-darkgrid",
        palette: Optional[str] = "viridis",
        title_fontsize: int = 16,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        legend_fontsize: int = 10,
        **kwargs
    ):
        """
        Initialize the base plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for the figure
            style: Matplotlib style to use
            palette: Color palette to use
            title_fontsize: Font size for plot titles
            label_fontsize: Font size for axis labels
            tick_fontsize: Font size for tick labels
            legend_fontsize: Font size for legends
            **kwargs: Additional plotter parameters
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.palette = palette
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Check if matplotlib backend supports interactive mode
        self._supports_interactive = self._check_interactive_support()
    
    def _check_interactive_support(self) -> bool:
        """
        Check if matplotlib backend supports interactive mode.
        
        Returns:
            True if interactive mode is supported, False otherwise
        """
        try:
            # Check if current backend is interactive
            return matplotlib.is_interactive()
        except:
            return False
    
    def create_figure(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = None,
        style: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create a matplotlib figure and axes.
        
        Args:
            figsize: Optional figure size (overrides instance default)
            dpi: Optional DPI (overrides instance default)
            style: Optional style (overrides instance default)
            
        Returns:
            Tuple of (figure, axes)
        """
        # Set style if provided
        if style is not None or self.style is not None:
            plt.style.use(style or self.style)
        
        # Create figure and axes
        fig, ax = plt.subplots(
            figsize=figsize or self.figsize,
            dpi=dpi or self.dpi
        )
        
        return fig, ax
    
    def apply_styling(
        self,
        ax: Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        legend: bool = False,
        legend_loc: str = "best",
        grid: bool = True,
        **kwargs
    ) -> Axes:
        """
        Apply styling to axes.
        
        Args:
            ax: Axes to style
            title: Title for the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            xlim: Limits for x-axis
            ylim: Limits for y-axis
            legend: Whether to show legend
            legend_loc: Legend location
            grid: Whether to show grid
            **kwargs: Additional styling parameters
            
        Returns:
            Styled axes
        """
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=self.title_fontsize)
        
        # Set axis labels if provided
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.label_fontsize)
        
        # Set axis limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        # Set tick label font size
        ax.tick_params(axis='both', labelsize=self.tick_fontsize)
        
        # Show legend if requested
        if legend:
            ax.legend(loc=legend_loc, fontsize=self.legend_fontsize)
        
        # Show grid if requested
        if grid:
            ax.grid(True)
        
        return ax
    
    def save_figure(
        self,
        fig: Figure,
        output_path: str,
        tight_layout: bool = True,
        dpi: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Save a figure to file.
        
        Args:
            fig: Figure to save
            output_path: Path to save the figure to
            tight_layout: Whether to use tight layout
            dpi: Optional DPI for the saved figure
            **kwargs: Additional saving parameters
            
        Returns:
            Path to the saved figure
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Apply tight layout if requested
        if tight_layout:
            fig.tight_layout()
        
        # Save figure
        fig.savefig(
            output_path,
            dpi=dpi or self.dpi,
            bbox_inches='tight',
            **kwargs
        )
        
        logger.info(f"Saved figure to {output_path}")
        return output_path
    
    def close_figure(self, fig: Figure) -> None:
        """
        Close a figure.
        
        Args:
            fig: Figure to close
        """
        plt.close(fig)
    
    def plot(
        self,
        data: Any,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Union[str, Tuple[Figure, Axes]]:
        """
        Create and save a plot.
        
        This method should be implemented by subclasses to create specific
        types of plots. The base implementation raises NotImplementedError.
        
        Args:
            data: Data to plot
            output_path: Optional path to save the plot to
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the saved plot if output_path is provided,
            otherwise tuple of (figure, axes)
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement plot()")
    
    @staticmethod
    def get_color_map(
        num_colors: int,
        cmap_name: str = "viridis"
    ) -> List[Tuple[float, float, float, float]]:
        """
        Get a list of colors from a colormap.
        
        Args:
            num_colors: Number of colors to get
            cmap_name: Name of the colormap to use
            
        Returns:
            List of RGBA color tuples
        """
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    
    @staticmethod
    def get_categorical_colors(
        categories: List[str],
        palette: Optional[str] = None
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get colors for categories.
        
        Args:
            categories: List of category names
            palette: Optional palette name
            
        Returns:
            Dictionary mapping categories to RGBA color tuples
        """
        # Get colormap
        cmap = plt.get_cmap(palette or "tab10")
        
        # Assign colors to categories
        return {
            category: cmap(i % 10) 
            for i, category in enumerate(categories)
        }
    
    @staticmethod
    def format_value(
        value: float,
        precision: int = 2,
        percentage: bool = False
    ) -> str:
        """
        Format a numeric value for display.
        
        Args:
            value: Value to format
            precision: Number of decimal places
            percentage: Whether to format as percentage
            
        Returns:
            Formatted value string
        """
        if percentage:
            return f"{value * 100:.{precision}f}%"
        else:
            return f"{value:.{precision}f}"
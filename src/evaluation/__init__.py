"""
Evaluation module for QLLM.

This module provides a framework for evaluating language models on various metrics,
including comprehensive evaluation suites, individual metrics, and visualization tools.
"""

# Import comprehensive suite
from src.evaluation.comprehensive_suite import ComprehensiveSuite

# Import core components that are definitely available
from src.evaluation.core.config import EvaluationConfig
from src.evaluation.core.suite import EvaluationSuite
from src.evaluation.core.model_utils import ModelUtils

# Import metric utilities
from src.evaluation.metrics import (
    calculate_metrics,
    get_metric_registry,
    compute_overall_score,
    calculate_perplexity,
    calculate_token_accuracy,
    calculate_sequence_accuracy,
    text_generation_metrics
)

# Import available visualization components
try:
    from src.evaluation.visualization.metric_plots import MetricPlotter
    from src.evaluation.visualization.base_plotter import BasePlotter
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

__all__ = [
    # Main components
    'ComprehensiveSuite',
    
    # Core components
    'EvaluationSuite',
    'EvaluationConfig',
    'ModelUtils',
    
    # Metrics utilities
    'calculate_metrics',
    'get_metric_registry',
    'compute_overall_score',
    'calculate_perplexity',
    'calculate_token_accuracy',
    'calculate_sequence_accuracy',
    'text_generation_metrics'
]

# Add visualization components if available
if HAS_VISUALIZATION:
    __all__.extend(['MetricPlotter', 'BasePlotter'])
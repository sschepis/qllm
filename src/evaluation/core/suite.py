"""
Evaluation suite for QLLM.

This module provides the base evaluation suite class that defines the
core functionality for model evaluation, with a consistent interface
that specialized evaluation suites can extend.
"""

import os
import json
import time
import datetime
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Set

import torch
import numpy as np

from src.evaluation.core.config import EvaluationConfig
from src.evaluation.core.model_utils import ModelUtils, load_model, prepare_model_for_evaluation

logger = logging.getLogger("qllm.evaluation")


class EvaluationSuite:
    """
    Base class for evaluation suites.
    
    This class provides the core functionality for model evaluation,
    including loading models, preparing data, running evaluations,
    and saving results. It defines a consistent interface that
    specialized evaluation suites can extend.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        metrics_registry: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the evaluation suite.
        
        Args:
            config: Evaluation configuration
            model: Pre-loaded model (if None, will load from config)
            tokenizer: Pre-loaded tokenizer (if None, will load with model)
            metrics_registry: Registry of metric functions (name -> function)
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Load metrics registry if not provided
        self.metrics_registry = metrics_registry or self._load_metrics_registry()
        
        # Initialize data storage
        self.evaluation_data = None
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Create model utilities
        self.model_utils = ModelUtils()
    
    def _load_metrics_registry(self) -> Dict[str, Callable]:
        """
        Load the metrics registry.
        
        Returns:
            Dictionary mapping metric names to metric functions
        """
        # Import the metrics registry
        try:
            from src.evaluation.metrics import get_metric_registry
            return get_metric_registry()
        except ImportError:
            logger.warning("Could not import metrics registry")
            return {}
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the model for evaluation.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is not None and self.tokenizer is not None:
            logger.info("Using pre-loaded model and tokenizer")
            return self.model, self.tokenizer
        
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_model(
            model_path=self.config.model_id,
            device=self.config.device,
            precision=self.config.precision
        )
        
        # Prepare model and tokenizer for evaluation
        self.model, self.tokenizer = prepare_model_for_evaluation(
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        return self.model, self.tokenizer
    
    def prepare_data(self) -> Any:
        """
        Prepare evaluation data.
        
        This method should be implemented by subclasses to load and prepare
        the specific data needed for their evaluation.
        
        Returns:
            Prepared evaluation data
        """
        raise NotImplementedError("Subclasses must implement prepare_data()")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation.
        
        This method orchestrates the evaluation process, including loading
        the model, preparing the data, running the metrics, and aggregating
        the results.
        
        Returns:
            Dictionary of evaluation results
        """
        # Record start time
        self.start_time = time.time()
        
        # Load model and tokenizer if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Prepare evaluation data if not already prepared
        if self.evaluation_data is None:
            logger.info("Preparing evaluation data")
            self.evaluation_data = self.prepare_data()
        
        # Run each metric
        metric_results = {}
        for metric_name in self.config.metrics:
            try:
                # Get metric function
                metric_fn = self.metrics_registry.get(metric_name)
                if metric_fn is None:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue
                
                # Run metric
                logger.info(f"Running metric: {metric_name}")
                result = self._run_metric(metric_fn, metric_name)
                
                # Store result
                metric_results[metric_name] = result
                
            except Exception as e:
                logger.error(f"Error running metric {metric_name}: {e}")
                metric_results[metric_name] = {"error": str(e)}
        
        # Record end time
        self.end_time = time.time()
        
        # Aggregate results
        self.results = self._aggregate_results(metric_results)
        
        # Return results
        return self.results
    
    def _run_metric(self, metric_fn: Callable, metric_name: str) -> Dict[str, Any]:
        """
        Run a single metric.
        
        Args:
            metric_fn: Function implementing the metric
            metric_name: Name of the metric
            
        Returns:
            Dictionary with metric results
        """
        # Run the metric
        try:
            metric_start_time = time.time()
            
            # Call the metric function
            result = metric_fn(
                model=self.model,
                tokenizer=self.tokenizer,
                data=self.evaluation_data,
                config=self.config
            )
            
            metric_end_time = time.time()
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"value": result}
            
            # Add execution time
            result["execution_time"] = metric_end_time - metric_start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in metric {metric_name}: {e}")
            return {"error": str(e)}
    
    def _aggregate_results(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate evaluation results.
        
        Args:
            metric_results: Dictionary mapping metric names to results
            
        Returns:
            Dictionary with aggregated results
        """
        # Calculate overall score if possible
        overall_score = None
        weighted_sum = 0
        total_weight = 0
        
        # Try to compute weighted average of scores
        for metric_name, result in metric_results.items():
            if "value" in result and isinstance(result["value"], (int, float)):
                # Get weight for this metric (default to 1.0)
                weight = getattr(self.config, f"{metric_name}_weight", 1.0)
                
                # Add to weighted sum
                weighted_sum += result["value"] * weight
                total_weight += weight
        
        # Calculate overall score if we have valid metrics
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        
        # Create results dictionary
        results = {
            "model_id": self.config.model_id,
            "metrics": metric_results,
            "overall_score": overall_score,
            "execution_time": (self.end_time - self.start_time) if self.end_time else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config.to_dict()
        }
        
        return results
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results to (if None, uses config.output_dir)
            
        Returns:
            Path to the saved results file
        """
        # Generate filepath if not provided
        if filepath is None:
            if not self.config.output_dir:
                raise ValueError("No output directory specified in config")
            
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Generate filename
            model_name = os.path.basename(self.config.model_id.rstrip("/"))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.config.output_dir,
                f"{model_name}_results_{timestamp}.json"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Convert results to JSON
        results_json = json.dumps(self.results, indent=2)
        
        # Save to file
        with open(filepath, "w") as f:
            f.write(results_json)
        
        logger.info(f"Saved evaluation results to {filepath}")
        return filepath
    
    @classmethod
    def load_results(cls, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary with evaluation results
        """
        with open(filepath, "r") as f:
            results = json.load(f)
        
        return results
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'EvaluationSuite':
        """
        Create an evaluation suite from a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured evaluation suite
        """
        # Load configuration
        config = EvaluationConfig.load(config_path)
        
        # Create suite
        return cls(config=config)


def run_evaluation_from_config(config_path: str) -> Dict[str, Any]:
    """
    Run an evaluation from a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with evaluation results
    """
    # Create evaluation suite
    suite = EvaluationSuite.from_config_file(config_path)
    
    # Run evaluation
    results = suite.run_evaluation()
    
    # Save results
    suite.save_results()
    
    return results
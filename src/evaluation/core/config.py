"""
Evaluation configuration for QLLM.

This module provides configuration classes for the evaluation framework,
defining the structure and validation for evaluation parameters.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Set

logger = logging.getLogger("qllm.evaluation")


class EvaluationConfig:
    """
    Configuration for evaluation runs.
    
    This class defines the structure and validation for evaluation parameters,
    providing a consistent interface for configuring evaluation runs.
    """
    
    def __init__(
        self,
        model_id: str,
        metrics: List[str],
        dataset: Optional[str] = None,
        data_path: Optional[str] = None,
        output_dir: str = "./evaluation_results",
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
        precision: str = "float16",
        max_length: int = 2048,
        seed: int = 42,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize evaluation configuration.
        
        Args:
            model_id: Identifier or path for the model to evaluate
            metrics: List of metrics to evaluate
            dataset: Name of the dataset to use (if using a standard dataset)
            data_path: Path to evaluation data (if using custom data)
            output_dir: Directory to store evaluation results
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate (None for all)
            device: Device to run evaluation on ('cuda', 'cpu', 'mps', or None for auto)
            precision: Precision to use for model parameters ('float16', 'float32', 'bfloat16')
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
            metadata: Additional metadata to include in results
            **kwargs: Additional configuration parameters
        """
        # Core parameters
        self.model_id = model_id
        self.metrics = metrics
        self.dataset = dataset
        self.data_path = data_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = device
        self.precision = precision
        self.max_length = max_length
        self.seed = seed
        self.metadata = metadata or {}
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate configuration
        self._validate()
    
    def _validate(self) -> None:
        """Validate the configuration."""
        # Check required parameters
        if not self.model_id:
            raise ValueError("Model ID is required")
        
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
        
        # Check that either dataset or data_path is provided
        if not self.dataset and not self.data_path:
            raise ValueError("Either dataset or data_path must be provided")
        
        # Check numeric parameters
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
        
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"Max samples must be positive: {self.max_samples}")
        
        if self.max_length <= 0:
            raise ValueError(f"Max length must be positive: {self.max_length}")
        
        # Check device
        if self.device is not None and self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Check precision
        if self.precision not in ["float16", "float32", "bfloat16"]:
            raise ValueError(f"Invalid precision: {self.precision}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_id": self.model_id,
            "metrics": self.metrics,
            "dataset": self.dataset,
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "device": self.device,
            "precision": self.precision,
            "max_length": self.max_length,
            "seed": self.seed,
            "metadata": self.metadata,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ["model_id", "metrics", "dataset", "data_path", 
                           "output_dir", "batch_size", "max_samples", 
                           "device", "precision", "max_length", "seed", 
                           "metadata"]}
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(self.to_json())
        
        logger.info(f"Saved evaluation configuration to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationConfig':
        """Create configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationConfig':
        """Load configuration from file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class MetricConfig:
    """
    Configuration for individual metrics.
    
    This class defines the parameters for configuring specific metrics,
    including weight, thresholds, and other metric-specific parameters.
    """
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        thresholds: Optional[Dict[str, float]] = None,
        enabled: bool = True,
        **kwargs
    ):
        """
        Initialize metric configuration.
        
        Args:
            name: Name of the metric
            weight: Weight of the metric in aggregated scores
            thresholds: Thresholds for the metric (e.g., {'good': 0.8, 'acceptable': 0.6})
            enabled: Whether the metric is enabled
            **kwargs: Additional metric-specific parameters
        """
        self.name = name
        self.weight = weight
        self.thresholds = thresholds or {}
        self.enabled = enabled
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "weight": self.weight,
            "thresholds": self.thresholds,
            "enabled": self.enabled,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ["name", "weight", "thresholds", "enabled"]}
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetricConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class DatasetConfig:
    """
    Configuration for evaluation datasets.
    
    This class defines the parameters for configuring datasets used
    in evaluation, including filtering, preprocessing, and splitting.
    """
    
    def __init__(
        self,
        name: str,
        path: Optional[str] = None,
        split: str = "test",
        filter_criteria: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize dataset configuration.
        
        Args:
            name: Name of the dataset
            path: Path to the dataset (if using custom data)
            split: Split to use ('train', 'validation', 'test')
            filter_criteria: Criteria for filtering samples
            max_samples: Maximum number of samples to use
            preprocessing: Preprocessing parameters
            **kwargs: Additional dataset-specific parameters
        """
        self.name = name
        self.path = path
        self.split = split
        self.filter_criteria = filter_criteria or {}
        self.max_samples = max_samples
        self.preprocessing = preprocessing or {}
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "split": self.split,
            "filter_criteria": self.filter_criteria,
            "max_samples": self.max_samples,
            "preprocessing": self.preprocessing,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ["name", "path", "split", "filter_criteria", 
                           "max_samples", "preprocessing"]}
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def create_default_config(
    model_id: str,
    metrics: Optional[List[str]] = None,
    dataset: Optional[str] = None,
    output_dir: Optional[str] = None
) -> EvaluationConfig:
    """
    Create a default evaluation configuration.
    
    Args:
        model_id: Identifier or path for the model to evaluate
        metrics: List of metrics to evaluate (default: basic metrics)
        dataset: Name of the dataset to use (default: None)
        output_dir: Directory to store evaluation results (default: auto-generated)
        
    Returns:
        Default evaluation configuration
    """
    # Default metrics
    if metrics is None:
        metrics = ["perplexity", "accuracy", "f1_score"]
    
    # Auto-generate output directory
    if output_dir is None:
        model_name = os.path.basename(model_id.rstrip("/"))
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./evaluation_results/{model_name}_{timestamp}"
    
    return EvaluationConfig(
        model_id=model_id,
        metrics=metrics,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=8,
        max_samples=None,
        device=None,  # Auto-detect
        precision="float16",
        max_length=2048,
        seed=42
    )
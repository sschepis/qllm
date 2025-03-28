"""
Comprehensive Evaluation Suite for QLLM.

This module provides a comprehensive evaluation suite that extends the
base evaluation suite with more advanced functionality, including
support for multiple metric categories, comparative analysis, and
advanced visualizations.
"""

import os
import json
import time
import logging
import itertools
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Set

import torch
import numpy as np

from src.evaluation.core.config import EvaluationConfig
from src.evaluation.core.suite import EvaluationSuite
from src.evaluation.core.model_utils import ModelUtils


logger = logging.getLogger("qllm.evaluation")


class ComprehensiveSuite(EvaluationSuite):
    """
    Comprehensive evaluation suite with advanced functionality.
    
    This suite extends the base evaluation suite with support for multiple
    metric categories, comparative analysis, and advanced visualizations.
    It provides a more detailed and structured approach to model evaluation.
    """
    
    # Define metric categories
    METRIC_CATEGORIES = {
        "general": [
            "perplexity", 
            "accuracy", 
            "f1_score", 
            "bleu_score", 
            "rouge_score"
        ],
        "compositional": [
            "compositional_accuracy", 
            "structural_consistency", 
            "tree_validity"
        ],
        "emergent": [
            "emergence_score", 
            "pattern_recognition", 
            "abstraction_level"
        ],
        "memory": [
            "context_retention", 
            "long_range_dependency", 
            "information_retrieval"
        ],
        "multimodal": [
            "image_text_alignment", 
            "cross_modal_coherence", 
            "multimodal_inference"
        ],
        "quantum": [
            "quantum_entanglement", 
            "superposition_score", 
            "quantum_fidelity"
        ],
        "resonance": [
            "resonance_magnitude", 
            "harmonic_consistency", 
            "phase_alignment"
        ]
    }
    
    def __init__(
        self,
        config: EvaluationConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        metrics_registry: Optional[Dict[str, Callable]] = None,
        categories: Optional[List[str]] = None,
        comparative_models: Optional[List[Dict[str, Any]]] = None,
        visualize_results: bool = True
    ):
        """
        Initialize the comprehensive evaluation suite.
        
        Args:
            config: Evaluation configuration
            model: Pre-loaded model (if None, will load from config)
            tokenizer: Pre-loaded tokenizer (if None, will load with model)
            metrics_registry: Registry of metric functions (name -> function)
            categories: List of metric categories to include (None for all)
            comparative_models: List of models to compare against
            visualize_results: Whether to generate visualizations
        """
        # Initialize base evaluation suite
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            metrics_registry=metrics_registry
        )
        
        # Set categories (all if not specified)
        self.categories = categories or list(self.METRIC_CATEGORIES.keys())
        
        # Filter metrics to selected categories
        self._filter_metrics_by_categories()
        
        # Set up comparative models
        self.comparative_models = comparative_models or []
        self.comparative_results = []
        
        # Set visualization flag
        self.visualize_results = visualize_results
        
        # Initialize data caches
        self.category_scores = {}
    
    def _filter_metrics_by_categories(self) -> None:
        """
        Filter metrics to only include those in selected categories.
        """
        # Get all metrics from selected categories
        category_metrics = []
        for category in self.categories:
            if category in self.METRIC_CATEGORIES:
                category_metrics.extend(self.METRIC_CATEGORIES[category])
        
        # Filter config metrics to intersection with category metrics
        if self.config.metrics == ["all"]:
            # Use all metrics from selected categories
            self.config.metrics = category_metrics
        else:
            # Filter to metrics in both config and selected categories
            self.config.metrics = [
                metric for metric in self.config.metrics
                if metric in category_metrics
            ]
        
        logger.info(f"Using metrics: {self.config.metrics}")
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare evaluation data for multiple categories.
        
        Returns:
            Dictionary mapping data keys to prepared data
        """
        logger.info("Preparing comprehensive evaluation data")
        
        # Load dataset based on config
        if self.config.dataset:
            # Load standard dataset
            data = self._load_standard_dataset()
        elif self.config.data_path:
            # Load custom data
            data = self._load_custom_data()
        else:
            raise ValueError("Either dataset or data_path must be provided")
        
        # Prepare category-specific data
        category_data = {}
        for category in self.categories:
            try:
                category_data[category] = self._prepare_category_data(category, data)
            except Exception as e:
                logger.error(f"Error preparing data for category {category}: {e}")
                category_data[category] = None
        
        # Return combined data
        return {
            "raw_data": data,
            "category_data": category_data
        }
    
    def _load_standard_dataset(self) -> Any:
        """
        Load a standard dataset.
        
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading standard dataset: {self.config.dataset}")
        
        try:
            # Try to load using Hugging Face datasets
            from datasets import load_dataset
            
            # Load dataset
            dataset = load_dataset(self.config.dataset)
            
            # Apply filter if specified
            if hasattr(self.config, "dataset_filter") and self.config.dataset_filter:
                dataset = dataset.filter(self.config.dataset_filter)
            
            # Limit samples if specified
            if self.config.max_samples is not None:
                dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))
            
            return dataset
            
        except ImportError:
            logger.error("Could not import datasets library")
            raise ImportError("Hugging Face datasets library is required")
        
        except Exception as e:
            logger.error(f"Error loading standard dataset: {e}")
            raise
    
    def _load_custom_data(self) -> Any:
        """
        Load custom evaluation data.
        
        Returns:
            Loaded data
        """
        logger.info(f"Loading custom data from: {self.config.data_path}")
        
        # Check if path exists
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data path not found: {self.config.data_path}")
        
        # Handle different file formats
        if os.path.isfile(self.config.data_path):
            # Single file
            _, ext = os.path.splitext(self.config.data_path)
            
            if ext.lower() == '.json':
                # Load JSON file
                with open(self.config.data_path, 'r') as f:
                    data = json.load(f)
                return data
                
            elif ext.lower() == '.jsonl':
                # Load JSONL file
                data = []
                with open(self.config.data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
                
            elif ext.lower() in ['.txt', '.csv', '.tsv']:
                # Load text file
                with open(self.config.data_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if ext.lower() == '.csv':
                    # Parse CSV
                    import csv
                    with open(self.config.data_path, 'r') as f:
                        reader = csv.DictReader(f)
                        data = list(reader)
                    return data
                    
                elif ext.lower() == '.tsv':
                    # Parse TSV
                    import csv
                    with open(self.config.data_path, 'r') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        data = list(reader)
                    return data
                    
                else:
                    # Simple text file
                    return lines
            
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        elif os.path.isdir(self.config.data_path):
            # Directory of files
            data = []
            
            # Process all files in directory
            for root, _, files in os.walk(self.config.data_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    _, ext = os.path.splitext(filename)
                    
                    # Skip non-text files
                    if ext.lower() not in ['.json', '.jsonl', '.txt', '.csv', '.tsv']:
                        continue
                    
                    try:
                        # Load file
                        with open(filepath, 'r') as f:
                            if ext.lower() == '.json':
                                file_data = json.load(f)
                                data.append(file_data)
                            elif ext.lower() == '.jsonl':
                                for line in f:
                                    if line.strip():
                                        data.append(json.loads(line))
                            else:
                                # Text file
                                data.append({"text": f.read(), "filename": filename})
                    except Exception as e:
                        logger.warning(f"Error loading file {filepath}: {e}")
            
            return data
            
        else:
            raise ValueError(f"Invalid data path: {self.config.data_path}")
    
    def _prepare_category_data(self, category: str, data: Any) -> Dict[str, Any]:
        """
        Prepare data for a specific metric category.
        
        Args:
            category: Category name
            data: Raw data
            
        Returns:
            Prepared data for the category
        """
        # Default implementation uses the same data for all categories
        # Specialized preparation can be implemented for specific categories
        
        if category == "general":
            # General metrics typically use raw text
            return self._prepare_general_data(data)
            
        elif category == "compositional":
            # Compositional metrics may need structured data
            return self._prepare_compositional_data(data)
            
        elif category == "emergent":
            # Emergent metrics may need specialized inputs
            return self._prepare_emergent_data(data)
            
        elif category == "memory":
            # Memory metrics need long contexts
            return self._prepare_memory_data(data)
            
        elif category == "multimodal":
            # Multimodal metrics need image-text pairs
            return self._prepare_multimodal_data(data)
            
        elif category == "quantum":
            # Quantum metrics need specialized encoding
            return self._prepare_quantum_data(data)
            
        elif category == "resonance":
            # Resonance metrics need specialized inputs
            return self._prepare_resonance_data(data)
            
        else:
            # Default for unknown categories
            return data
    
    def _prepare_general_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for general metrics."""
        # Extract text samples from data
        text_samples = self._extract_text_samples(data)
        
        # Tokenize samples
        tokenized_samples = []
        for text in text_samples:
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    text, 
                    max_length=self.config.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                tokenized_samples.append(tokens)
        
        return {
            "text_samples": text_samples,
            "tokenized_samples": tokenized_samples
        }
    
    def _prepare_compositional_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for compositional metrics."""
        # Extract text samples
        text_samples = self._extract_text_samples(data)
        
        # Filter for samples with compositional elements
        compositional_samples = [
            text for text in text_samples
            if "if" in text.lower() or "then" in text.lower() or "=" in text
        ]
        
        return {
            "text_samples": compositional_samples
        }
    
    def _prepare_emergent_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for emergent metrics."""
        # Use general data preparation
        return self._prepare_general_data(data)
    
    def _prepare_memory_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for memory metrics."""
        # Extract text samples
        text_samples = self._extract_text_samples(data)
        
        # Filter for longer samples
        long_samples = [
            text for text in text_samples
            if len(text.split()) > 100
        ]
        
        return {
            "text_samples": long_samples
        }
    
    def _prepare_multimodal_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for multimodal metrics."""
        # Extract image-text pairs if available
        image_text_pairs = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "image" in item and "text" in item:
                    image_text_pairs.append(item)
        
        return {
            "image_text_pairs": image_text_pairs
        }
    
    def _prepare_quantum_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for quantum metrics."""
        # Use general data preparation
        return self._prepare_general_data(data)
    
    def _prepare_resonance_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for resonance metrics."""
        # Use general data preparation
        return self._prepare_general_data(data)
    
    def _extract_text_samples(self, data: Any) -> List[str]:
        """
        Extract text samples from data.
        
        Args:
            data: Raw data
            
        Returns:
            List of text samples
        """
        text_samples = []
        
        # Handle different data formats
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    # Item is a string
                    text_samples.append(item)
                elif isinstance(item, dict):
                    # Item is a dictionary, look for text field
                    for key in ["text", "content", "prompt", "input", "source"]:
                        if key in item and isinstance(item[key], str):
                            text_samples.append(item[key])
                            break
        elif isinstance(data, dict):
            # Look for text fields in dictionary
            for key, value in data.items():
                if isinstance(value, str):
                    text_samples.append(value)
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    text_samples.extend(value)
        
        # Limit samples if needed
        if self.config.max_samples is not None:
            text_samples = text_samples[:self.config.max_samples]
        
        return text_samples
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the comprehensive evaluation.
        
        This method extends the base evaluation to add category-specific
        scoring and comparative analysis.
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Run base evaluation
        results = super().run_evaluation()
        
        # Calculate category-specific scores
        category_scores = self._calculate_category_scores(results["metrics"])
        results["category_scores"] = category_scores
        
        # Run comparative analysis if requested
        if self.comparative_models:
            comparative_results = self._run_comparative_analysis()
            results["comparative_analysis"] = comparative_results
        
        # Generate visualizations if requested
        if self.visualize_results:
            visualization_paths = self._generate_visualizations(results)
            results["visualizations"] = visualization_paths
        
        return results
    
    def _calculate_category_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate scores for each metric category.
        
        Args:
            metrics: Dictionary of metric results
            
        Returns:
            Dictionary mapping categories to scores
        """
        category_scores = {}
        
        # Calculate scores for each category
        for category, category_metrics in self.METRIC_CATEGORIES.items():
            # Skip categories that weren't evaluated
            if category not in self.categories:
                continue
            
            # Get results for metrics in this category
            category_results = []
            for metric in category_metrics:
                if metric in metrics and "value" in metrics[metric]:
                    category_results.append(metrics[metric]["value"])
            
            # Calculate category score if we have results
            if category_results:
                category_scores[category] = sum(category_results) / len(category_results)
        
        # Store category scores
        self.category_scores = category_scores
        
        return category_scores
    
    def _run_comparative_analysis(self) -> List[Dict[str, Any]]:
        """
        Run comparative analysis against other models.
        
        Returns:
            List of comparative results
        """
        comparative_results = []
        
        # Run evaluation for each comparative model
        for model_info in self.comparative_models:
            try:
                # Create configuration for this model
                model_config = EvaluationConfig.from_dict({
                    **self.config.to_dict(),
                    "model_id": model_info["model_id"]
                })
                
                # Create suite for this model
                model_suite = ComprehensiveSuite(
                    config=model_config,
                    categories=self.categories,
                    visualize_results=False  # Skip visualizations for comparative models
                )
                
                # Run evaluation
                model_results = model_suite.run_evaluation()
                
                # Add to comparative results
                comparative_results.append({
                    "model_id": model_info["model_id"],
                    "results": model_results
                })
                
            except Exception as e:
                logger.error(f"Error in comparative analysis for {model_info['model_id']}: {e}")
                comparative_results.append({
                    "model_id": model_info["model_id"],
                    "error": str(e)
                })
        
        # Store comparative results
        self.comparative_results = comparative_results
        
        return comparative_results
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations for evaluation results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualization_paths = {}
        
        try:
            # Import visualization components
            from src.evaluation.visualization import (
                MetricPlotter,
                ComparisonPlotter,
                KnowledgeGraphVisualizer,
                Dashboard
            )
            
            # Create output directory
            vis_dir = os.path.join(self.config.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate metric plots
            metric_plotter = MetricPlotter()
            metric_path = os.path.join(vis_dir, "metrics.png")
            metric_plotter.plot_metrics(results["metrics"], output_path=metric_path)
            visualization_paths["metrics"] = metric_path
            
            # Generate category plots
            if self.category_scores:
                metric_path = os.path.join(vis_dir, "categories.png")
                metric_plotter.plot_categories(self.category_scores, output_path=metric_path)
                visualization_paths["categories"] = metric_path
            
            # Generate comparative plots if available
            if self.comparative_results:
                comparison_plotter = ComparisonPlotter()
                comparison_path = os.path.join(vis_dir, "comparison.png")
                comparison_plotter.plot_comparison(
                    [results] + [r["results"] for r in self.comparative_results if "results" in r],
                    output_path=comparison_path
                )
                visualization_paths["comparison"] = comparison_path
            
            # Generate dashboard
            dashboard = Dashboard()
            dashboard_path = os.path.join(vis_dir, "dashboard.html")
            dashboard.generate(
                results=results,
                comparative_results=self.comparative_results,
                output_path=dashboard_path
            )
            visualization_paths["dashboard"] = dashboard_path
            
        except ImportError as e:
            logger.warning(f"Could not generate visualizations: {e}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualization_paths
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save comprehensive evaluation results to file.
        
        Args:
            filepath: Path to save results to (if None, uses config.output_dir)
            
        Returns:
            Path to the saved results file
        """
        # Use base implementation to save results
        saved_path = super().save_results(filepath)
        
        # Save additional outputs if needed
        if self.visualize_results and "visualizations" in self.results:
            logger.info(f"Visualization outputs saved to {self.config.output_dir}/visualizations")
        
        return saved_path


def run_comprehensive_evaluation(
    model_id: str,
    dataset: Optional[str] = None,
    data_path: Optional[str] = None,
    categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a comprehensive evaluation with default settings.
    
    Args:
        model_id: Identifier or path for the model to evaluate
        dataset: Name of the dataset to use (if using a standard dataset)
        data_path: Path to evaluation data (if using custom data)
        categories: List of metric categories to include (None for all)
        output_dir: Directory to store evaluation results (default: auto-generated)
        
    Returns:
        Dictionary with evaluation results
    """
    # Create default configuration
    config = EvaluationConfig(
        model_id=model_id,
        metrics=["all"],  # Use all available metrics
        dataset=dataset,
        data_path=data_path,
        output_dir=output_dir or f"./evaluation_results/{os.path.basename(model_id)}"
    )
    
    # Create comprehensive suite
    suite = ComprehensiveSuite(
        config=config,
        categories=categories,
        visualize_results=True
    )
    
    # Run evaluation
    results = suite.run_evaluation()
    
    # Save results
    suite.save_results()
    
    return results
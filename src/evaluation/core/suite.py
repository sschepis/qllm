"""
Core evaluation suite for QLLM extensions.

This module provides the main EvaluationSuite class for coordinating
comprehensive evaluations of QLLM models and their extensions.
"""

import os
import time
import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Union

# Import from refactored modules
from src.evaluation.core.model_utils import initialize_evaluation_model
from src.evaluation.core.config import EvaluationConfig
from src.evaluation.utils.serialization import make_serializable, save_results_to_json
from src.evaluation.metrics import (
    # General metrics
    perplexity,
    parameter_efficiency,
    memory_usage,
    inference_speed,
    generation_diversity,
    
    # Extension-specific evaluations
    evaluate_multimodal_extension,
    evaluate_memory_extension,
    evaluate_quantum_extension
)


class EvaluationSuite:
    """
    Comprehensive evaluation suite for QLLM extensions.
    
    This class provides a framework for running evaluations on the QLLM model
    and its extensions, collecting metrics, and reporting results.
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        model_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "evaluation_results",
        device: str = None
    ):
        """
        Initialize the evaluation suite.
        
        Args:
            model: Pre-initialized model (optional)
            model_config: Configuration for creating a model if none provided
            output_dir: Directory for storing evaluation results
            device: Device to run evaluation on (defaults to auto-detection)
        """
        self.model = model
        self.model_config = model_config or {}
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            "multimodal": {},
            "memory": {},
            "quantum": {},
            "integrated": {},
            "baseline": {}
        }
        
        # Initialize model if not provided
        if self.model is None and self.model_config:
            self.model = initialize_evaluation_model(self.model_config, self.device)
    
    def run_evaluation(
        self,
        eval_config: Union[Dict[str, Any], EvaluationConfig]
    ) -> Dict[str, Any]:
        """
        Run a comprehensive evaluation based on configuration.
        
        Args:
            eval_config: Configuration for the evaluation run
                {
                    "extensions_to_evaluate": ["multimodal", "memory", "quantum"],
                    "metrics": ["perplexity", "parameter_efficiency", ...],
                    "datasets": {...},
                    "ablation_studies": [...],
                    ...
                }
            
        Returns:
            Dictionary of evaluation results
        """
        print("\n=== Starting Comprehensive Evaluation ===\n")
        
        # Convert dict config to EvaluationConfig if needed
        if isinstance(eval_config, dict):
            config = EvaluationConfig(eval_config)
        else:
            config = eval_config
        
        # Extract configuration
        extensions = config.get("extensions_to_evaluate", ["multimodal", "memory", "quantum"])
        metrics = config.get("metrics", ["perplexity", "parameter_efficiency"])
        datasets = config.get("datasets", {})
        run_ablation = config.get("run_ablation_studies", True)
        
        # Ensure model is available
        if self.model is None:
            raise ValueError("Model not initialized. Provide a model or model_config.")
        
        # Run baseline evaluation (no extensions)
        if "baseline" in extensions or run_ablation:
            print("\nEvaluating baseline model (no extensions)...")
            self.results["baseline"] = self._evaluate_configuration(
                enabled_extensions=[],
                metrics=metrics,
                datasets=datasets
            )
        
        # Evaluate each extension individually
        for ext in ["multimodal", "memory", "quantum"]:
            if ext in extensions:
                print(f"\nEvaluating {ext} extension...")
                self.results[ext] = self._evaluate_configuration(
                    enabled_extensions=[ext],
                    metrics=metrics,
                    datasets=datasets
                )
        
        # Evaluate all extensions together
        if "integrated" in extensions or len(extensions) > 1:
            print("\nEvaluating all extensions together...")
            enabled = [ext for ext in ["multimodal", "memory", "quantum"] if ext in extensions]
            self.results["integrated"] = self._evaluate_configuration(
                enabled_extensions=enabled,
                metrics=metrics,
                datasets=datasets
            )
        
        # Run ablation studies if requested
        if run_ablation:
            print("\nRunning ablation studies...")
            self._run_ablation_studies(metrics, datasets)
        
        # Generate summary report
        summary = self._generate_summary()
        
        # Save results
        self._save_results()
        
        print("\n=== Evaluation Completed ===\n")
        return summary
    
    def _evaluate_configuration(
        self,
        enabled_extensions: List[str],
        metrics: List[str],
        datasets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a specific configuration of extensions.
        
        Args:
            enabled_extensions: List of extensions to enable
            metrics: List of metrics to compute
            datasets: Datasets to use for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        # Configure model with specified extensions
        self.model.reset_extensions()
        
        # Enable specified extensions
        for ext in enabled_extensions:
            if ext == "multimodal":
                self.model.enable_multimodal_extension()
            elif ext == "memory":
                self.model.enable_memory_extension()
            elif ext == "quantum":
                self.model.enable_quantum_extension()
        
        # Initialize results
        results = {
            "enabled_extensions": enabled_extensions,
            "metrics": {}
        }
        
        # Evaluate each metric
        for metric in metrics:
            print(f"  Evaluating metric: {metric}")
            
            if metric == "perplexity":
                results["metrics"][metric] = self._evaluate_perplexity(datasets.get("text", []))
            
            elif metric == "parameter_efficiency":
                results["metrics"][metric] = parameter_efficiency(self.model)
            
            elif metric == "memory_usage":
                results["metrics"][metric] = memory_usage(self.model)
            
            elif metric == "inference_speed":
                results["metrics"][metric] = inference_speed(
                    self.model, 
                    datasets.get("inference_inputs", [])
                )
            
            elif metric == "generation_diversity":
                results["metrics"][metric] = generation_diversity(
                    self.model,
                    datasets.get("generation_prompts", [])
                )
        
        # Extension-specific evaluations
        if "multimodal" in enabled_extensions:
            results["multimodal_metrics"] = evaluate_multimodal_extension(self.model, datasets)
        
        if "memory" in enabled_extensions:
            results["memory_metrics"] = evaluate_memory_extension(self.model, datasets)
        
        if "quantum" in enabled_extensions:
            results["quantum_metrics"] = evaluate_quantum_extension(self.model, datasets)
        
        return results
    
    def _evaluate_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate perplexity on test texts.
        
        Args:
            test_texts: List of texts for perplexity evaluation
            
        Returns:
            Dictionary with perplexity metrics
        """
        if not test_texts:
            return {"error": "No test texts provided for perplexity evaluation"}
        
        perplexities = []
        
        for text in test_texts:
            try:
                ppl = perplexity(self.model, text)
                perplexities.append(ppl)
            except Exception as e:
                print(f"Error calculating perplexity: {str(e)}")
        
        if not perplexities:
            return {"error": "Failed to calculate perplexity for any test texts"}
        
        return {
            "mean": sum(perplexities) / len(perplexities),
            "min": min(perplexities),
            "max": max(perplexities),
            "values": perplexities
        }
    
    def _run_ablation_studies(
        self,
        metrics: List[str],
        datasets: Dict[str, Any]
    ) -> None:
        """
        Run ablation studies to measure contributions of each extension.
        
        Args:
            metrics: List of metrics to evaluate
            datasets: Evaluation datasets
        """
        ablation_results = {}
        
        # Baseline (no extensions)
        ablation_results["none"] = self._evaluate_configuration([], metrics, datasets)
        
        # Individual extensions
        for ext in ["multimodal", "memory", "quantum"]:
            ablation_results[ext] = self._evaluate_configuration([ext], metrics, datasets)
        
        # Pairs of extensions
        pairs = [
            ["multimodal", "memory"],
            ["multimodal", "quantum"],
            ["memory", "quantum"]
        ]
        
        for pair in pairs:
            pair_name = "+".join(pair)
            ablation_results[pair_name] = self._evaluate_configuration(pair, metrics, datasets)
        
        # All extensions
        ablation_results["all"] = self._evaluate_configuration(
            ["multimodal", "memory", "quantum"], 
            metrics, 
            datasets
        )
        
        # Store ablation results
        self.results["ablation_studies"] = ablation_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of evaluation results.
        
        Returns:
            Dictionary with evaluation summary
        """
        summary = {
            "configurations_evaluated": list(self.results.keys()),
            "metrics_evaluated": set()
        }
        
        # Collect all metrics evaluated
        for config_name, config_results in self.results.items():
            if config_name == "ablation_studies":
                continue
                
            if "metrics" in config_results:
                for metric in config_results["metrics"].keys():
                    summary["metrics_evaluated"].add(metric)
        
        summary["metrics_evaluated"] = list(summary["metrics_evaluated"])
        
        # Get best configuration for each metric
        summary["best_configurations"] = {}
        
        for metric in summary["metrics_evaluated"]:
            best_config = None
            best_value = None
            
            for config_name, config_results in self.results.items():
                if config_name == "ablation_studies":
                    continue
                    
                if "metrics" in config_results and metric in config_results["metrics"]:
                    metric_data = config_results["metrics"][metric]
                    
                    # Extract value based on metric format
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        value = metric_data["mean"]
                    elif isinstance(metric_data, dict) and "value" in metric_data:
                        value = metric_data["value"]
                    elif isinstance(metric_data, (int, float)):
                        value = metric_data
                    else:
                        continue
                    
                    # Determine if this is better
                    # Note: For some metrics like perplexity, lower is better
                    # We assume lower is better for "perplexity" metric
                    if metric == "perplexity":
                        if best_value is None or value < best_value:
                            best_value = value
                            best_config = config_name
                    else:
                        # For other metrics, higher is assumed to be better
                        if best_value is None or value > best_value:
                            best_value = value
                            best_config = config_name
            
            if best_config and best_value is not None:
                summary["best_configurations"][metric] = {
                    "configuration": best_config,
                    "value": best_value
                }
        
        # Add ablation insights if available
        if "ablation_studies" in self.results:
            summary["ablation_insights"] = self._analyze_ablation_studies()
        
        return summary
    
    def _analyze_ablation_studies(self) -> Dict[str, Any]:
        """
        Analyze ablation studies to determine extension contributions.
        
        Returns:
            Dictionary with ablation analysis
        """
        if "ablation_studies" not in self.results:
            return {}
            
        ablation_studies = self.results["ablation_studies"]
        insights = {}
        
        # Check which metrics we have in the ablation studies
        available_metrics = set()
        for config_results in ablation_studies.values():
            if "metrics" in config_results:
                for metric in config_results["metrics"].keys():
                    available_metrics.add(metric)
        
        for metric in available_metrics:
            # Extract baseline value (no extensions)
            if "none" in ablation_studies and "metrics" in ablation_studies["none"] and metric in ablation_studies["none"]["metrics"]:
                baseline_data = ablation_studies["none"]["metrics"][metric]
                
                if isinstance(baseline_data, dict) and "mean" in baseline_data:
                    baseline = baseline_data["mean"]
                elif isinstance(baseline_data, dict) and "value" in baseline_data:
                    baseline = baseline_data["value"]
                elif isinstance(baseline_data, (int, float)):
                    baseline = baseline_data
                else:
                    continue
                
                contributions = {}
                
                # Calculate individual contributions
                for ext in ["multimodal", "memory", "quantum"]:
                    if ext in ablation_studies and "metrics" in ablation_studies[ext] and metric in ablation_studies[ext]["metrics"]:
                        ext_data = ablation_studies[ext]["metrics"][metric]
                        
                        if isinstance(ext_data, dict) and "mean" in ext_data:
                            ext_value = ext_data["mean"]
                        elif isinstance(ext_data, dict) and "value" in ext_data:
                            ext_value = ext_data["value"]
                        elif isinstance(ext_data, (int, float)):
                            ext_value = ext_data
                        else:
                            continue
                            
                        # Calculate improvement (or deterioration)
                        if metric == "perplexity":  # Lower is better
                            contribution = baseline - ext_value
                            percent = (baseline - ext_value) / baseline * 100 if baseline > 0 else 0
                        else:  # Higher is better
                            contribution = ext_value - baseline
                            percent = (ext_value - baseline) / baseline * 100 if baseline > 0 else 0
                            
                        contributions[ext] = {
                            "absolute": contribution,
                            "percent": percent
                        }
                
                insights[metric] = {
                    "baseline": baseline,
                    "contributions": contributions
                }
                
                # Add best combination
                best_combo = None
                best_value = None
                
                for combo_name, combo_results in ablation_studies.items():
                    if combo_name == "none" or combo_name in ["multimodal", "memory", "quantum"]:
                        continue
                        
                    if "metrics" in combo_results and metric in combo_results["metrics"]:
                        combo_data = combo_results["metrics"][metric]
                        
                        if isinstance(combo_data, dict) and "mean" in combo_data:
                            combo_value = combo_data["mean"]
                        elif isinstance(combo_data, dict) and "value" in combo_data:
                            combo_value = combo_data["value"]
                        elif isinstance(combo_data, (int, float)):
                            combo_value = combo_data
                        else:
                            continue
                            
                        if metric == "perplexity":  # Lower is better
                            if best_value is None or combo_value < best_value:
                                best_value = combo_value
                                best_combo = combo_name
                        else:  # Higher is better
                            if best_value is None or combo_value > best_value:
                                best_value = combo_value
                                best_combo = combo_name
                
                if best_combo and best_value is not None:
                    insights[metric]["best_combination"] = {
                        "combination": best_combo,
                        "value": best_value
                    }
        
        return insights
    
    def _save_results(self) -> None:
        """Save evaluation results to JSON file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        
        # Create a serializable version of the results
        serializable_results = make_serializable(self.results)
        
        # Save to file
        save_results_to_json(serializable_results, output_file)
        
        # Create a timestamped directory for additional results
        eval_dir = os.path.join(self.output_dir, f"evaluation_{timestamp.replace('-', '_')}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save evaluation configuration
        if hasattr(self, 'config'):
            config_file = os.path.join(eval_dir, "evaluation_config.json")
            with open(config_file, 'w') as f:
                json.dump(make_serializable(self.config), f, indent=2)
        
        # Save evaluation summary
        summary = self._generate_summary()
        summary_file = os.path.join(eval_dir, "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(make_serializable(summary), f, indent=2)
        
        # Create visualizations directory
        vis_dir = os.path.join(eval_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"Results saved to {output_file}")
        print(f"Summary saved to {summary_file}")


def run_evaluation_suite(config_path: str) -> Dict[str, Any]:
    """
    Run the evaluation suite using a configuration file.
    
    Args:
        config_path: Path to evaluation configuration JSON file
        
    Returns:
        Evaluation results summary
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create evaluation suite
    suite = EvaluationSuite(
        model_config=config.get("model_config", {}),
        output_dir=config.get("output_dir", "evaluation_results"),
        device=config.get("device")
    )
    
    # Run evaluation
    summary = suite.run_evaluation(config)
    
    return summary
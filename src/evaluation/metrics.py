"""
Centralized metrics utilities for QLLM evaluation.

This module provides utilities for managing, registering, and computing
metrics for model evaluation, centralizing functionality that was previously
duplicated across different metrics implementations.
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, List, Tuple, Union, Optional, Callable

import torch
import numpy as np


logger = logging.getLogger("qllm.evaluation")


# Global registry of metrics
_METRICS_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: str, metric_fn: Callable) -> None:
    """
    Register a metric function in the global registry.
    
    Args:
        name: Name of the metric
        metric_fn: Function implementing the metric
    """
    if name in _METRICS_REGISTRY:
        logger.warning(f"Overwriting existing metric: {name}")
    
    _METRICS_REGISTRY[name] = metric_fn
    logger.debug(f"Registered metric: {name}")


def get_metric_registry() -> Dict[str, Callable]:
    """
    Get the global metrics registry.
    
    Returns:
        Dictionary mapping metric names to metric functions
    """
    # Populate registry if it's empty
    if not _METRICS_REGISTRY:
        _populate_registry()
    
    return _METRICS_REGISTRY


def _populate_registry() -> None:
    """
    Populate the metrics registry with all available metrics.
    """
    # Import all metric categories to register their metrics
    try:
        # Import general metrics
        from src.evaluation.metrics.general import (
            perplexity, 
            accuracy, 
            f1_score,
            bleu_score,
            rouge_score
        )
        
        # Register general metrics
        register_metric("perplexity", perplexity)
        register_metric("accuracy", accuracy)
        register_metric("f1_score", f1_score)
        register_metric("bleu_score", bleu_score)
        register_metric("rouge_score", rouge_score)
        
        # Import and register compositional metrics
        from src.evaluation.metrics.compositional import (
            compositional_accuracy,
            structural_consistency,
            tree_validity
        )
        
        register_metric("compositional_accuracy", compositional_accuracy)
        register_metric("structural_consistency", structural_consistency)
        register_metric("tree_validity", tree_validity)
        
        # Import and register emergent metrics
        from src.evaluation.metrics.emergent import (
            emergence_score,
            pattern_recognition,
            abstraction_level
        )
        
        register_metric("emergence_score", emergence_score)
        register_metric("pattern_recognition", pattern_recognition)
        register_metric("abstraction_level", abstraction_level)
        
        # Import and register memory metrics
        from src.evaluation.metrics.memory import (
            context_retention,
            long_range_dependency,
            information_retrieval
        )
        
        register_metric("context_retention", context_retention)
        register_metric("long_range_dependency", long_range_dependency)
        register_metric("information_retrieval", information_retrieval)
        
        # Import and register multimodal metrics if available
        try:
            from src.evaluation.metrics.multimodal import (
                image_text_alignment,
                cross_modal_coherence,
                multimodal_inference
            )
            
            register_metric("image_text_alignment", image_text_alignment)
            register_metric("cross_modal_coherence", cross_modal_coherence)
            register_metric("multimodal_inference", multimodal_inference)
        except ImportError:
            logger.info("Multimodal metrics not available")
        
        # Import and register quantum metrics if available
        try:
            from src.evaluation.metrics.quantum import (
                quantum_entanglement,
                superposition_score,
                quantum_fidelity
            )
            
            register_metric("quantum_entanglement", quantum_entanglement)
            register_metric("superposition_score", superposition_score)
            register_metric("quantum_fidelity", quantum_fidelity)
        except ImportError:
            logger.info("Quantum metrics not available")
        
        # Import and register resonance metrics if available
        try:
            from src.evaluation.metrics.resonance import (
                resonance_magnitude,
                harmonic_consistency,
                phase_alignment
            )
            
            register_metric("resonance_magnitude", resonance_magnitude)
            register_metric("harmonic_consistency", harmonic_consistency)
            register_metric("phase_alignment", phase_alignment)
        except ImportError:
            logger.info("Resonance metrics not available")
        
    except ImportError as e:
        logger.warning(f"Could not import all metrics: {e}")


def calculate_metrics(
    model: Any,
    tokenizer: Any,
    data: Any,
    metrics: List[str],
    config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Calculate multiple metrics on a model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        data: Evaluation data
        metrics: List of metrics to calculate
        config: Optional configuration for metrics
        
    Returns:
        Dictionary mapping metric names to metric results
    """
    # Get metrics registry
    registry = get_metric_registry()
    
    # Calculate each metric
    results = {}
    for metric_name in metrics:
        try:
            # Get metric function
            metric_fn = registry.get(metric_name)
            if metric_fn is None:
                logger.warning(f"Unknown metric: {metric_name}")
                continue
            
            # Calculate metric
            logger.info(f"Calculating metric: {metric_name}")
            metric_result = metric_fn(model, tokenizer, data, config)
            
            # Ensure result is a dictionary
            if not isinstance(metric_result, dict):
                metric_result = {"value": metric_result}
            
            # Store result
            results[metric_name] = metric_result
            
        except Exception as e:
            logger.error(f"Error calculating metric {metric_name}: {e}")
            results[metric_name] = {"error": str(e)}
    
    return results


def compute_overall_score(
    metric_results: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute an overall score from multiple metric results.
    
    Args:
        metric_results: Dictionary mapping metric names to results
        weights: Optional dictionary mapping metric names to weights
        
    Returns:
        Overall weighted score
    """
    if not metric_results:
        return 0.0
    
    # Use equal weights if not provided
    if weights is None:
        weights = {metric: 1.0 for metric in metric_results}
    
    # Calculate weighted average
    weighted_sum = 0.0
    total_weight = 0.0
    
    for metric_name, result in metric_results.items():
        # Skip metrics with errors
        if "error" in result:
            continue
        
        # Extract value
        if "value" in result and isinstance(result["value"], (int, float)):
            value = result["value"]
            
            # Get weight for this metric
            weight = weights.get(metric_name, 1.0)
            
            # Add to weighted sum
            weighted_sum += value * weight
            total_weight += weight
    
    # Return weighted average
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0


# Utility functions for common metric operations

def calculate_perplexity(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate perplexity for a batch of inputs.
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs
        attention_mask: Optional attention mask
        
    Returns:
        Perplexity score (lower is better)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Prepare inputs
    inputs = {"input_ids": input_ids}
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask
    
    # Forward pass with no gradient computation
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
    
    # Extract loss
    if hasattr(outputs, "loss"):
        loss = outputs.loss
    else:
        loss = outputs[0]
    
    # Calculate perplexity
    return torch.exp(loss).item()


def calculate_token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: Predicted token IDs or logits
        targets: Target token IDs
        ignore_index: Index to ignore in targets
        
    Returns:
        Accuracy score (higher is better)
    """
    # Convert logits to predictions if needed
    if len(predictions.shape) > len(targets.shape):
        predictions = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid tokens
    mask = (targets != ignore_index)
    
    # Calculate accuracy
    correct = ((predictions == targets) * mask).sum().item()
    total = mask.sum().item()
    
    # Return accuracy
    if total > 0:
        return correct / total
    else:
        return 0.0


def calculate_sequence_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate sequence-level accuracy (all tokens correct).
    
    Args:
        predictions: Predicted token IDs or logits
        targets: Target token IDs
        ignore_index: Index to ignore in targets
        
    Returns:
        Accuracy score (higher is better)
    """
    # Convert logits to predictions if needed
    if len(predictions.shape) > len(targets.shape):
        predictions = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid tokens
    mask = (targets != ignore_index)
    
    # Calculate sequence accuracy
    correct_seqs = 0
    total_seqs = targets.size(0)
    
    for i in range(total_seqs):
        seq_mask = mask[i]
        if seq_mask.sum() == 0:
            # Skip sequences with no valid tokens
            total_seqs -= 1
            continue
        
        # Check if all valid tokens match
        if ((predictions[i] == targets[i]) * seq_mask).sum() == seq_mask.sum():
            correct_seqs += 1
    
    # Return accuracy
    if total_seqs > 0:
        return correct_seqs / total_seqs
    else:
        return 0.0


def text_generation_metrics(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    references: List[str],
    max_new_tokens: int = 100,
    **generation_kwargs
) -> Dict[str, float]:
    """
    Calculate metrics for text generation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        prompts: List of input prompts
        references: List of reference texts
        max_new_tokens: Maximum tokens to generate
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Dictionary with generation metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Generate texts
    generated_texts = []
    for prompt in prompts:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        
        generated_texts.append(generated_text)
    
    # Calculate metrics
    metrics = {}
    
    # BLEU score
    try:
        from nltk.translate.bleu_score import corpus_bleu
        references_tokenized = [[r.split()] for r in references]
        generated_tokenized = [g.split() for g in generated_texts]
        metrics["bleu"] = corpus_bleu(references_tokenized, generated_tokenized)
    except ImportError:
        metrics["bleu"] = None
    
    # ROUGE score
    try:
        from rouge import Rouge
        rouge = Rouge()
        rouge_scores = rouge.get_scores(generated_texts, references, avg=True)
        metrics["rouge"] = rouge_scores
    except ImportError:
        metrics["rouge"] = None
    
    # Exact match
    exact_matches = sum(1 for gen, ref in zip(generated_texts, references) if gen.strip() == ref.strip())
    metrics["exact_match"] = exact_matches / len(prompts) if prompts else 0
    
    return metrics
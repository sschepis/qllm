"""
Quantum extension evaluation metrics for QLLM models.

This module provides metrics specifically for evaluating quantum symmetry mask
and related capabilities of QLLM models.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def quantum_efficiency_gain(
    model: torch.nn.Module,
    test_inputs: List[str]
) -> Dict[str, float]:
    """
    Measure efficiency gains from quantum extensions.
    
    Args:
        model: The model to evaluate
        test_inputs: List of input texts
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "quantum_extension") or model.quantum_extension is None:
        return {"error": "Quantum extension not enabled"}
    
    # Measure baseline metrics with quantum extension disabled
    original_state = model.quantum_extension.masks_applied
    model.quantum_extension.disable_masks()
    
    baseline_times = []
    for text in test_inputs:
        start_time = time.time()
        with torch.no_grad():
            _ = model(text)
        baseline_times.append(time.time() - start_time)
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    
    # Measure metrics with quantum extension enabled
    model.quantum_extension.apply_masks()
    
    masked_times = []
    for text in test_inputs:
        start_time = time.time()
        with torch.no_grad():
            _ = model(text)
        masked_times.append(time.time() - start_time)
    
    masked_avg = sum(masked_times) / len(masked_times)
    
    # Restore original state
    if original_state:
        model.quantum_extension.apply_masks()
    else:
        model.quantum_extension.disable_masks()
    
    # Calculate speedup
    speedup = baseline_avg / masked_avg if masked_avg > 0 else 1.0
    
    # Get mask statistics
    mask_stats = model.quantum_extension.get_mask_statistics()
    
    return {
        "baseline_time": baseline_avg,
        "masked_time": masked_avg,
        "speedup": speedup,
        "time_reduction_percent": (1.0 - masked_avg / baseline_avg) * 100 if baseline_avg > 0 else 0,
        "sparsity": mask_stats.get("overall_sparsity", 0.0),
        "mask_stats": mask_stats
    }


def pattern_effectiveness(
    model: torch.nn.Module,
    patterns: List[str]
) -> Dict[str, Any]:
    """
    Evaluate effectiveness of different mask patterns.
    
    Args:
        model: The model to evaluate
        patterns: List of pattern types to evaluate
        
    Returns:
        Dictionary of pattern effectiveness metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "quantum_extension") or model.quantum_extension is None:
        return {"error": "Quantum extension not enabled"}
    
    # Save original pattern
    original_pattern = "harmonic"  # Default
    if hasattr(model.quantum_extension, "get_pattern_type"):
        original_pattern = model.quantum_extension.get_pattern_type()
    
    pattern_metrics = {}
    
    # Evaluate each pattern
    for pattern in patterns:
        # Set pattern type
        model.quantum_extension.set_pattern_type(pattern)
        
        # Apply masks with this pattern
        model.quantum_extension.apply_masks()
        
        # Get mask statistics
        mask_stats = model.quantum_extension.get_mask_statistics()
        
        pattern_metrics[pattern] = {
            "sparsity": mask_stats.get("overall_sparsity", 0.0),
            "masked_params": mask_stats.get("masked_params", 0),
            "total_params": mask_stats.get("total_params", 0),
            "pattern_type": pattern
        }
    
    # Restore original pattern
    model.quantum_extension.set_pattern_type(original_pattern)
    model.quantum_extension.apply_masks()
    
    return {
        "pattern_metrics": pattern_metrics,
        "best_pattern": max(patterns, key=lambda p: pattern_metrics[p]["sparsity"]) if patterns else None
    }


def sparsity_accuracy_tradeoff(
    model: torch.nn.Module,
    test_texts: List[str],
    sparsity_levels: List[float]
) -> Dict[str, Any]:
    """
    Measure the tradeoff between sparsity and model accuracy.
    
    Args:
        model: The model to evaluate
        test_texts: List of test inputs
        sparsity_levels: List of sparsity levels to evaluate
        
    Returns:
        Dictionary of tradeoff metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "quantum_extension") or model.quantum_extension is None:
        return {"error": "Quantum extension not enabled"}
    
    # Save original sparsity
    original_sparsity = 0.8  # Default
    if hasattr(model.quantum_extension, "get_base_sparsity"):
        original_sparsity = model.quantum_extension.get_base_sparsity()
    
    tradeoff_data = []
    
    # Get tokenizer
    tokenizer = model.tokenizer
    encoded_texts = [tokenizer(text, return_tensors="pt") for text in test_texts]
    
    # Measure baseline perplexity with no masks
    model.quantum_extension.disable_masks()
    baseline_losses = []
    
    for encoded in encoded_texts:
        with torch.no_grad():
            output = model(**encoded)
            logits = output["logits"] if isinstance(output, dict) else output
            
            # Calculate loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encoded["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            baseline_losses.append(loss.item())
    
    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    
    # Evaluate each sparsity level
    for sparsity in sparsity_levels:
        # Set sparsity
        if hasattr(model.quantum_extension, "set_base_sparsity"):
            model.quantum_extension.set_base_sparsity(sparsity)
        else:
            # If not directly settable, try to approximate
            print(f"Warning: Cannot directly set sparsity to {sparsity}")
        
        # Apply masks
        model.quantum_extension.apply_masks()
        
        # Measure performance
        masked_losses = []
        for encoded in encoded_texts:
            with torch.no_grad():
                output = model(**encoded)
                logits = output["logits"] if isinstance(output, dict) else output
                
                # Calculate loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = encoded["input_ids"][:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                masked_losses.append(loss.item())
        
        masked_loss = sum(masked_losses) / len(masked_losses)
        
        # Get actual sparsity achieved
        mask_stats = model.quantum_extension.get_mask_statistics()
        actual_sparsity = mask_stats.get("overall_sparsity", 0.0)
        
        tradeoff_data.append({
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "loss": masked_loss,
            "loss_increase": masked_loss - baseline_loss,
            "loss_increase_percent": (masked_loss / baseline_loss - 1.0) * 100 if baseline_loss > 0 else 0
        })
    
    # Restore original sparsity
    if hasattr(model.quantum_extension, "set_base_sparsity"):
        model.quantum_extension.set_base_sparsity(original_sparsity)
    model.quantum_extension.apply_masks()
    
    return {
        "baseline_loss": baseline_loss,
        "tradeoff_data": tradeoff_data,
        "optimal_sparsity": max([d["target_sparsity"] for d in tradeoff_data if d["loss_increase_percent"] < 10]) if tradeoff_data else 0
    }


def evaluate_quantum_extension(
    model: torch.nn.Module,
    datasets: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of quantum extension.
    
    Args:
        model: The model to evaluate
        datasets: Dictionary containing evaluation datasets
            - "efficiency_test": Efficiency test inputs
            - "patterns": List of patterns to evaluate
            - "sparsity_levels": List of sparsity levels to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Check if extension is enabled
    if not hasattr(model, "quantum_extension") or model.quantum_extension is None:
        return {"error": "Quantum extension not enabled"}
    
    # Evaluate efficiency gain if test inputs provided
    if "efficiency_test" in datasets:
        test_inputs = datasets["efficiency_test"]
        results["efficiency_gain"] = quantum_efficiency_gain(
            model,
            test_inputs
        )
    
    # Evaluate pattern effectiveness if patterns provided
    if "patterns" in datasets:
        patterns = datasets["patterns"]
        results["pattern_effectiveness"] = pattern_effectiveness(
            model,
            patterns
        )
    
    # Evaluate sparsity-accuracy tradeoff if test data provided
    if "accuracy_test" in datasets and "sparsity_levels" in datasets:
        test_texts = datasets["accuracy_test"]
        sparsity_levels = datasets["sparsity_levels"]
        results["sparsity_accuracy_tradeoff"] = sparsity_accuracy_tradeoff(
            model,
            test_texts,
            sparsity_levels
        )
    
    # Extension-specific metrics
    results["mask_statistics"] = model.quantum_extension.get_mask_statistics()
    
    results["extension_info"] = {
        "type": model.quantum_extension.type if hasattr(model.quantum_extension, "type") else "quantum",
        "name": model.quantum_extension.name if hasattr(model.quantum_extension, "name") else "unknown"
    }
    
    return results
"""
Resonance stability metrics for QLLM models.

This module provides metrics to evaluate the stability and effectiveness of 
quantum-inspired resonance mechanisms, entropy-driven collapse, and prime-based
pattern recognition.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def entropy_collapse_efficiency(
    model: torch.nn.Module,
    inputs: List[str],
    expected_iterations: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluate efficiency of entropy-driven collapse mechanism.
    
    Args:
        model: The model to evaluate
        inputs: List of input texts with varying complexity
        expected_iterations: Optional list of expected iterations for each input
        
    Returns:
        Dictionary of entropy collapse metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "resonance_blocks") or not hasattr(model.resonance_blocks[0], "track_iterations"):
        return {"error": "Model doesn't support iteration tracking for resonance blocks"}
    
    # Enable iteration tracking
    for block in model.resonance_blocks:
        block.track_iterations = True
        block.reset_iteration_stats()
    
    results = {
        "input_complexity": [],
        "iterations_per_layer": [],
        "avg_iterations": [],
        "efficiency_scores": [],
        "correlation_complexity_iterations": 0.0
    }
    
    # Process each input
    for i, text in enumerate(inputs):
        # Tokenize
        tokenizer = model.tokenizer
        encoded = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        
        # Forward pass with iteration tracking
        with torch.no_grad():
            _ = model(**encoded)
        
        # Collect iteration statistics
        iterations_per_layer = []
        for j, block in enumerate(model.resonance_blocks):
            if hasattr(block, "iterations_taken"):
                iterations_per_layer.append(block.iterations_taken)
        
        avg_iterations = sum(iterations_per_layer) / len(iterations_per_layer) if iterations_per_layer else 0
        
        # Determine complexity (token length as proxy)
        complexity = encoded.input_ids.size(1)
        
        # Calculate efficiency score
        if expected_iterations and i < len(expected_iterations):
            expected = expected_iterations[i]
            efficiency = expected / avg_iterations if avg_iterations > 0 else 0.0
        else:
            # Without expected values, normalize by max iterations
            max_iters = max(getattr(block, "max_iterations", 10) for block in model.resonance_blocks)
            efficiency = 1.0 - (avg_iterations / max_iters)
        
        # Store results
        results["input_complexity"].append(complexity)
        results["iterations_per_layer"].append(iterations_per_layer)
        results["avg_iterations"].append(avg_iterations)
        results["efficiency_scores"].append(efficiency)
    
    # Calculate correlation between complexity and iterations
    if len(results["input_complexity"]) > 1:
        correlation = np.corrcoef(
            results["input_complexity"],
            results["avg_iterations"]
        )[0, 1]
        results["correlation_complexity_iterations"] = correlation
    
    # Calculate summary statistics
    results["mean_efficiency"] = sum(results["efficiency_scores"]) / len(results["efficiency_scores"]) if results["efficiency_scores"] else 0
    results["mean_iterations"] = sum(results["avg_iterations"]) / len(results["avg_iterations"]) if results["avg_iterations"] else 0
    
    # Clean up
    for block in model.resonance_blocks:
        block.track_iterations = False
    
    return results


def prime_resonance_metrics(
    model: torch.nn.Module,
    test_patterns: List[str]
) -> Dict[str, Any]:
    """
    Evaluate effectiveness of prime-based resonance patterns.
    
    Args:
        model: The model to evaluate
        test_patterns: List of test patterns to evaluate resonance
        
    Returns:
        Dictionary of prime resonance metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "prime_hilbert_encoder"):
        return {"error": "Model doesn't have a prime Hilbert encoder"}
    
    results = {}
    
    # Get the model's prime numbers
    primes = getattr(model.prime_hilbert_encoder, "primes", [])
    if not primes:
        return {"error": "No prime numbers found in the encoder"}
    
    results["primes_used"] = primes
    
    # Evaluate prime subspace activations
    activations_by_prime = {p: [] for p in primes}
    prime_correlation = np.zeros((len(primes), len(primes)))
    
    # Process test patterns
    for pattern in test_patterns:
        tokenizer = model.tokenizer
        encoded = tokenizer(pattern, return_tensors="pt").to(next(model.parameters()).device)
        
        with torch.no_grad():
            # Get prime subspace activations
            prime_activations = model.prime_hilbert_encoder(encoded.input_ids)
            
            # Assume output is a list of tensors, one per prime subspace
            if isinstance(prime_activations, list) and len(prime_activations) == len(primes):
                for i, p in enumerate(primes):
                    # Store average activation magnitude
                    act_mag = torch.abs(prime_activations[i]).mean().item()
                    activations_by_prime[p].append(act_mag)
                    
                # Calculate correlations between prime subspaces
                for i in range(len(primes)):
                    for j in range(len(primes)):
                        if i != j:
                            # Flatten activations
                            act_i = prime_activations[i].view(-1).cpu().numpy()
                            act_j = prime_activations[j].view(-1).cpu().numpy()
                            
                            # Calculate correlation
                            corr = np.corrcoef(act_i, act_j)[0, 1]
                            prime_correlation[i, j] += corr
    
    # Average correlations across all test patterns
    prime_correlation /= len(test_patterns)
    
    # Calculate metrics
    avg_activations = {p: sum(vals) / len(vals) if vals else 0 for p, vals in activations_by_prime.items()}
    
    results["avg_prime_activations"] = avg_activations
    results["prime_correlation_matrix"] = prime_correlation.tolist()
    
    # Calculate orthogonality score (average absolute correlation, lower is better)
    off_diag_indices = ~np.eye(len(primes), dtype=bool)
    avg_abs_corr = np.abs(prime_correlation[off_diag_indices]).mean()
    orthogonality = 1.0 - avg_abs_corr
    
    results["orthogonality_score"] = orthogonality
    
    # Calculate activation balance (variance across primes, lower is better)
    activation_values = np.array(list(avg_activations.values()))
    activation_balance = 1.0 - np.std(activation_values) / np.mean(activation_values) if np.mean(activation_values) > 0 else 0
    
    results["activation_balance"] = activation_balance
    
    # Combine into an overall resonance quality score
    results["resonance_quality"] = (orthogonality + activation_balance) / 2
    
    return results


def mask_evolution_stability(
    model: torch.nn.Module,
    tracking_steps: List[int],
    test_inputs: List[str]
) -> Dict[str, Any]:
    """
    Evaluate stability of mask evolution over time.
    
    Args:
        model: The model to evaluate
        tracking_steps: Steps at which to track mask evolution
        test_inputs: Test inputs for evaluating model performance
        
    Returns:
        Dictionary of mask evolution stability metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "quantum_extension") or model.quantum_extension is None:
        return {"error": "Quantum extension not enabled"}
    
    # Enable mask evolution tracking
    if not hasattr(model.quantum_extension, "enable_tracking"):
        return {"error": "Quantum extension doesn't support evolution tracking"}
    
    model.quantum_extension.enable_tracking(True)
    
    # Initialize results
    results = {
        "tracking_steps": tracking_steps,
        "mask_changes": [],
        "performance_metrics": [],
        "stability_score": 0.0
    }
    
    baseline_loss = measure_loss_on_inputs(model, test_inputs)
    
    # Track evolution over steps
    previous_masks = None
    
    for step in tracking_steps:
        # Evolve masks for specified steps
        for _ in range(step):
            model.quantum_extension.evolve_masks(model)
        
        # Get current masks
        current_masks = model.quantum_extension.get_current_masks()
        
        # Calculate mask change ratio
        if previous_masks:
            change_ratio = calculate_mask_change(previous_masks, current_masks)
        else:
            change_ratio = 0.0
        
        # Measure model performance
        current_loss = measure_loss_on_inputs(model, test_inputs)
        performance_change = (baseline_loss - current_loss) / baseline_loss if baseline_loss > 0 else 0.0
        
        # Store results
        results["mask_changes"].append(change_ratio)
        results["performance_metrics"].append({
            "loss": current_loss,
            "performance_change": performance_change
        })
        
        # Update previous masks
        previous_masks = current_masks
    
    # Calculate stability metrics
    if len(results["mask_changes"]) > 1:
        # Convergence: decreasing rate of change indicates stability
        changes = np.array(results["mask_changes"])
        change_diffs = np.diff(changes)
        converging = np.all(change_diffs < 0)
        
        # Performance stability
        perf_changes = [m["performance_change"] for m in results["performance_metrics"]]
        perf_stability = 1.0 - np.std(perf_changes) if perf_changes else 1.0
        
        # Final mask change rate
        final_change_rate = results["mask_changes"][-1] if results["mask_changes"] else 0.0
        
        # Overall stability score
        results["stability_score"] = 0.5 * perf_stability + 0.3 * converging + 0.2 * (1.0 - final_change_rate)
    
    # Disable tracking
    model.quantum_extension.enable_tracking(False)
    
    return results


def resonance_stability_evaluation(
    model: torch.nn.Module,
    evaluation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of resonance stability metrics.
    
    Args:
        model: The model to evaluate
        evaluation_data: Dictionary containing evaluation data
            - "collapse_test_inputs": List of inputs for entropy collapse testing
            - "expected_iterations": Expected iterations for collapse
            - "prime_test_patterns": Test patterns for prime resonance
            - "mask_evolution": Dict with tracking_steps and test_inputs
        
    Returns:
        Dictionary of resonance stability metrics
    """
    results = {}
    
    # Evaluate entropy collapse efficiency
    if "collapse_test_inputs" in evaluation_data:
        expected_iterations = evaluation_data.get("expected_iterations")
        results["entropy_collapse"] = entropy_collapse_efficiency(
            model,
            evaluation_data["collapse_test_inputs"],
            expected_iterations
        )
    
    # Evaluate prime resonance metrics
    if "prime_test_patterns" in evaluation_data:
        results["prime_resonance"] = prime_resonance_metrics(
            model,
            evaluation_data["prime_test_patterns"]
        )
    
    # Evaluate mask evolution stability
    if "mask_evolution" in evaluation_data:
        mask_eval = evaluation_data["mask_evolution"]
        tracking_steps = mask_eval.get("tracking_steps", [10, 50, 100])
        test_inputs = mask_eval.get("test_inputs", [])
        
        results["mask_stability"] = mask_evolution_stability(
            model,
            tracking_steps,
            test_inputs
        )
    
    # Calculate overall resonance stability score
    stability_scores = []
    
    if "entropy_collapse" in results and "mean_efficiency" in results["entropy_collapse"]:
        stability_scores.append(results["entropy_collapse"]["mean_efficiency"])
    
    if "prime_resonance" in results and "resonance_quality" in results["prime_resonance"]:
        stability_scores.append(results["prime_resonance"]["resonance_quality"])
    
    if "mask_stability" in results and "stability_score" in results["mask_stability"]:
        stability_scores.append(results["mask_stability"]["stability_score"])
    
    if stability_scores:
        results["overall_stability_score"] = sum(stability_scores) / len(stability_scores)
    else:
        results["overall_stability_score"] = 0.0
    
    return results


# Helper functions

def measure_loss_on_inputs(model: torch.nn.Module, inputs: List[str]) -> float:
    """Measure average loss on a set of inputs"""
    losses = []
    tokenizer = model.tokenizer
    
    for text in inputs:
        encoded = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model(input_ids=encoded.input_ids)
            
            # Get logits
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encoded.input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            losses.append(loss.item())
    
    return sum(losses) / len(losses) if losses else float('inf')


def calculate_mask_change(
    previous_masks: Dict[str, torch.Tensor],
    current_masks: Dict[str, torch.Tensor]
) -> float:
    """Calculate the ratio of mask changes between two sets of masks"""
    if not previous_masks or not current_masks:
        return 0.0
    
    total_elements = 0
    total_changes = 0
    
    for name, mask in current_masks.items():
        if name in previous_masks:
            prev_mask = previous_masks[name]
            
            # Calculate number of different elements
            if prev_mask.shape == mask.shape:
                diff = (prev_mask != mask).sum().item()
                total_changes += diff
                total_elements += mask.numel()
    
    # Return change ratio
    return total_changes / total_elements if total_elements > 0 else 0.0
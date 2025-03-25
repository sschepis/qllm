"""
Evaluation metrics for language models.

This module provides functions for computing common language model
evaluation metrics such as perplexity and accuracy.
"""

import math
import torch
import numpy as np
from typing import Dict, List, Union, Tuple


def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity is defined as exp(entropy), where entropy is the 
    average negative log-likelihood per token.
    
    Args:
        loss (float): Cross-entropy loss value (negative log-likelihood)
        
    Returns:
        float: Perplexity value
    """
    return math.exp(loss)


def compute_accuracy(logits, labels, ignore_index=-100):
    """
    Compute token prediction accuracy.
    
    Args:
        logits (torch.Tensor): Prediction logits of shape [batch_size, seq_len, vocab_size]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, seq_len]
        ignore_index (int): Index to ignore in accuracy calculation (padding tokens)
        
    Returns:
        float: Accuracy as a value between 0 and 1
    """
    # Convert logits to predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens (excluding padding and ignored tokens)
    mask = (labels != ignore_index)
    
    # Count correct predictions
    correct = (predictions == labels) & mask
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    accuracy = correct.sum().item() / total
    return accuracy


def compute_metrics(logits, labels, loss=None, ignore_index=-100):
    """
    Compute multiple evaluation metrics.
    
    Args:
        logits (torch.Tensor): Prediction logits of shape [batch_size, seq_len, vocab_size]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, seq_len]
        loss (float, optional): Pre-computed loss value
        ignore_index (int): Index to ignore in metrics calculation
        
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Compute accuracy
    accuracy = compute_accuracy(logits, labels, ignore_index)
    metrics["accuracy"] = accuracy
    
    # Compute perplexity if loss is provided
    if loss is not None:
        perplexity = compute_perplexity(loss)
        metrics["perplexity"] = perplexity
    
    return metrics


def compute_entropy(probs):
    """
    Compute Shannon entropy of a probability distribution.
    
    Args:
        probs (torch.Tensor): Probability distribution tensor
        
    Returns:
        torch.Tensor: Entropy value(s)
    """
    # Add small epsilon to avoid log(0)
    probs = probs + 1e-10
    
    # Compute entropy: -∑ p_i * log(p_i)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    
    return entropy


def compute_entropy_stats(model_outputs, attention_metadata_key="metadata"):
    """
    Compute statistics on the entropy values from model outputs.
    
    Args:
        model_outputs (Dict): Model outputs containing attention metadata
        attention_metadata_key (str): Key for accessing metadata in outputs
        
    Returns:
        Dict: Dictionary containing entropy statistics
    """
    entropy_stats = {}
    
    if attention_metadata_key not in model_outputs:
        return entropy_stats
    
    # Extract all entropy values from block metadata
    entropy_values = []
    iteration_counts = []
    convergence_gaps = []
    entropy_thresholds = []
    
    for block_metadata in model_outputs[attention_metadata_key]:
        if "metadata" in block_metadata:
            metadata = block_metadata["metadata"]
            block_type = block_metadata.get("type", "unknown")
            layer = block_metadata.get("layer", "unknown")
            
            # Extract entropy information
            if "entropy" in metadata:
                entropy = metadata["entropy"]
                if isinstance(entropy, torch.Tensor):
                    key = f"{block_type}_{layer}_entropy"
                    entropy_stats[key] = entropy.mean().item()
                    entropy_values.append(entropy.mean().item())
            
            # Extract iteration information
            if "iterations" in metadata:
                iterations = metadata["iterations"]
                if isinstance(iterations, torch.Tensor):
                    key = f"{block_type}_{layer}_iterations"
                    entropy_stats[key] = iterations.float().mean().item()
                    iteration_counts.append(iterations.float().mean().item())
            
            # Extract convergence gap information
            if "convergence_gap" in metadata:
                gap = metadata["convergence_gap"]
                if isinstance(gap, torch.Tensor):
                    key = f"{block_type}_{layer}_convergence_gap"
                    entropy_stats[key] = gap.mean().item()
                    convergence_gaps.append(gap.mean().item())
            
            # Extract entropy threshold
            if "entropy_threshold" in metadata:
                threshold = metadata["entropy_threshold"]
                if isinstance(threshold, (float, int)):
                    key = f"{block_type}_{layer}_threshold"
                    entropy_stats[key] = threshold
                    entropy_thresholds.append(threshold)
                    
            # Extract entropy history information for debugging
            if "entropy_history" in metadata:
                history = metadata["entropy_history"]
                if history and len(history) > 1:
                    first_ent = history[0]["mean_entropy"].mean().item() if len(history) > 0 else 0
                    last_ent = history[-1]["mean_entropy"].mean().item() if len(history) > 0 else 0
                    reduction = first_ent - last_ent
                    
                    key = f"{block_type}_{layer}_entropy_reduction"
                    entropy_stats[key] = reduction
    
    # Compute overall statistics
    if entropy_values:
        entropy_stats["mean_entropy"] = sum(entropy_values) / len(entropy_values)
        entropy_stats["min_entropy"] = min(entropy_values)
        entropy_stats["max_entropy"] = max(entropy_values)
    
    if iteration_counts:
        entropy_stats["mean_iterations"] = sum(iteration_counts) / len(iteration_counts)
        entropy_stats["min_iterations"] = min(iteration_counts)
        entropy_stats["max_iterations"] = max(iteration_counts)
    
    if convergence_gaps:
        entropy_stats["mean_convergence_gap"] = sum(convergence_gaps) / len(convergence_gaps)
        entropy_stats["min_convergence_gap"] = min(convergence_gaps)
        entropy_stats["max_convergence_gap"] = max(convergence_gaps)
    
    if entropy_thresholds:
        entropy_stats["entropy_threshold"] = entropy_thresholds[0]  # Assume same threshold for all layers
    
    return entropy_stats


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate a model using standard metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_metrics = {}
    
    # Initialize lists for entropy and iteration statistics
    entropy_stats = {
        "mean_entropy": [],
        "mean_iterations": []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch, return_dict=True)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Count tokens
            mask = batch["attention_mask"]
            num_tokens = mask.sum().item()
            
            # Update statistics
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Compute batch metrics
            batch_metrics = compute_metrics(logits, batch["labels"], loss=loss.item())
            
            # Update all metrics
            for key, value in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
            
            # Compute entropy statistics
            batch_entropy_stats = compute_entropy_stats(outputs)
            for key, value in batch_entropy_stats.items():
                if key in entropy_stats:
                    entropy_stats[key].append(value)
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = compute_perplexity(avg_loss)
    
    # Calculate average metrics
    final_metrics = {
        "loss": avg_loss,
        "perplexity": perplexity
    }
    
    for key, values in all_metrics.items():
        final_metrics[key] = sum(values) / len(values)
    
    # Add entropy statistics
    for key, values in entropy_stats.items():
        if values:
            final_metrics[key] = sum(values) / len(values)
    
    return final_metrics
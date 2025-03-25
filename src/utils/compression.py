"""
Compression utilities for Semantic Resonance Language Model.

This module provides functions for model compression through structured
prime resonance masking and pruning techniques as described in the paper.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Tuple


def create_prime_resonance_mask(dim, primes=[7, 11, 13, 17, 19], mask_type="mod"):
    """
    Create a structured mask based on prime resonance conditions.
    
    Args:
        dim (int): Dimension of the weight matrix
        primes (List[int]): List of prime numbers for resonance conditions
        mask_type (str): Type of mask to create:
            - "mod": (i-j) mod p = 0 for some prime p
            - "coprime": gcd(i,j) = 1 (i and j are coprime)
            - "primeprod": i*j is divisible by some prime p
            
    Returns:
        torch.Tensor: Boolean mask of shape [dim, dim]
    """
    mask = torch.zeros((dim, dim), dtype=torch.bool)
    
    if mask_type == "mod":
        # Set indices to 1 if they pass the prime resonance condition (i-j) mod p = 0
        for i in range(dim):
            for j in range(dim):
                if any((i-j) % p == 0 for p in primes):
                    mask[i, j] = True
    
    elif mask_type == "coprime":
        # Set indices to 1 if i and j are coprime
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        for i in range(1, dim):
            for j in range(1, dim):
                if gcd(i, j) == 1:
                    mask[i, j] = True
    
    elif mask_type == "primeprod":
        # Set indices to 1 if i*j is divisible by some prime p
        for i in range(dim):
            for j in range(dim):
                if any((i+1)*(j+1) % p == 0 for p in primes):
                    mask[i, j] = True
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    
    return mask


def prime_importance_pruning(weight_matrix, primes=[7, 11, 13, 17, 19], threshold=0.8):
    """
    Apply prime-based importance pruning to a weight matrix.
    
    This implements the prime resonance selection described in the paper:
        W'_{i,j} = W_{i,j} if i,j ∈ P and I_{i,j} > τ, else 0
    
    Args:
        weight_matrix (torch.Tensor): Weight matrix to prune
        primes (List[int]): List of prime numbers for importance pruning
        threshold (float): Importance threshold (quantile)
        
    Returns:
        torch.Tensor: Pruned weight matrix
    """
    dim1, dim2 = weight_matrix.shape
    pruned_matrix = torch.zeros_like(weight_matrix)
    
    # Create prime indices
    prime_indices_1 = []
    prime_indices_2 = []
    
    # Collect prime indices for each dimension
    # Note: We include indices that are divisible by primes or are primes themselves
    for i in range(dim1):
        if any(i % p == 0 for p in primes) or any(i == p for p in primes):
            prime_indices_1.append(i)
    
    for j in range(dim2):
        if any(j % p == 0 for p in primes) or any(j == p for p in primes):
            prime_indices_2.append(j)
    
    # Calculate importance threshold
    importance = torch.abs(weight_matrix)
    threshold_value = torch.quantile(importance.view(-1), threshold)
    
    # Keep only important weights at prime indices
    for i in prime_indices_1:
        for j in prime_indices_2:
            if importance[i, j] > threshold_value:
                pruned_matrix[i, j] = weight_matrix[i, j]
    
    return pruned_matrix


def compress_model(model, compression_config):
    """
    Compress a model using prime resonance selection.
    
    Args:
        model (torch.nn.Module): Model to compress
        compression_config: Configuration for compression
            - method (str): Compression method ("mask", "prune", or "both")
            - primes (List[int]): List of prime numbers
            - threshold (float): Importance threshold for pruning
            - mask_type (str): Type of mask to create
            
    Returns:
        torch.nn.Module: Compressed model
        float: Compression ratio (original_params / compressed_params)
    """
    # Make a copy of the model to avoid modifying the original
    compressed_model = type(model)(model.config)
    compressed_model.load_state_dict(model.state_dict())
    
    # Get compression parameters
    method = compression_config.get("method", "both")
    primes = compression_config.get("primes", [7, 11, 13, 17, 19])
    threshold = compression_config.get("threshold", 0.8)
    mask_type = compression_config.get("mask_type", "mod")
    
    # Count parameters before compression
    original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    original_nonzero = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
    
    # Apply compression
    with torch.no_grad():
        for name, param in compressed_model.named_parameters():
            # Skip parameters that are not weight matrices
            if len(param.shape) != 2:
                continue
            
            # Skip embeddings and layer norms
            if "embedding" in name.lower() or "norm" in name.lower():
                continue
            
            # Apply compression based on method
            if method == "mask" or method == "both":
                # Create mask
                mask = create_prime_resonance_mask(
                    min(param.shape),
                    primes=primes,
                    mask_type=mask_type
                )
                
                # Expand mask to parameter shape if needed
                if mask.shape != param.shape:
                    expanded_mask = torch.zeros_like(param, dtype=torch.bool)
                    min_dim = min(param.shape)
                    expanded_mask[:min_dim, :min_dim] = mask
                    mask = expanded_mask
                
                # Apply mask
                param.data = param.data * mask.to(param.device)
            
            if method == "prune" or method == "both":
                # Apply prime importance pruning
                param.data = prime_importance_pruning(
                    param.data,
                    primes=primes,
                    threshold=threshold
                )
    
    # Count parameters after compression
    compressed_nonzero = sum((p != 0).sum().item() for p in compressed_model.parameters() if p.requires_grad)
    
    # Calculate compression ratio
    compression_ratio = original_nonzero / max(1, compressed_nonzero)
    
    return compressed_model, compression_ratio


def load_compressed_model(model_path, config, device=None):
    """
    Load a compressed model.
    
    Args:
        model_path (str): Path to the compressed model
        config: Model configuration
        device (torch.device, optional): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded compressed model
    """
    from src.model.semantic_resonance_model import SemanticResonanceModel
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the compressed model
    compressed_model = SemanticResonanceModel(config)
    compressed_model.load_state_dict(torch.load(model_path, map_location=device))
    compressed_model.to(device)
    
    # Count non-zero parameters
    nonzero_params = sum((p != 0).sum().item() for p in compressed_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
    
    print(f"Loaded compressed model from {model_path}")
    print(f"Non-zero parameters: {nonzero_params:,} out of {total_params:,} "
          f"({nonzero_params/total_params:.2%})")
    
    return compressed_model


def compare_models(original_model, compressed_model):
    """
    Compare original and compressed models.
    
    Args:
        original_model (torch.nn.Module): Original model
        compressed_model (torch.nn.Module): Compressed model
        
    Returns:
        Dict: Comparison statistics
    """
    stats = {}
    
    # Count parameters
    original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    compressed_nonzero = sum((p != 0).sum().item() for p in compressed_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
    
    # Calculate compression statistics
    stats["original_params"] = original_params
    stats["compressed_params"] = compressed_nonzero
    stats["total_params"] = total_params
    stats["compression_ratio"] = original_params / max(1, compressed_nonzero)
    stats["sparsity"] = 1.0 - (compressed_nonzero / total_params)
    
    return stats
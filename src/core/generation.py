"""
Text generation utilities for QLLM.

This module provides functions for text generation that were previously
duplicated across different model implementations. It includes utilities
for:
- Temperature-based sampling
- Top-k filtering
- Top-p (nucleus) filtering
- Complete text generation pipeline
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable


def apply_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature to logits for sampling.
    
    Args:
        logits: Raw logits from model output
        temperature: Temperature value (higher = more random, lower = more deterministic)
        
    Returns:
        Temperature-adjusted logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Apply temperature scaling
    return logits / temperature


def apply_top_k_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    filter_value: float = -float('inf')
) -> torch.Tensor:
    """
    Apply top-k filtering to logits.
    
    Args:
        logits: Raw or temperature-adjusted logits
        top_k: Number of top tokens to keep (0 = no filtering)
        filter_value: Value to assign to filtered logits
        
    Returns:
        Filtered logits
    """
    if top_k <= 0:
        return logits
    
    # Get top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    
    # Create a mask of values to keep
    filter_mask = torch.zeros_like(logits)
    filter_mask.scatter_(-1, top_k_indices, 1.0)
    
    # Apply the mask
    return torch.where(filter_mask > 0, logits, torch.full_like(logits, filter_value))


def apply_top_p_filtering(
    logits: torch.Tensor,
    top_p: float = 1.0,
    filter_value: float = -float('inf'),
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.
    
    Args:
        logits: Raw or temperature-adjusted logits
        top_p: Probability threshold (0.0-1.0)
        filter_value: Value to assign to filtered logits
        min_tokens_to_keep: Minimum number of tokens to keep regardless of probability
        
    Returns:
        Filtered logits
    """
    if top_p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    
    # Shift the indices to the right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Create the mask for the original logits order
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    
    # Apply the mask
    return torch.where(indices_to_remove, torch.full_like(logits, filter_value), logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repetition_penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to avoid repeating tokens.
    
    Args:
        logits: Raw logits from model output (batch_size, vocab_size)
        input_ids: Input token IDs (batch_size, seq_length)
        repetition_penalty: Penalty factor (1.0 = no penalty)
        
    Returns:
        Logits with repetition penalty applied
    """
    if repetition_penalty == 1.0:
        return logits
    
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        # Get the unique token IDs in the sequence to penalize
        for token_id in set(input_ids[i].tolist()):
            # Penalize by dividing the logit by the penalty factor
            logits[i, token_id] /= repetition_penalty
    
    return logits


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool = True,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Sample next token from logits.
    
    Args:
        logits: Processed logits
        do_sample: Whether to sample or take argmax
        num_samples: Number of samples to generate
        
    Returns:
        Sampled token IDs
    """
    if do_sample:
        # Convert logits to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=num_samples)
    else:
        # Take the most likely token
        next_tokens = torch.argmax(logits, dim=-1, keepdim=(num_samples > 1))
    
    return next_tokens


def generate_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 50,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_return_sequences: int = 1,
    tokenizer = None,
    **model_kwargs
) -> torch.Tensor:
    """
    Generate text using the provided model.
    
    This function consolidates text generation logic found in multiple model implementations
    into a single, reusable function.
    
    Args:
        model: The model to use for generation
        input_ids: Initial token IDs
        attention_mask: Attention mask
        max_length: Maximum length of generated text
        temperature: Temperature for sampling
        do_sample: Whether to sample or take argmax
        top_k: Top-k filtering parameter
        top_p: Top-p filtering parameter
        repetition_penalty: Penalty for repeating tokens
        pad_token_id: ID of the padding token
        eos_token_id: ID of the end-of-sequence token
        batch_size: Batch size for generation
        num_return_sequences: Number of sequences to return
        tokenizer: Optional tokenizer for string inputs
        **model_kwargs: Additional keyword arguments for the model
        
    Returns:
        Generated token IDs
    """
    # Handle string input if tokenizer is provided
    if isinstance(input_ids, str) and tokenizer is not None:
        input_text = input_ids
        encoded = tokenizer(input_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    
    # Get device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Setup batch size and attention mask
    if batch_size is None:
        batch_size = input_ids.shape[0]
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    
    # Clone input_ids to avoid modifying the original
    generated = input_ids.clone()
    
    # Set eos_token_id to pad_token_id if not provided
    if eos_token_id is None and pad_token_id is not None:
        eos_token_id = pad_token_id
    
    # Track which sequences have completed
    if eos_token_id is not None:
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Generate until max_length is reached or all sequences complete
    with torch.no_grad():
        for _ in range(max_length - generated.shape[1]):
            # Forward pass
            model_inputs = {
                "input_ids": generated,
                "attention_mask": attention_mask,
                **model_kwargs
            }
            
            # Get model output
            if hasattr(model, "forward"):
                outputs = model(**model_inputs)
            else:
                # Handle different model interfaces
                outputs = model(generated, attention_mask=attention_mask, **model_kwargs)
            
            # Extract logits from model output
            if isinstance(outputs, dict):
                next_token_logits = outputs.get("logits", outputs.get("scores"))
                if next_token_logits is None:
                    raise ValueError("Model output doesn't contain logits or scores")
                # Take last token logits
                next_token_logits = next_token_logits[:, -1, :]
            elif isinstance(outputs, tuple):
                # First element is often logits in tuple outputs
                next_token_logits = outputs[0][:, -1, :]
            else:
                # Assume direct logits output
                next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = apply_temperature(next_token_logits, temperature)
            
            # Apply repetition penalty
            next_token_logits = apply_repetition_penalty(
                next_token_logits, generated, repetition_penalty)
            
            # Apply top-k filtering
            next_token_logits = apply_top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p filtering
            next_token_logits = apply_top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            next_tokens = sample_next_token(
                next_token_logits, do_sample, num_samples=1)
            
            # Handle EOS token if provided
            if eos_token_id is not None:
                # Set tokens to eos_token_id where sequences should end
                next_tokens = next_tokens.where(
                    unfinished_sequences.unsqueeze(-1).bool(),
                    torch.full_like(next_tokens, eos_token_id)
                )
                
                # Update unfinished_sequences
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long().squeeze()
                )
                
                # Stop if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
            
            # Add next token to generated
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)
    
    # Return generated ids (or text if tokenizer provided)
    return generated
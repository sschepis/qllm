"""
General evaluation metrics for QLLM models.

This module provides fundamental metrics for evaluating model performance,
efficiency, and output quality.
"""

import time
import torch
import numpy as np
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union

import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def perplexity(
    model: torch.nn.Module,
    text: str,
    stride: int = 512,
    max_length: int = 1024
) -> float:
    """
    Calculate perplexity for a given text.
    
    Args:
        model: The model to evaluate
        text: Input text
        stride: Stride length for processing long texts
        max_length: Maximum sequence length
        
    Returns:
        Perplexity value
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Tokenize input text
    tokenizer = model.tokenizer
    encodings = tokenizer(text, return_tensors="pt")
    
    # Initialize variables for perplexity calculation
    nlls = []
    device = next(model.parameters()).device
    
    # Process text in chunks for long sequences
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
            # Shift logits and targets for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Store negative log likelihood
            nll = loss.view(shift_labels.size(0), -1).sum(1)
            nlls.append(nll)
    
    # Combine results
    if nlls:
        return torch.exp(torch.cat(nlls, dim=0).sum() / end_loc).item()
    else:
        return float('inf')


def parameter_efficiency(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate parameter efficiency metrics.
    
    Args:
        model: The model to evaluate
        
    Returns:
        Dictionary of parameter efficiency metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate sparsity if quantum extension is enabled
    sparsity = 0.0
    if hasattr(model, "quantum_extension") and model.quantum_extension is not None:
        mask_stats = model.quantum_extension.get_mask_statistics()
        sparsity = mask_stats.get("overall_sparsity", 0.0)
    
    # Calculate effective parameters (accounting for sparsity)
    effective_params = total_params * (1.0 - sparsity)
    
    # Compression ratio
    compression_ratio = total_params / effective_params if effective_params > 0 else 1.0
    
    # Parameter breakdown by extension
    extension_params = {}
    
    if hasattr(model, "multimodal_extension") and model.multimodal_extension is not None:
        mm_params = sum(p.numel() for p in model.multimodal_extension.parameters())
        extension_params["multimodal"] = mm_params
    
    if hasattr(model, "memory_extension") and model.memory_extension is not None:
        mem_params = sum(p.numel() for p in model.memory_extension.parameters())
        extension_params["memory"] = mem_params
    
    if hasattr(model, "quantum_extension") and model.quantum_extension is not None:
        q_params = sum(p.numel() for p in model.quantum_extension.parameters())
        extension_params["quantum"] = q_params
    
    # Base model parameters
    base_params = total_params - sum(extension_params.values())
    extension_params["base_model"] = base_params
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "sparsity": sparsity,
        "effective_params": effective_params,
        "compression_ratio": compression_ratio,
        "extension_params": extension_params
    }


def memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate memory usage metrics.
    
    Args:
        model: The model to evaluate
        
    Returns:
        Dictionary of memory usage metrics
    """
    # Get system memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Calculate model size in memory
    model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Get peak memory usage for CUDA if available
    peak_gpu_mb = 0
    if torch.cuda.is_available():
        peak_gpu_bytes = torch.cuda.max_memory_allocated()
        peak_gpu_mb = peak_gpu_bytes / (1024 * 1024)
    
    # Memory usage by extension
    extension_memory = {}
    
    if hasattr(model, "multimodal_extension") and model.multimodal_extension is not None:
        mm_bytes = sum(p.nelement() * p.element_size() for p in model.multimodal_extension.parameters())
        extension_memory["multimodal"] = mm_bytes / (1024 * 1024)
    
    if hasattr(model, "memory_extension") and model.memory_extension is not None:
        mem_bytes = sum(p.nelement() * p.element_size() for p in model.memory_extension.parameters())
        extension_memory["memory"] = mem_bytes / (1024 * 1024)
    
    if hasattr(model, "quantum_extension") and model.quantum_extension is not None:
        q_bytes = sum(p.nelement() * p.element_size() for p in model.quantum_extension.parameters())
        extension_memory["quantum"] = q_bytes / (1024 * 1024)
    
    return {
        "process_rss_mb": memory_info.rss / (1024 * 1024),
        "process_vms_mb": memory_info.vms / (1024 * 1024),
        "model_size_mb": model_size_mb,
        "peak_gpu_mb": peak_gpu_mb,
        "extension_memory_mb": extension_memory
    }


def inference_speed(
    model: torch.nn.Module,
    inputs: List[str],
    num_repeats: int = 3
) -> Dict[str, float]:
    """
    Measure inference speed.
    
    Args:
        model: The model to evaluate
        inputs: List of input strings
        num_repeats: Number of repeated measurements
        
    Returns:
        Dictionary of speed metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare inputs
    tokenizer = model.tokenizer
    encoded_inputs = [tokenizer(text, return_tensors="pt").to(device) for text in inputs]
    
    # Warm-up run
    with torch.no_grad():
        for encoded in encoded_inputs:
            _ = model(**encoded)
    
    # Measure forward pass time
    forward_times = []
    
    for _ in range(num_repeats):
        for encoded in encoded_inputs:
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                _ = model(**encoded)
            forward_times.append(time.time() - start_time)
    
    # Measure generation time
    generation_times = []
    
    for _ in range(num_repeats):
        for text in inputs:
            # Text generation
            start_time = time.time()
            _ = model.generate(text, max_length=50)
            generation_times.append(time.time() - start_time)
    
    # Calculate metrics
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_generation_time = sum(generation_times) / len(generation_times)
    
    # Calculate tokens per second for forward pass
    total_tokens = sum(encoded.input_ids.numel() for encoded in encoded_inputs)
    tokens_per_second = total_tokens / sum(forward_times)
    
    return {
        "avg_forward_time": avg_forward_time,
        "avg_generation_time": avg_generation_time,
        "tokens_per_second": tokens_per_second,
        "forward_times_std": np.std(forward_times),
        "generation_times_std": np.std(generation_times)
    }


def generation_diversity(
    model: torch.nn.Module,
    prompts: List[str],
    num_samples: int = 3,
    max_length: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """
    Measure diversity of generated text.
    
    Args:
        model: The model to evaluate
        prompts: List of prompts to generate from
        num_samples: Number of samples to generate per prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        
    Returns:
        Dictionary of diversity metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Generate multiple samples from each prompt
    all_generations = []
    bleu_scores = []
    
    for prompt in prompts:
        prompt_generations = []
        
        for _ in range(num_samples):
            generation = model.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            prompt_generations.append(generation)
        
        all_generations.append(prompt_generations)
        
        # Calculate pairwise BLEU scores within this prompt's generations
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                gen1_tokens = word_tokenize(prompt_generations[i])
                gen2_tokens = word_tokenize(prompt_generations[j])
                
                # Calculate BLEU in both directions and average
                bleu1 = sentence_bleu([gen1_tokens], gen2_tokens)
                bleu2 = sentence_bleu([gen2_tokens], gen1_tokens)
                avg_bleu = (bleu1 + bleu2) / 2
                
                bleu_scores.append(avg_bleu)
    
    # Calculate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # Higher BLEU means more similarity, so diversity is the inverse
    diversity_score = 1.0 - avg_bleu
    
    # Calculate unique n-grams
    all_text = " ".join([gen for prompt_gens in all_generations for gen in prompt_gens])
    tokens = word_tokenize(all_text)
    
    unigrams = set(tokens)
    bigrams = set(zip(tokens[:-1], tokens[1:]))
    trigrams = set(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    
    return {
        "diversity_score": diversity_score,
        "avg_bleu": avg_bleu,
        "unique_unigrams": len(unigrams),
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "unique_ratio": len(unigrams) / len(tokens) if tokens else 0
    }
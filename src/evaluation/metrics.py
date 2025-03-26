"""
Evaluation Metrics for QLLM Extensions.

This module provides metrics for evaluating the performance, efficiency,
and quality of QLLM models and extensions.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

import psutil
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


def multimodal_accuracy(
    model: torch.nn.Module,
    image_text_pairs: List[Dict[str, Any]],
    options: List[str],
    reference_answers: List[str]
) -> Dict[str, float]:
    """
    Measure multimodal accuracy on a visual question answering task.
    
    Args:
        model: The model to evaluate
        image_text_pairs: List of image and question pairs
        options: List of possible answers for each question
        reference_answers: List of correct answers
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "multimodal_extension") or model.multimodal_extension is None:
        return {"error": "Multimodal extension not enabled"}
    
    correct_count = 0
    
    for i, pair in enumerate(image_text_pairs):
        image = pair["image"]
        question = pair["text"]
        
        # Process image
        vision_features = model.multimodal_extension.process_images([image])[0]
        
        # Compute likelihood for each option
        option_scores = []
        
        for option in options[i]:
            input_text = f"{question} {option}"
            
            # Get model output
            with torch.no_grad():
                output = model(
                    input_text,
                    vision_features=vision_features
                )
            
            # Compute probability of the option given the context
            if isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            else:
                logits = output
            
            # Simplistic scoring based on final token prediction
            final_logits = logits[0, -1, :]
            option_score = F.softmax(final_logits, dim=0).max().item()
            option_scores.append(option_score)
        
        # Select highest scoring option
        predicted_idx = option_scores.index(max(option_scores))
        predicted_answer = options[i][predicted_idx]
        
        # Check if correct
        if predicted_answer == reference_answers[i]:
            correct_count += 1
    
    # Calculate accuracy
    accuracy = correct_count / len(image_text_pairs) if image_text_pairs else 0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(image_text_pairs)
    }


def knowledge_graph_retrieval(
    model: torch.nn.Module,
    queries: List[Dict[str, Any]],
    expected_results: List[Any]
) -> Dict[str, float]:
    """
    Measure accuracy of knowledge graph retrieval.
    
    Args:
        model: The model to evaluate
        queries: List of knowledge graph queries
        expected_results: List of expected results
        
    Returns:
        Dictionary of retrieval metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    if not hasattr(model, "memory_extension") or model.memory_extension is None:
        return {"error": "Memory extension not enabled"}
    
    correct_count = 0
    partial_matches = 0
    
    for i, query in enumerate(queries):
        query_type = query.get("type", "entity")
        query_params = query.get("params", {})
        
        # Execute query
        if query_type == "entity":
            result = model.memory_extension.retrieve_entity(**query_params)
        elif query_type == "relation":
            result = model.memory_extension.retrieve_relations(**query_params)
        else:
            continue
        
        # Compare with expected result
        expected = expected_results[i]
        
        # Exact match
        if result == expected:
            correct_count += 1
        # Partial match for lists of entities/relations
        elif isinstance(result, list) and isinstance(expected, list):
            # Calculate overlap
            if expected:
                overlap = len(set(result) & set(expected)) / len(expected)
                if overlap >= 0.5:  # At least 50% match
                    partial_matches += 1
    
    # Calculate metrics
    total_queries = len(queries)
    exact_accuracy = correct_count / total_queries if total_queries else 0
    partial_accuracy = (correct_count + partial_matches) / total_queries if total_queries else 0
    
    return {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "correct_count": correct_count,
        "partial_matches": partial_matches,
        "total_queries": total_queries
    }


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
    
    # Get parameter counts
    baseline_params = sum(p.numel() for p in model.parameters())
    
    # Measure metrics with quantum extension enabled
    model.quantum_extension.apply_masks()
    
    quantum_times = []
    for text in test_inputs:
        start_time = time.time()
        with torch.no_grad():
            _ = model(text)
        quantum_times.append(time.time() - start_time)
    
    quantum_avg = sum(quantum_times) / len(quantum_times)
    
    # Get mask statistics
    mask_stats = model.quantum_extension.get_mask_statistics()
    sparsity = mask_stats.get("overall_sparsity", 0.0)
    
    # Calculate effective parameters
    effective_params = baseline_params * (1.0 - sparsity)
    
    # Calculate speedup
    speedup = baseline_avg / quantum_avg if quantum_avg > 0 else 1.0
    
    # Calculate efficiency gain (params reduction * speedup)
    efficiency_gain = (baseline_params / effective_params) * speedup if effective_params > 0 else 1.0
    
    # Restore original state
    if original_state:
        model.quantum_extension.apply_masks()
    else:
        model.quantum_extension.disable_masks()
    
    return {
        "baseline_time": baseline_avg,
        "quantum_time": quantum_avg,
        "speedup": speedup,
        "sparsity": sparsity,
        "baseline_params": baseline_params,
        "effective_params": effective_params,
        "parameter_reduction": baseline_params / effective_params if effective_params > 0 else 1.0,
        "efficiency_gain": efficiency_gain
    }
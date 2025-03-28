"""
Text generation utilities for QLLM.

This module provides utilities for text generation, including sampling methods,
beam search, and other generation strategies that can be used across different
model implementations.
"""

import math
import torch
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

logger = logging.getLogger("qllm.utils.generation")


def prepare_inputs_for_generation(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Prepare inputs for text generation.
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Dictionary of prepared inputs
    """
    inputs = {"input_ids": input_ids}
    
    # Add attention mask if provided
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask
    
    # Add other model inputs
    for k, v in model_kwargs.items():
        inputs[k] = v
    
    return inputs


def get_logits_processor(
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    bad_words_ids: Optional[List[List[int]]] = None,
    min_length: int = 0,
    eos_token_id: Optional[int] = None,
    **kwargs
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get a logits processor function that applies various constraints.
    
    Args:
        repetition_penalty: Penalty for repeated tokens
        no_repeat_ngram_size: Size of n-grams that shouldn't be repeated
        bad_words_ids: List of token sequences that shouldn't be generated
        min_length: Minimum length of generated sequence
        eos_token_id: End-of-sentence token ID
        **kwargs: Additional parameters
        
    Returns:
        Function that processes logits
    """
    def process_logits(
        scores: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                for prev_token in set(input_ids[i].tolist()):
                    # If score > 0, then repetition penalty lowers it
                    # If score < 0, then repetition penalty increases it
                    if scores[i, prev_token] > 0:
                        scores[i, prev_token] /= repetition_penalty
                    else:
                        scores[i, prev_token] *= repetition_penalty
        
        # Prevent EOS before min_length
        if min_length > 0 and cur_len < min_length and eos_token_id is not None:
            scores[:, eos_token_id] = -float("inf")
        
        # No repeat n-gram constraint
        if no_repeat_ngram_size > 0:
            # For each batch
            for i in range(batch_size):
                # Extract input ids for this batch item
                generated_ids = input_ids[i].tolist()
                
                # Skip if current sequence is too short for n-gram
                if len(generated_ids) < no_repeat_ngram_size:
                    continue
                
                # Get the last n-gram
                ngram = tuple(generated_ids[-(no_repeat_ngram_size-1):])
                
                # Find prohibited tokens (which would create a repeated n-gram)
                for token_id in range(scores.shape[1]):
                    # Check if this token would create a repeat
                    check_ngram = ngram + (token_id,)
                    
                    # Check all possible positions of this n-gram
                    for j in range(len(generated_ids) - no_repeat_ngram_size + 1):
                        if tuple(generated_ids[j:j+no_repeat_ngram_size]) == check_ngram:
                            scores[i, token_id] = -float("inf")
                            break
        
        # Process bad words
        if bad_words_ids is not None:
            for i in range(batch_size):
                # Extract input ids for this batch item
                generated_ids = input_ids[i].tolist()
                
                # Check each bad word
                for bad_word_ids in bad_words_ids:
                    # Skip if the bad word is longer than the current sequence
                    if len(bad_word_ids) > len(generated_ids):
                        continue
                    
                    # Check if the end of the sequence matches the start of the bad word
                    match_size = min(len(generated_ids), len(bad_word_ids) - 1)
                    if match_size == 0:
                        continue
                    
                    if generated_ids[-match_size:] == bad_word_ids[:match_size]:
                        # Set the score for the next token in the bad word to -inf
                        next_token = bad_word_ids[match_size]
                        scores[i, next_token] = -float("inf")
        
        return scores
    
    return process_logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: Logits distribution (batch_size, vocabulary_size)
        top_k: Keep only top k tokens with highest probability (0 = no filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (1.0 = no filtering)
        filter_value: Value to assign to filtered tokens
        min_tokens_to_keep: Minimum number of tokens to keep
        
    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Ensure we keep at least min_tokens_to_keep
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, 
            index=sorted_indices, 
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from a distribution with temperature.
    
    Args:
        logits: Logits distribution (batch_size, vocabulary_size)
        temperature: Sampling temperature; higher is more random, lower is more greedy
        
    Returns:
        Sampled token IDs
    """
    if temperature == 0.0 or temperature < 1e-7:
        # When temperature approaches 0, use greedy sampling
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Sample from the distribution
    probs = torch.softmax(scaled_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_tokens


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: Union[str, List[str]],
    max_length: int = 100,
    min_length: int = 0,
    do_sample: bool = True,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    bad_words_ids: Optional[List[List[int]]] = None,
    num_return_sequences: int = 1,
    **kwargs
) -> List[str]:
    """
    Generate text from a model given input text.
    
    Args:
        model: Model to generate text with
        tokenizer: Tokenizer to encode/decode text
        input_text: Input text or batch of texts
        max_length: Maximum length of output sequence
        min_length: Minimum length of output sequence
        do_sample: Whether to use sampling (if False, uses greedy decoding)
        num_beams: Number of beams for beam search (1 = no beam search)
        temperature: Temperature for sampling
        top_k: Number of highest probability tokens to keep
        top_p: Probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeated tokens
        no_repeat_ngram_size: Size of n-grams that shouldn't be repeated
        bad_words_ids: List of token sequences that shouldn't be generated
        num_return_sequences: Number of sequences to return
        **kwargs: Additional model-specific parameters
        
    Returns:
        List of generated text strings
    """
    # Prepare input
    if isinstance(input_text, str):
        input_text = [input_text]
    
    batch_size = len(input_text)
    
    # Tokenize input
    encoding = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to model device
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Create logits processor
    logits_processor = get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    
    # Set up generation config based on strategy
    if num_beams > 1:
        # Beam search generation
        outputs = beam_search(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
    else:
        # Sampling or greedy generation
        if do_sample:
            outputs = sample(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
        else:
            outputs = greedy_search(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
            if num_return_sequences > 1:
                # Repeat outputs for multiple sequences in greedy mode
                outputs = outputs.repeat_interleave(num_return_sequences, dim=0)
    
    # Decode generated sequences
    result_texts = []
    input_lengths = [len(enc) for enc in encoding["input_ids"]]
    
    for i, output in enumerate(outputs):
        # Determine which input this output corresponds to
        input_idx = i // num_return_sequences
        input_length = input_lengths[input_idx]
        
        # Skip input tokens for decoding
        output_tokens = output[input_length:]
        
        # Decode to text
        output_text = tokenizer.decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        result_texts.append(output_text)
    
    return result_texts


def greedy_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 100,
    logits_processor: Optional[Callable] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **model_kwargs
) -> torch.Tensor:
    """
    Generate sequences using greedy search.
    
    Args:
        model: Model to generate with
        input_ids: Input token IDs
        attention_mask: Attention mask
        max_length: Maximum length of output sequence
        logits_processor: Function to process logits
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Generated sequences
    """
    # Initialize generation variables
    batch_size = input_ids.shape[0]
    device = input_ids.device
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Clone input_ids to avoid modifying the original
    input_ids = input_ids.clone()
    
    # Initialize attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # Loop until max length or all sequences are finished
    for _ in range(max_length - input_ids.shape[1]):
        # Prepare model inputs
        model_inputs = prepare_inputs_for_generation(
            input_ids,
            attention_mask,
            **model_kwargs
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
        
        # Process logits
        if logits_processor is not None:
            next_token_logits = logits_processor(next_token_logits, input_ids)
        
        # Get the next token (greedy)
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # Finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        # Update attention mask and add new tokens to input_ids
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        # Update unfinished sequences
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
        
        # Stop when all sequences are finished
        if unfinished_sequences.max() == 0:
            break
    
    return input_ids


def sample(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    logits_processor: Optional[Callable] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    num_return_sequences: int = 1,
    **model_kwargs
) -> torch.Tensor:
    """
    Generate sequences using sampling.
    
    Args:
        model: Model to generate with
        input_ids: Input token IDs
        attention_mask: Attention mask
        max_length: Maximum length of output sequence
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to keep
        top_p: Probability threshold for nucleus sampling
        logits_processor: Function to process logits
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        num_return_sequences: Number of sequences to return
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Generated sequences
    """
    # Initialize generation variables
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Expand inputs for multiple return sequences
    if num_return_sequences > 1:
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        
        effective_batch_size = batch_size * num_return_sequences
        unfinished_sequences = torch.ones(effective_batch_size, dtype=torch.long, device=device)
    else:
        effective_batch_size = batch_size
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Clone input_ids to avoid modifying the original
    input_ids = input_ids.clone()
    
    # Initialize attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # Loop until max length or all sequences are finished
    for _ in range(max_length - input_ids.shape[1]):
        # Prepare model inputs
        model_inputs = prepare_inputs_for_generation(
            input_ids,
            attention_mask,
            **model_kwargs
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
        
        # Process logits
        if logits_processor is not None:
            next_token_logits = logits_processor(next_token_logits, input_ids)
        
        # Apply top-k and top-p filtering
        if top_k > 0 or top_p < 1.0:
            next_token_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
        
        # Sample from the filtered distribution
        next_tokens = sample_with_temperature(next_token_logits, temperature)
        
        # Finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        # Update attention mask and add new tokens to input_ids
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((effective_batch_size, 1))], dim=-1)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        # Update unfinished sequences
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
        
        # Stop when all sequences are finished
        if unfinished_sequences.max() == 0:
            break
    
    return input_ids


def beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 100,
    num_beams: int = 5,
    logits_processor: Optional[Callable] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: float = 1.0,
    num_return_sequences: int = 1,
    **model_kwargs
) -> torch.Tensor:
    """
    Generate sequences using beam search.
    
    Args:
        model: Model to generate with
        input_ids: Input token IDs
        attention_mask: Attention mask
        max_length: Maximum length of output sequence
        num_beams: Number of beams
        logits_processor: Function to process logits
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        length_penalty: Length penalty factor (values < 1.0 promote shorter sequences)
        num_return_sequences: Number of sequences to return (must be <= num_beams)
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Generated sequences
    """
    # Check that we can return the desired number of sequences
    if num_return_sequences > num_beams:
        raise ValueError(f"num_return_sequences ({num_return_sequences}) has to be <= num_beams ({num_beams})")
    
    # Initialize generation variables
    batch_size = input_ids.shape[0]
    device = input_ids.device
    vocab_size = model.config.vocab_size if hasattr(model, "config") else 30000
    
    # Initialize attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    # Expand input to beam size
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous().view(batch_size * num_beams, -1)
    attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous().view(batch_size * num_beams, -1)
    
    # Initialize beam storage
    beam_scores = torch.zeros((batch_size, num_beams), device=device)
    beam_scores[:, 1:] = -1e9  # Start with first beam only
    beam_scores = beam_scores.view(-1)  # Flatten
    
    # Initialize generation loop variables
    done = [False for _ in range(batch_size)]
    generated_hyps = [[] for _ in range(batch_size)]
    
    # Loop until max length or all sequences are finished
    for step in range(max_length - input_ids.shape[1]):
        # Prepare model inputs
        model_inputs = prepare_inputs_for_generation(
            input_ids,
            attention_mask,
            **model_kwargs
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
        
        # Process logits
        if logits_processor is not None:
            next_token_logits = logits_processor(next_token_logits, input_ids)
        
        # Apply softmax to convert logits to probabilities
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
        
        # Update beam scores
        next_scores = beam_scores.unsqueeze(-1) + next_token_scores
        
        # Reshape for beam search
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        
        # Get the best beams and tokens
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1)
        
        # Convert token indices
        next_beam_indices = next_tokens // vocab_size
        next_token_indices = next_tokens % vocab_size
        
        # Form next input IDs and adjust scores
        next_input_ids = []
        beam_scores = next_scores.view(-1)
        
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                # This batch is already done, pad with zeros
                next_input_ids.extend([input_ids[batch_idx * num_beams]])
                continue
            
            # Find finished beams
            for beam_idx in range(num_beams):
                global_beam_idx = batch_idx * num_beams + beam_idx
                token_idx = next_token_indices[batch_idx, beam_idx].item()
                
                # Check if this beam is finished
                if eos_token_id is not None and token_idx == eos_token_id:
                    # Add finished hypothesis
                    beam_id = next_beam_indices[batch_idx, beam_idx].item()
                    current_beam_idx = batch_idx * num_beams + beam_id
                    seq = input_ids[current_beam_idx].clone()
                    score = beam_scores[global_beam_idx].item()
                    
                    # Apply length penalty
                    normalized_score = score / ((5 + len(seq)) ** length_penalty)
                    
                    generated_hyps[batch_idx].append((normalized_score, seq))
                    
                    # If we have enough hypotheses, mark as done
                    if len(generated_hyps[batch_idx]) >= num_beams:
                        done[batch_idx] = True
                
                # Add to next input IDs
                if not done[batch_idx]:
                    beam_id = next_beam_indices[batch_idx, beam_idx].item()
                    current_beam_idx = batch_idx * num_beams + beam_id
                    seq = input_ids[current_beam_idx].clone()
                    next_token = next_token_indices[batch_idx, beam_idx].unsqueeze(-1)
                    next_input_ids.append(torch.cat([seq, next_token], dim=-1))
            
            # If all batches are done, stop early
            if all(done):
                break
        
        # Update input IDs
        input_ids = torch.stack(next_input_ids)
        
        # Update attention mask
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    
    # Finalize results
    results = []
    
    # Process completed sequences
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            # Get the top num_return_sequences hypotheses
            generated_hyps[batch_idx].sort(key=lambda x: x[0], reverse=True)
            for j in range(min(num_return_sequences, len(generated_hyps[batch_idx]))):
                results.append(generated_hyps[batch_idx][j][1])
        else:
            # No finished sequences, use current beams
            for beam_idx in range(num_return_sequences):
                global_beam_idx = batch_idx * num_beams + beam_idx
                results.append(input_ids[global_beam_idx])
    
    # Pad to same length if needed
    max_gen_length = max(result.shape[0] for result in results)
    for i, result in enumerate(results):
        if result.shape[0] < max_gen_length:
            padding = result.new_ones((max_gen_length - result.shape[0],)) * pad_token_id
            results[i] = torch.cat([result, padding], dim=0)
    
    return torch.stack(results)


def token_probabilities(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Get probabilities for next token prediction.
    
    Args:
        model: Model to use for prediction
        tokenizer: Tokenizer to use
        input_text: Input text to generate from
        top_k: Number of top tokens to return
        
    Returns:
        List of dictionaries with token and probability for top_k tokens
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Convert to probabilities
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Get top-k tokens and probabilities
        topk_probs, topk_indices = torch.topk(next_token_probs, top_k)
        
        # Convert to list of dictionaries
        results = []
        for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
            token = tokenizer.decode([idx])
            results.append({
                "token": token,
                "id": idx.item(),
                "probability": prob.item()
            })
        
        return results
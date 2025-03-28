"""
Model utilities for QLLM evaluation.

This module provides utilities for model loading, preparation, and usage
during evaluation, centralizing functionality that was previously duplicated
across different evaluation components.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable, Type

import torch
import numpy as np

logger = logging.getLogger("qllm.evaluation")


class ModelUtils:
    """
    Utilities for model operations during evaluation.
    
    This class centralizes common model operations needed during evaluation,
    such as loading models, preparing them for evaluation, generating text,
    and extracting model outputs in standardized formats.
    """
    
    @staticmethod
    def load_model(
        model_path: str,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        precision: str = "float16",
        **model_kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model from the specified path.
        
        Args:
            model_path: Path to the model or model identifier
            model_type: Type of model to load (auto-detected if None)
            device: Device to load the model on (auto-detected if None)
            precision: Precision to use for model parameters ('float16', 'float32', 'bfloat16')
            **model_kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        return load_model(
            model_path=model_path,
            model_type=model_type,
            device=device,
            precision=precision,
            **model_kwargs
        )
    
    @staticmethod
    def prepare_for_evaluation(
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        max_length: int = 2048,
        padding_side: str = "left",
        truncation_side: str = "left",
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Prepare model and tokenizer for evaluation.
        
        Args:
            model: Model to prepare
            tokenizer: Tokenizer to prepare
            device: Device to move the model to
            max_length: Maximum sequence length
            padding_side: Side to pad sequences on ('left' or 'right')
            truncation_side: Side to truncate sequences on ('left' or 'right')
            **kwargs: Additional preparation parameters
            
        Returns:
            Tuple of (prepared_model, prepared_tokenizer)
        """
        return prepare_model_for_evaluation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            **kwargs
        )
    
    @staticmethod
    def generate_text(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text from a model given a prompt.
        
        Args:
            model: Model to generate text with
            tokenizer: Tokenizer to use
            prompt: Prompt to generate text from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling (if False, uses greedy decoding)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated text strings
        """
        # Handle different model types
        try:
            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            
            # Set up generation parameters
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                **generation_kwargs
            }
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(input_ids, **generation_config)
            
            # Decode outputs
            prompt_length = inputs["input_ids"].size(1)
            generated_texts = []
            
            for output in outputs:
                # Skip prompt tokens
                output_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                generated_texts.append(output_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ["Error generating text."]
    
    @staticmethod
    def extract_hidden_states(
        model: Any,
        tokenizer: Any,
        text: str,
        layers: Optional[List[int]] = None,
        aggregate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states from a model for a given text.
        
        Args:
            model: Model to extract hidden states from
            tokenizer: Tokenizer to use
            text: Text to extract hidden states for
            layers: List of layers to extract hidden states from (None for all)
            aggregate: Whether to aggregate hidden states across tokens
            
        Returns:
            Dictionary mapping layer names to hidden state tensors
        """
        # Encode the text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Set up extraction
        model.config.output_hidden_states = True
        
        # Get outputs
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Extract hidden states
        all_hidden_states = outputs.hidden_states
        
        # Select layers
        if layers is not None:
            selected_hidden_states = {f"layer_{layer}": all_hidden_states[layer] for layer in layers}
        else:
            selected_hidden_states = {f"layer_{i}": layer for i, layer in enumerate(all_hidden_states)}
        
        # Aggregate if requested
        if aggregate:
            for layer_name, hidden_state in selected_hidden_states.items():
                # Average across tokens
                selected_hidden_states[layer_name] = hidden_state.mean(dim=1)
        
        return selected_hidden_states
    
    @staticmethod
    def extract_attention_weights(
        model: Any,
        tokenizer: Any,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from a model for a given text.
        
        Args:
            model: Model to extract attention weights from
            tokenizer: Tokenizer to use
            text: Text to extract attention weights for
            layers: List of layers to extract attention weights from (None for all)
            
        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        # Encode the text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Set up extraction
        model.config.output_attentions = True
        
        # Get outputs
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Extract attention weights
        all_attentions = outputs.attentions
        
        # Select layers
        if layers is not None:
            selected_attentions = {f"layer_{layer}": all_attentions[layer] for layer in layers}
        else:
            selected_attentions = {f"layer_{i}": layer for i, layer in enumerate(all_attentions)}
        
        return selected_attentions
    
    @staticmethod
    def get_model_embedding(
        model: Any,
        tokenizer: Any,
        text: str,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get an embedding for a text from a model.
        
        Args:
            model: Model to get embedding from
            tokenizer: Tokenizer to use
            text: Text to get embedding for
            pooling: Pooling method ('mean', 'max', 'cls', 'last')
            
        Returns:
            Embedding tensor
        """
        # Encode the text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Get outputs
        with torch.no_grad():
            if hasattr(model, "get_embeddings"):
                # Use model's embedding method if available
                embedding = model.get_embeddings(input_ids)
            else:
                # Extract embedding from model outputs
                outputs = model(input_ids, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                
                # Apply pooling
                if pooling == "mean":
                    embedding = last_hidden_state.mean(dim=1)
                elif pooling == "max":
                    embedding = last_hidden_state.max(dim=1).values
                elif pooling == "cls":
                    embedding = last_hidden_state[:, 0, :]
                elif pooling == "last":
                    embedding = last_hidden_state[:, -1, :]
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")
        
        return embedding.squeeze(0)


def load_model(
    model_path: str,
    model_type: Optional[str] = None,
    device: Optional[str] = None,
    precision: str = "float16",
    **model_kwargs
) -> Tuple[Any, Any]:
    """
    Load a model from the specified path.
    
    Args:
        model_path: Path to the model or model identifier
        model_type: Type of model to load (auto-detected if None)
        device: Device to load the model on (auto-detected if None)
        precision: Precision to use for model parameters ('float16', 'float32', 'bfloat16')
        **model_kwargs: Additional model-specific parameters
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Loading model from {model_path} on {device}")
    
    try:
        # Try to load using Hugging Face transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine dtype for model loading
            if precision == "float16":
                dtype = torch.float16
            elif precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device != "mps" else None,  # device_map not supported for MPS
                **model_kwargs
            )
            
            # Handle MPS device separately
            if device == "mps":
                model = model.to(device)
            
            # Set evaluation mode
            model.eval()
            
            return model, tokenizer
            
        except (ImportError, ValueError) as e:
            logger.warning(f"Failed to load with transformers: {e}")
            raise
            
    except Exception as e:
        # Fall back to generic PyTorch loading
        logger.warning(f"Could not load model with standard methods: {e}")
        try:
            # Try to load with PyTorch
            model = torch.load(model_path, map_location=device)
            
            # Try to load tokenizer from the same directory
            tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "r") as f:
                    tokenizer_config = json.load(f)
                
                # Simple tokenizer implementation
                class SimpleTokenizer:
                    def __init__(self, config):
                        self.vocab = config.get("vocab", {})
                        self.eos_token_id = config.get("eos_token_id", 0)
                        self.pad_token_id = config.get("pad_token_id", 0)
                
                tokenizer = SimpleTokenizer(tokenizer_config)
            else:
                # Dummy tokenizer
                tokenizer = None
                logger.warning("No tokenizer found, using None")
            
            return model, tokenizer
            
        except Exception as inner_e:
            logger.error(f"Failed to load model: {inner_e}")
            raise ValueError(f"Could not load model from {model_path}: {inner_e}")


def prepare_model_for_evaluation(
    model: Any,
    tokenizer: Any,
    device: Optional[str] = None,
    max_length: int = 2048,
    padding_side: str = "left",
    truncation_side: str = "left",
    **kwargs
) -> Tuple[Any, Any]:
    """
    Prepare model and tokenizer for evaluation.
    
    Args:
        model: Model to prepare
        tokenizer: Tokenizer to prepare
        device: Device to move the model to
        max_length: Maximum sequence length
        padding_side: Side to pad sequences on ('left' or 'right')
        truncation_side: Side to truncate sequences on ('left' or 'right')
        **kwargs: Additional preparation parameters
        
    Returns:
        Tuple of (prepared_model, prepared_tokenizer)
    """
    # Auto-detect device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Move model to device if needed
    if str(next(model.parameters()).device) != str(device):
        logger.info(f"Moving model to {device}")
        model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Configure tokenizer
    if tokenizer is not None:
        # Set padding and truncation sides
        tokenizer.padding_side = padding_side
        tokenizer.truncation_side = truncation_side
        
        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set default parameters
        tokenizer.model_max_length = max_length
    
    return model, tokenizer
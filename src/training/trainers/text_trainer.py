"""
Text Trainer for QLLM.

This module provides a specialized trainer for text generation models,
extending the base trainer with text-specific functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class TextTrainer(BaseTrainer):
    """
    Specialized trainer for text generation models.
    
    This trainer extends the base trainer with text-specific functionality,
    including specialized text-based metrics and handling for various
    text generation scenarios.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the text trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Text-specific initialization
        self.vocab_size = getattr(self.model_config, "vocab_size", 0)
        if self.vocab_size == 0 and hasattr(self.model, "vocab_size"):
            self.vocab_size = self.model.vocab_size
        
        # Add text-specific hooks
        self.add_hook("post_step", self._calculate_text_metrics)
        self.add_hook("post_eval", self._generate_text_samples)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for text generation training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle standard input formats for text models
        if "labels" not in batch and "input_ids" in batch:
            # For causal language modeling, labels are typically the input shifted right
            input_ids = batch["input_ids"]
            
            # Create shifted labels (next tokens)
            labels = input_ids.clone()
            
            # Set the first token's label to -100 (ignored in loss calculation)
            if labels.shape[1] > 1:
                labels[:, 0] = -100
                labels[:, 1:] = input_ids[:, :-1]
            
            # Add labels to batch
            batch["labels"] = labels
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate perplexity if loss is available
        if "loss" in outputs and torch.is_tensor(outputs["loss"]):
            outputs["perplexity"] = torch.exp(outputs["loss"])
        
        return outputs
    
    def _calculate_text_metrics(self, **kwargs) -> None:
        """
        Calculate text-specific metrics after each step.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        metrics = kwargs.get("metrics", {})
        
        # Skip if no metrics available
        if not metrics:
            return
        
        # Add metrics to accumulated metrics
        for key, value in metrics.items():
            if key in ["loss", "perplexity"]:
                if key not in self.accumulated_metrics:
                    self.accumulated_metrics[key] = 0.0
                
                # Use moving average for accumulation
                alpha = 0.1  # Weight for new value
                self.accumulated_metrics[key] = (
                    (1 - alpha) * self.accumulated_metrics[key] + alpha * value
                )
    
    def _generate_text_samples(self, **kwargs) -> None:
        """
        Generate text samples during evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if not evaluation
        if not hasattr(self.model, "generate") or self.eval_dataloader is None:
            return
        
        # Get a sample batch from the evaluation dataset
        sample_batch = next(iter(self.eval_dataloader))
        sample_batch = self._batch_to_device(sample_batch)
        
        # Try to generate text
        try:
            # Get input ids
            input_ids = sample_batch.get("input_ids", None)
            if input_ids is None:
                return
            
            # Select a random sample from the batch
            import random
            sample_idx = random.randint(0, input_ids.size(0) - 1)
            sample_input = input_ids[sample_idx:sample_idx+1]
            
            # Generate continuation
            with torch.no_grad():
                generated = self.model.generate(
                    sample_input,
                    max_length=min(512, sample_input.size(1) + 50),
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    temperature=0.8
                )
            
            # Try to decode generated text
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                # Decode the sample input
                sample_text = self.model.tokenizer.decode(
                    sample_input[0], skip_special_tokens=True
                )
                
                # Get the newly generated continuation
                generated_text = self.model.tokenizer.decode(
                    generated[0, sample_input.size(1):], skip_special_tokens=True
                )
                
                # Log the sample
                self._log_text_sample(sample_text, generated_text)
        except Exception as e:
            # Log error but don't interrupt training
            self._log_error(f"Error generating text sample: {e}")
    
    def _log_text_sample(self, prompt: str, continuation: str) -> None:
        """
        Log a text sample with prompt and continuation.
        
        Args:
            prompt: Input prompt text
            continuation: Generated continuation
        """
        self._log_info("\n" + "=" * 40)
        self._log_info("TEXT GENERATION SAMPLE")
        self._log_info("=" * 40)
        self._log_info(f"PROMPT: {prompt}")
        self._log_info("-" * 40)
        self._log_info(f"CONTINUATION: {continuation}")
        self._log_info("=" * 40 + "\n")
    
    def _log_info(self, message: str) -> None:
        """Log an info message."""
        import logging
        logger = logging.getLogger("qllm.training")
        logger.info(message)
    
    def _log_error(self, message: str) -> None:
        """Log an error message."""
        import logging
        logger = logging.getLogger("qllm.training")
        logger.error(message)
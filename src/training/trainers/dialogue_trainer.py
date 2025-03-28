"""
Dialogue Trainer for QLLM.

This module provides a specialized trainer for dialogue models,
extending the base trainer with dialogue-specific functionality
for training conversational agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class DialogueTrainer(BaseTrainer):
    """
    Specialized trainer for dialogue models.
    
    This trainer extends the base trainer with dialogue-specific functionality,
    including handling of conversation context, response generation,
    and dialogue-specific metrics.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the dialogue trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Extract dialogue-specific parameters
        self.max_history_turns = kwargs.pop("max_history_turns", 3)
        self.response_prefix = kwargs.pop("response_prefix", "")
        self.separate_input_response = kwargs.pop("separate_input_response", True)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Add dialogue-specific hooks
        self.add_hook("post_step", self._calculate_dialogue_metrics)
        self.add_hook("post_eval", self._generate_dialogue_samples)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for dialogue training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle dialogue-specific format
        if "input_ids" in batch and "response_ids" in batch and "labels" not in batch:
            # Prepare labels for dialogue training
            input_ids = batch["input_ids"]
            response_ids = batch["response_ids"]
            
            if self.separate_input_response:
                # Use response_ids as labels but mask the input part
                labels = response_ids.clone()
                
                # If input_lengths is provided, use it to mask the input part
                if "input_lengths" in batch:
                    input_lengths = batch["input_lengths"]
                    for i, length in enumerate(input_lengths):
                        if length > 0:
                            labels[i, :length] = -100  # Mask with -100 to ignore in loss
                
                # Add labels to batch
                batch["labels"] = labels
            else:
                # Concatenate input and response, then create shifted labels
                combined_ids = torch.cat([input_ids, response_ids], dim=1)
                labels = combined_ids.clone()
                
                # Shift labels to predict next token
                labels[:, :-1] = combined_ids[:, 1:]
                labels[:, -1] = -100  # Mask the last token
                
                # Add attention mask that covers the full sequence
                attention_mask = torch.ones_like(labels)
                
                # Update batch
                batch["input_ids"] = combined_ids
                batch["labels"] = labels
                batch["attention_mask"] = attention_mask
        
        # Create a filtered batch with only the keys the model expects
        filtered_batch = {}
        
        # Model expects standard HF Transformer keys
        allowed_keys = [
            "input_ids", "attention_mask", "labels", "position_ids",
            "output_hidden_states", "output_attentions", "return_dict"
        ]
        
        # Standardize attention mask naming
        if "input_attention_mask" in batch and "attention_mask" not in batch:
            batch["attention_mask"] = batch.pop("input_attention_mask")
            
        # Copy only the allowed keys to the filtered batch, ensuring correct data types
        for key in allowed_keys:
            if key in batch:
                # Make sure input_ids and labels are long tensors
                if key in ["input_ids", "labels"]:
                    filtered_batch[key] = batch[key].long()
                else:
                    filtered_batch[key] = batch[key]
        
                # Special case to handle input_ids with zero length
                if key == "input_ids" and filtered_batch[key].size(1) == 0 and "response_ids" in batch:
                    # Use response_ids as input_ids if input_ids is empty
                    filtered_batch[key] = batch["response_ids"].long()
        
        # Add token_type_ids if needed
        if "token_type_ids" in batch:
            filtered_batch["token_type_ids"] = batch["token_type_ids"].long()
        
        # Truncate sequences to max_length of 64 for the model's internal constraints
        max_model_seq_len = 64
        # First, ensure all tensors are at most max_model_seq_len
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in filtered_batch and filtered_batch[key].size(1) > max_model_seq_len:
                filtered_batch[key] = filtered_batch[key][:, :max_model_seq_len]
        
        # Ensure all tensors have matching sequence lengths
        if "input_ids" in filtered_batch:
            input_seq_len = filtered_batch["input_ids"].size(1)
            batch_size = filtered_batch["input_ids"].size(0)
            
            # Force labels to match input_ids length
            if "labels" in filtered_batch and filtered_batch["labels"].size(1) != input_seq_len:
                device = filtered_batch["labels"].device
                
                if filtered_batch["labels"].size(1) > input_seq_len:
                    # Truncate labels
                    filtered_batch["labels"] = filtered_batch["labels"][:, :input_seq_len]
                else:
                    # Pad labels with -100 (ignore index)
                    padding_length = input_seq_len - filtered_batch["labels"].size(1)
                    padding = torch.full(
                        (batch_size, padding_length),
                        fill_value=-100,
                        dtype=filtered_batch["labels"].dtype,
                        device=device
                    )
                    filtered_batch["labels"] = torch.cat([
                        filtered_batch["labels"], padding
                    ], dim=1)
                
            # Fix attention_mask to match input_ids length
            if "attention_mask" not in filtered_batch or filtered_batch["attention_mask"].size(1) == 0:
                # Create new attention mask
                filtered_batch["attention_mask"] = torch.ones(
                    batch_size, input_seq_len,
                    dtype=torch.long,
                    device=filtered_batch["input_ids"].device
                )
            elif filtered_batch["attention_mask"].size(1) != input_seq_len:
                device = filtered_batch["attention_mask"].device
                
                if filtered_batch["attention_mask"].size(1) > input_seq_len:
                    # Truncate attention mask
                    filtered_batch["attention_mask"] = filtered_batch["attention_mask"][:, :input_seq_len]
                else:
                    # Pad attention mask with ones
                    padding_length = input_seq_len - filtered_batch["attention_mask"].size(1)
                    padding = torch.ones(
                        (batch_size, padding_length),
                        dtype=filtered_batch["attention_mask"].dtype,
                        device=device
                    )
                    filtered_batch["attention_mask"] = torch.cat([
                        filtered_batch["attention_mask"], padding
                    ], dim=1)
            filtered_batch["labels"] = filtered_batch["labels"][:, :max_model_seq_len]
            
        # Forward pass with only the filtered batch
        try:
            outputs = self.model(**filtered_batch)
        except RuntimeError as e:
            # Print diagnostic info if there's an error
            batch_sizes = {k: v.shape if hasattr(v, 'shape') else 'not tensor' for k, v in filtered_batch.items()}
            print(f"Error with batch shapes: {batch_sizes}")
            raise
        
        # Calculate dialogue-specific metrics
        if "loss" in outputs and torch.is_tensor(outputs["loss"]):
            # Calculate response perplexity (only on the response part)
            response_loss = outputs.get("response_loss", outputs["loss"])
            outputs["response_perplexity"] = torch.exp(response_loss)
        
        return outputs
    
    def _calculate_dialogue_metrics(self, step_metrics=None, **kwargs) -> None:
        """
        Calculate dialogue-specific metrics after each step.
        
        Args:
            step_metrics: Metrics from the current step
            **kwargs: Additional keyword arguments from the hook
        """
        # Simply log the metrics
        if hasattr(self, 'accumulated_metrics'):
            try:
                # Get metrics from kwargs if not directly provided
                metrics = {}
                if step_metrics is not None and isinstance(step_metrics, dict):
                    metrics = step_metrics
                elif "metrics" in kwargs and isinstance(kwargs["metrics"], dict):
                    metrics = kwargs["metrics"]
                
                # Process only if we have valid metrics
                if metrics:
                    for key, value in metrics.items():
                        if key in ["loss", "response_perplexity", "response_loss"]:
                            if key not in self.accumulated_metrics:
                                self.accumulated_metrics[key] = 0.0
                            
                            # Use moving average for accumulation
                            alpha = 0.1  # Weight for new value
                            self.accumulated_metrics[key] = (
                                (1 - alpha) * self.accumulated_metrics[key] + alpha * value
                            )
            except Exception as e:
                # Silently handle any errors to prevent training interruption
                pass
    
    def _generate_dialogue_samples(self, **kwargs) -> None:
        """
        Generate dialogue samples during evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if not evaluation or model doesn't support generation
        if not hasattr(self.model, "generate") or self.eval_dataloader is None:
            return
        
        # Get a sample batch from the evaluation dataset
        sample_batch = next(iter(self.eval_dataloader))
        sample_batch = self._batch_to_device(sample_batch)
        
        # Try to generate dialogue responses
        try:
            # Get input ids
            input_ids = sample_batch.get("input_ids", None)
            if input_ids is None:
                return
            
            # Select a random sample from the batch
            import random
            sample_idx = random.randint(0, input_ids.size(0) - 1)
            sample_input = input_ids[sample_idx:sample_idx+1]
            
            # Generate response
            with torch.no_grad():
                generated = self.model.generate(
                    sample_input,
                    max_length=min(512, sample_input.size(1) + 100),
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    temperature=0.85,
                    num_return_sequences=1
                )
            
            # Try to decode generated text
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                # Decode the sample input (user message)
                user_message = self.model.tokenizer.decode(
                    sample_input[0], skip_special_tokens=True
                )
                
                # Get the newly generated response
                generated_response = self.model.tokenizer.decode(
                    generated[0, sample_input.size(1):], skip_special_tokens=True
                )
                
                # Get reference response if available
                reference_response = ""
                if "response_ids" in sample_batch:
                    response_ids = sample_batch["response_ids"][sample_idx]
                    reference_response = self.model.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                
                # Log the dialogue sample
                self._log_dialogue_sample(user_message, generated_response, reference_response)
        except Exception as e:
            # Log error but don't interrupt training
            self._log_error(f"Error generating dialogue sample: {e}")
    
    def _log_dialogue_sample(
        self, 
        user_message: str, 
        generated_response: str,
        reference_response: str = ""
    ) -> None:
        """
        Log a dialogue sample with user message and response.
        
        Args:
            user_message: User message
            generated_response: Generated model response
            reference_response: Reference (ground truth) response if available
        """
        self._log_info("\n" + "=" * 40)
        self._log_info("DIALOGUE GENERATION SAMPLE")
        self._log_info("=" * 40)
        self._log_info(f"USER: {user_message}")
        self._log_info(f"MODEL: {generated_response}")
        
        if reference_response:
            self._log_info("-" * 40)
            self._log_info(f"REFERENCE: {reference_response}")
        
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
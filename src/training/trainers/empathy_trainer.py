"""
Empathy Trainer for QLLM.

This module provides a specialized trainer for empathy models,
extending the base trainer with functionality specific to training
models that can generate empathetic responses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class EmpathyTrainer(BaseTrainer):
    """
    Specialized trainer for empathy models.
    
    This trainer extends the base trainer with functionality specific to
    training models that generate empathetic responses, including specialized
    empathy metrics and empathy-focused generation.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the empathy trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Extract empathy-specific parameters
        self.empathy_levels = kwargs.pop("empathy_levels", 5)
        self.emotion_categories = kwargs.pop("emotion_categories", None)
        self.empathy_weight = kwargs.pop("empathy_weight", 1.0)
        self.content_weight = kwargs.pop("content_weight", 1.0)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Add empathy-specific hooks
        self.add_hook("post_step", self._calculate_empathy_metrics)
        self.add_hook("post_eval", self._generate_empathy_samples)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for empathy training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Process empathy-specific inputs
        if "empathy_scores" in batch and "empathy_weights" not in batch:
            # Create empathy weights based on scores if not provided
            empathy_scores = batch["empathy_scores"]
            # Higher scores get higher weights to emphasize high-empathy examples
            batch["empathy_weights"] = 1.0 + empathy_scores / self.empathy_levels
        
        if "emotion_labels" in batch and hasattr(self.model, "set_emotion_context"):
            # Set emotion context in model if supported
            self.model.set_emotion_context(batch["emotion_labels"])
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate custom empathy loss if needed and not already calculated by model
        if "loss" in outputs and "empathy_loss" not in outputs and "empathy_scores" in batch:
            # Standard content loss from model
            content_loss = outputs["loss"]
            
            # Calculate empathy-weighted loss
            if "empathy_weights" in batch:
                empathy_weights = batch["empathy_weights"]
                batch_size = empathy_weights.size(0)
                
                # Scale weights to mean 1.0
                empathy_weights = empathy_weights / empathy_weights.mean()
                
                # Apply weights to loss
                empathy_loss = content_loss * empathy_weights.mean()
                
                # Combine losses
                outputs["content_loss"] = content_loss
                outputs["empathy_loss"] = empathy_loss
                outputs["loss"] = (
                    self.content_weight * content_loss + 
                    self.empathy_weight * empathy_loss
                )
            
            # Calculate empathy prediction loss if model outputs empathy predictions
            if "empathy_predictions" in outputs and "empathy_scores" in batch:
                empathy_pred_loss = F.mse_loss(
                    outputs["empathy_predictions"], 
                    batch["empathy_scores"]
                )
                outputs["empathy_pred_loss"] = empathy_pred_loss
                
                # Add to total loss
                outputs["loss"] = outputs["loss"] + 0.1 * empathy_pred_loss
        
        return outputs
    
    def _calculate_empathy_metrics(self, **kwargs) -> None:
        """
        Calculate empathy-specific metrics after each step.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        metrics = kwargs.get("metrics", {})
        batch = kwargs.get("batch", {})
        
        # Skip if no metrics or batch available
        if not metrics or not batch:
            return
        
        # Calculate empathy prediction accuracy if not already calculated
        if ("empathy_prediction_accuracy" not in metrics and 
            "empathy_predictions" in metrics and 
            "empathy_scores" in batch):
            
            empathy_preds = metrics["empathy_predictions"]
            empathy_scores = batch["empathy_scores"]
            
            if isinstance(empathy_preds, torch.Tensor) and isinstance(empathy_scores, torch.Tensor):
                # Calculate mean absolute error
                mae = torch.abs(empathy_preds - empathy_scores).mean().item()
                metrics["empathy_mae"] = mae
                
                # Calculate empathy level match accuracy (rounded to nearest level)
                rounded_preds = torch.round(empathy_preds * self.empathy_levels) / self.empathy_levels
                rounded_scores = torch.round(empathy_scores * self.empathy_levels) / self.empathy_levels
                accuracy = (rounded_preds == rounded_scores).float().mean().item()
                metrics["empathy_level_accuracy"] = accuracy
        
        # Calculate emotion recognition accuracy if available
        if ("emotion_accuracy" not in metrics and 
            "emotion_predictions" in metrics and 
            "emotion_labels" in batch):
            
            emotion_preds = metrics["emotion_predictions"]
            emotion_labels = batch["emotion_labels"]
            
            if isinstance(emotion_preds, torch.Tensor) and isinstance(emotion_labels, torch.Tensor):
                # Get predicted emotion
                pred_emotions = torch.argmax(emotion_preds, dim=-1)
                
                # Calculate accuracy
                accuracy = (pred_emotions == emotion_labels).float().mean().item()
                metrics["emotion_accuracy"] = accuracy
        
        # Add metrics to accumulated metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in self.accumulated_metrics:
                    self.accumulated_metrics[key] = 0.0
                
                # Use moving average for accumulation
                alpha = 0.1  # Weight for new value
                self.accumulated_metrics[key] = (
                    (1 - alpha) * self.accumulated_metrics[key] + alpha * value
                )
    
    def _generate_empathy_samples(self, **kwargs) -> None:
        """
        Generate empathy samples during evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if not evaluation or model doesn't support generation
        if not hasattr(self.model, "generate") or self.eval_dataloader is None:
            return
        
        # Get a sample batch from the evaluation dataset
        sample_batch = next(iter(self.eval_dataloader))
        sample_batch = self._batch_to_device(sample_batch)
        
        # Try to generate empathetic responses at different empathy levels
        try:
            # Get input ids
            input_ids = sample_batch.get("input_ids", None)
            if input_ids is None:
                return
            
            # Select a random sample from the batch
            import random
            sample_idx = random.randint(0, input_ids.size(0) - 1)
            sample_input = input_ids[sample_idx:sample_idx+1]
            
            # Generate responses at different empathy levels
            empathy_levels = [0.0, 0.5, 1.0]  # Low, medium, high
            responses = []
            
            with torch.no_grad():
                for empathy_level in empathy_levels:
                    # Set empathy level if model supports it
                    if hasattr(self.model, "set_empathy_level"):
                        self.model.set_empathy_level(empathy_level)
                    
                    # Generate response
                    generated = self.model.generate(
                        sample_input,
                        max_length=min(512, sample_input.size(1) + 100),
                        do_sample=True,
                        top_p=0.92,
                        top_k=50,
                        temperature=0.85
                    )
                    
                    # Get the newly generated response
                    if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                        response_text = self.model.tokenizer.decode(
                            generated[0, sample_input.size(1):], skip_special_tokens=True
                        )
                        responses.append(response_text)
            
            # Get input text and reference response
            input_text = ""
            reference_response = ""
            reference_empathy = -1
            
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                # Decode input text
                input_text = self.model.tokenizer.decode(
                    sample_input[0], skip_special_tokens=True
                )
                
                # Get reference response if available
                if "response_ids" in sample_batch:
                    response_ids = sample_batch["response_ids"][sample_idx]
                    reference_response = self.model.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                
                # Get reference empathy score if available
                if "empathy_scores" in sample_batch:
                    reference_empathy = sample_batch["empathy_scores"][sample_idx].item()
            
            # Get reference emotion if available
            reference_emotion = ""
            if "emotion_labels" in sample_batch and self.emotion_categories:
                emotion_idx = sample_batch["emotion_labels"][sample_idx].item()
                if 0 <= emotion_idx < len(self.emotion_categories):
                    reference_emotion = self.emotion_categories[emotion_idx]
            
            # Log the empathy samples
            self._log_empathy_samples(
                input_text, 
                responses, 
                empathy_levels, 
                reference_response, 
                reference_empathy,
                reference_emotion
            )
        except Exception as e:
            # Log error but don't interrupt training
            self._log_error(f"Error generating empathy samples: {e}")
    
    def _log_empathy_samples(
        self,
        input_text: str,
        responses: List[str],
        empathy_levels: List[float],
        reference_response: str = "",
        reference_empathy: float = -1,
        reference_emotion: str = ""
    ) -> None:
        """
        Log empathy samples with different empathy levels.
        
        Args:
            input_text: Input text
            responses: List of generated responses
            empathy_levels: List of empathy levels used
            reference_response: Reference (ground truth) response if available
            reference_empathy: Reference empathy score if available
            reference_emotion: Reference emotion if available
        """
        self._log_info("\n" + "=" * 40)
        self._log_info("EMPATHY GENERATION SAMPLES")
        self._log_info("=" * 40)
        self._log_info(f"INPUT: {input_text}")
        
        if reference_emotion:
            self._log_info(f"DETECTED EMOTION: {reference_emotion}")
        
        self._log_info("-" * 40)
        
        # Log responses at different empathy levels
        for i, (response, level) in enumerate(zip(responses, empathy_levels)):
            label = "LOW" if level < 0.3 else "MEDIUM" if level < 0.7 else "HIGH"
            self._log_info(f"RESPONSE (EMPATHY LEVEL {label}, {level:.1f}):")
            self._log_info(f"{response}")
            self._log_info("-" * 40)
        
        # Log reference response if available
        if reference_response:
            empathy_label = ""
            if reference_empathy >= 0:
                label = "LOW" if reference_empathy < 0.3 else "MEDIUM" if reference_empathy < 0.7 else "HIGH"
                empathy_label = f" (EMPATHY LEVEL {label}, {reference_empathy:.1f})"
            
            self._log_info(f"REFERENCE RESPONSE{empathy_label}:")
            self._log_info(f"{reference_response}")
        
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
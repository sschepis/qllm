"""
Intent Trainer for QLLM.

This module provides a specialized trainer for intent classification models,
extending the base trainer with functionality specific to training models
that can classify user intentions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class IntentTrainer(BaseTrainer):
    """
    Specialized trainer for intent classification models.
    
    This trainer extends the base trainer with intent classification-specific
    functionality, including specialized metrics and training procedures
    for models that classify user intentions.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the intent trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Extract intent-specific parameters
        self.intent_categories = kwargs.pop("intent_categories", None)
        self.num_intents = kwargs.pop("num_intents", 0)
        self.multi_intent = kwargs.pop("multi_intent", False)
        self.intent_threshold = kwargs.pop("intent_threshold", 0.5)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Set number of intents if not provided
        if self.num_intents == 0 and self.intent_categories is not None:
            self.num_intents = len(self.intent_categories)
        
        # Add intent-specific hooks
        self.add_hook("post_step", self._calculate_intent_metrics)
        self.add_hook("post_eval", self._log_intent_confusion_matrix)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for intent classification training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle intent-specific format
        if "intent_labels" in batch and "labels" not in batch:
            batch["labels"] = batch["intent_labels"]
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate classification metrics if not already calculated
        if "logits" in outputs and "labels" in batch:
            intent_logits = outputs["logits"]
            intent_labels = batch["labels"]
            
            # For multi-label classification
            if self.multi_intent:
                # Use binary cross-entropy loss for multi-label
                if "loss" not in outputs:
                    loss = F.binary_cross_entropy_with_logits(
                        intent_logits.float(), 
                        intent_labels.float()
                    )
                    outputs["loss"] = loss
                
                # Calculate F1 score
                with torch.no_grad():
                    predictions = (torch.sigmoid(intent_logits) > self.intent_threshold).float()
                    tp = (predictions * intent_labels).sum(dim=1)
                    precision_denom = predictions.sum(dim=1)
                    recall_denom = intent_labels.sum(dim=1)
                    
                    # Avoid division by zero
                    precision = torch.where(
                        precision_denom > 0,
                        tp / precision_denom,
                        torch.ones_like(tp)
                    )
                    recall = torch.where(
                        recall_denom > 0,
                        tp / recall_denom,
                        torch.ones_like(tp)
                    )
                    
                    # Calculate F1
                    f1_denom = precision + recall
                    f1 = torch.where(
                        f1_denom > 0,
                        2 * (precision * recall) / f1_denom,
                        torch.zeros_like(tp)
                    ).mean()
                    
                    outputs["f1_score"] = f1.item()
            else:
                # For single-label classification
                if "loss" not in outputs:
                    loss = F.cross_entropy(intent_logits, intent_labels)
                    outputs["loss"] = loss
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(intent_logits, dim=-1)
                    accuracy = (predictions == intent_labels).float().mean()
                    outputs["accuracy"] = accuracy.item()
        
        return outputs
    
    def _calculate_intent_metrics(self, **kwargs) -> None:
        """
        Calculate intent-specific metrics after each step.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        metrics = kwargs.get("metrics", {})
        batch = kwargs.get("batch", {})
        
        # Skip if no metrics or batch available
        if not metrics or not batch:
            return
        
        # Calculate confusion matrix for single-label intents
        if (not self.multi_intent and 
            "intent_confusion" not in self.accumulated_metrics and
            "logits" in metrics and 
            "labels" in batch):
            
            intent_logits = metrics["logits"]
            intent_labels = batch["labels"]
            
            if isinstance(intent_logits, torch.Tensor) and isinstance(intent_labels, torch.Tensor):
                # Get predicted intents
                predictions = torch.argmax(intent_logits, dim=-1)
                
                # Initialize confusion matrix if not exists
                if "intent_confusion" not in self.accumulated_metrics:
                    self.accumulated_metrics["intent_confusion"] = torch.zeros(
                        self.num_intents, self.num_intents,
                        device=predictions.device
                    )
                
                # Update confusion matrix
                for pred, label in zip(predictions, intent_labels):
                    self.accumulated_metrics["intent_confusion"][label, pred] += 1
        
        # Add other metrics to accumulated metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in self.accumulated_metrics:
                    self.accumulated_metrics[key] = 0.0
                
                # Use moving average for accumulation
                alpha = 0.1  # Weight for new value
                self.accumulated_metrics[key] = (
                    (1 - alpha) * self.accumulated_metrics[key] + alpha * value
                )
    
    def _log_intent_confusion_matrix(self, **kwargs) -> None:
        """
        Log intent confusion matrix after evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if multi-intent or no confusion matrix
        if self.multi_intent or "intent_confusion" not in self.accumulated_metrics:
            return
        
        # Get confusion matrix
        confusion = self.accumulated_metrics["intent_confusion"]
        
        # Skip if empty
        if confusion.sum() == 0:
            return
        
        # Convert confusion matrix to percentages (row-normalized)
        confusion_pct = confusion / (confusion.sum(dim=1, keepdim=True) + 1e-10)
        
        # Generate confusion matrix visualization
        self._log_info("\n" + "=" * 40)
        self._log_info("INTENT CLASSIFICATION CONFUSION MATRIX")
        self._log_info("=" * 40)
        
        # Header row with predicted intents
        header = "True\\Pred |"
        for i in range(min(10, self.num_intents)):  # Limit to 10 intents for readability
            intent_name = str(i)
            if self.intent_categories and i < len(self.intent_categories):
                intent_name = self.intent_categories[i][:5]  # Truncate to 5 chars
            header += f" {intent_name:>5} |"
        self._log_info(header)
        
        # Separator
        separator = "-" * len(header)
        self._log_info(separator)
        
        # Rows for each true intent
        for i in range(min(10, self.num_intents)):
            intent_name = str(i)
            if self.intent_categories and i < len(self.intent_categories):
                intent_name = self.intent_categories[i][:5]  # Truncate to 5 chars
            
            row = f"{intent_name:>10} |"
            for j in range(min(10, self.num_intents)):
                pct = confusion_pct[i, j].item() * 100
                row += f" {pct:>5.1f} |"
            self._log_info(row)
        
        # Show overall accuracy
        correct = confusion.diag().sum()
        total = confusion.sum()
        accuracy = (correct / total).item() * 100
        self._log_info(separator)
        self._log_info(f"Overall Accuracy: {accuracy:.2f}%")
        self._log_info("=" * 40 + "\n")
    
    def _evaluate_intent_performance(self, **kwargs) -> Dict[str, float]:
        """
        Evaluate intent classification performance.
        
        Args:
            **kwargs: Keyword arguments from the hook
            
        Returns:
            Dictionary with performance metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Skip if no confusion matrix
        if "intent_confusion" not in self.accumulated_metrics:
            return metrics
        
        # Get confusion matrix
        confusion = self.accumulated_metrics["intent_confusion"]
        
        # Skip if empty
        if confusion.sum() == 0:
            return metrics
        
        # Calculate metrics
        # Overall accuracy
        correct = confusion.diag().sum()
        total = confusion.sum()
        accuracy = (correct / total).item()
        metrics["accuracy"] = accuracy
        
        # Per-class precision and recall
        precision = confusion.diag() / (confusion.sum(dim=0) + 1e-10)
        recall = confusion.diag() / (confusion.sum(dim=1) + 1e-10)
        
        # Macro-averaged precision and recall
        metrics["macro_precision"] = precision.mean().item()
        metrics["macro_recall"] = recall.mean().item()
        
        # Macro-averaged F1
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        metrics["macro_f1"] = f1.mean().item()
        
        # Weighted-averaged precision and recall
        weights = confusion.sum(dim=1) / total
        metrics["weighted_precision"] = (precision * weights).sum().item()
        metrics["weighted_recall"] = (recall * weights).sum().item()
        metrics["weighted_f1"] = (f1 * weights).sum().item()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model with intent-specific metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Run standard evaluation
        metrics = super().evaluate()
        
        # Add intent-specific metrics
        intent_metrics = self._evaluate_intent_performance()
        metrics.update(intent_metrics)
        
        return metrics
    
    def _log_info(self, message: str) -> None:
        """Log an info message."""
        import logging
        logger = logging.getLogger("qllm.training")
        logger.info(message)
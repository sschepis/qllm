"""
Structured Output Trainer for QLLM.

This module provides a specialized trainer for models that generate structured
outputs (e.g., JSON, XML, YAML), extending the base trainer with functionality
specific to training structured data generation.
"""

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class StructuredOutputTrainer(BaseTrainer):
    """
    Specialized trainer for structured output generation models.
    
    This trainer extends the base trainer with functionality specific to
    training models that generate structured outputs like JSON, XML, or
    YAML, with emphasis on structural correctness and schema compliance.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the structured output trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Extract structured output-specific parameters
        self.output_format = kwargs.pop("output_format", "json")
        self.schema = kwargs.pop("schema", None)
        self.structure_weight = kwargs.pop("structure_weight", 1.0)
        self.content_weight = kwargs.pop("content_weight", 1.0)
        self.validate_output = kwargs.pop("validate_output", True)
        
        # Validator function
        self.validator = kwargs.pop("validator", None)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Add structured output-specific hooks
        self.add_hook("post_step", self._calculate_structure_metrics)
        self.add_hook("post_eval", self._generate_structure_samples)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for structured output training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle structured output-specific format
        if "structure_labels" in batch and "labels" not in batch:
            batch["labels"] = batch["structure_labels"]
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate structure-specific metrics if model doesn't already
        if "loss" in outputs and "structure_accuracy" not in outputs and "logits" in outputs:
            # Standard content loss from model
            content_loss = outputs["loss"]
            
            # Calculate structure loss if the model supports it
            structure_loss = 0.0
            if hasattr(self.model, "calculate_structure_loss"):
                structure_loss = self.model.calculate_structure_loss(
                    outputs["logits"], batch["labels"]
                )
                outputs["structure_loss"] = structure_loss
                
                # Combine losses
                outputs["content_loss"] = content_loss
                outputs["loss"] = (
                    self.content_weight * content_loss + 
                    self.structure_weight * structure_loss
                )
        
        return outputs
    
    def _calculate_structure_metrics(self, **kwargs) -> None:
        """
        Calculate structure-specific metrics after each step.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        metrics = kwargs.get("metrics", {})
        batch = kwargs.get("batch", {})
        
        # Skip if no metrics or batch available
        if not metrics or not batch:
            return
        
        # If model didn't calculate structure metrics, try to calculate them here
        if ("structure_accuracy" not in metrics and 
            "predictions" in metrics and 
            "labels" in batch):
            
            structure_metrics = self._evaluate_structure(
                metrics["predictions"], batch["labels"]
            )
            metrics.update(structure_metrics)
        
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
    
    def _evaluate_structure(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate structural accuracy and other metrics.
        
        Args:
            predictions: Predicted token IDs
            labels: Label token IDs
            
        Returns:
            Dictionary with structure evaluation metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Skip if tokenizer not available
        if not hasattr(self.model, "tokenizer") or self.model.tokenizer is None:
            return metrics
        
        # Decode predictions and labels
        decoded_preds = []
        decoded_labels = []
        
        for pred, label in zip(predictions, labels):
            # Decode prediction
            pred_text = self.model.tokenizer.decode(pred, skip_special_tokens=True)
            decoded_preds.append(pred_text)
            
            # Decode label
            label_text = self.model.tokenizer.decode(label, skip_special_tokens=True)
            decoded_labels.append(label_text)
        
        # Calculate structure metrics
        structure_acc = 0.0
        validation_rate = 0.0
        num_samples = len(decoded_preds)
        
        if num_samples > 0:
            # Validate each prediction
            valid_count = 0
            structure_match_count = 0
            
            for pred_text, label_text in zip(decoded_preds, decoded_labels):
                # Validate prediction format
                pred_valid = self._validate_structure(pred_text)
                label_valid = self._validate_structure(label_text)
                
                # Count valid predictions
                if pred_valid:
                    valid_count += 1
                
                # Compare structures if both are valid
                if pred_valid and label_valid:
                    structure_match = self._compare_structures(pred_text, label_text)
                    if structure_match:
                        structure_match_count += 1
            
            # Calculate metrics
            validation_rate = valid_count / num_samples
            
            # Structure accuracy (among valid samples)
            valid_samples = max(1, valid_count)  # Avoid division by zero
            structure_acc = structure_match_count / valid_samples
        
        # Add metrics
        metrics["structure_accuracy"] = structure_acc
        metrics["validation_rate"] = validation_rate
        
        return metrics
    
    def _validate_structure(self, text: str) -> bool:
        """
        Validate if text contains valid structured output.
        
        Args:
            text: Text to validate
            
        Returns:
            True if the structure is valid
        """
        # Use custom validator if provided
        if self.validator is not None:
            return self.validator(text)
        
        # Format-specific validation
        if self.output_format.lower() == "json":
            return self._validate_json(text)
        elif self.output_format.lower() == "xml":
            return self._validate_xml(text)
        elif self.output_format.lower() == "yaml":
            return self._validate_yaml(text)
        else:
            # Default to json
            return self._validate_json(text)
    
    def _validate_json(self, text: str) -> bool:
        """
        Validate if text contains valid JSON.
        
        Args:
            text: Text to validate
            
        Returns:
            True if the text contains valid JSON
        """
        # Extract JSON portion from text if it contains more than just JSON
        json_pattern = r'(\{.*\}|\[.*\])'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            json_text = match.group(0)
        else:
            json_text = text
        
        # Try to parse as JSON
        try:
            json.loads(json_text)
            return True
        except json.JSONDecodeError:
            return False
    
    def _validate_xml(self, text: str) -> bool:
        """
        Validate if text contains valid XML.
        
        Args:
            text: Text to validate
            
        Returns:
            True if the text contains valid XML
        """
        # Basic XML validation with regex
        # This is a simplified check - a real implementation might use xml.etree.ElementTree
        # Check for opening and closing tags match
        import re
        
        # Extract tags
        open_tags = re.findall(r'<([^/\s>]+)[^>]*>', text)
        close_tags = re.findall(r'</([^>]+)>', text)
        
        # Check if the number of opening and closing tags match
        if len(open_tags) != len(close_tags):
            return False
        
        # Check if tags are properly nested (simplified)
        close_tags.reverse()  # Reverse for proper matching order
        for open_tag, close_tag in zip(open_tags, close_tags):
            if open_tag != close_tag:
                return False
        
        return True
    
    def _validate_yaml(self, text: str) -> bool:
        """
        Validate if text contains valid YAML.
        
        Args:
            text: Text to validate
            
        Returns:
            True if the text contains valid YAML
        """
        # Try to parse as YAML
        try:
            import yaml
            yaml.safe_load(text)
            return True
        except (yaml.YAMLError, ImportError):
            return False
    
    def _compare_structures(self, text1: str, text2: str) -> bool:
        """
        Compare if two structured texts have the same structure.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if the texts have the same structure
        """
        # Format-specific comparison
        if self.output_format.lower() == "json":
            return self._compare_json_structures(text1, text2)
        elif self.output_format.lower() == "xml":
            return self._compare_xml_structures(text1, text2)
        elif self.output_format.lower() == "yaml":
            return self._compare_yaml_structures(text1, text2)
        else:
            # Default to json
            return self._compare_json_structures(text1, text2)
    
    def _compare_json_structures(self, text1: str, text2: str) -> bool:
        """
        Compare if two JSON texts have the same structure.
        
        Args:
            text1: First JSON text
            text2: Second JSON text
            
        Returns:
            True if the texts have the same JSON structure
        """
        # Extract JSON portions
        json_pattern = r'(\{.*\}|\[.*\])'
        match1 = re.search(json_pattern, text1, re.DOTALL)
        match2 = re.search(json_pattern, text2, re.DOTALL)
        
        if match1 and match2:
            json_text1 = match1.group(0)
            json_text2 = match2.group(0)
        else:
            json_text1 = text1
            json_text2 = text2
        
        # Parse JSON
        try:
            obj1 = json.loads(json_text1)
            obj2 = json.loads(json_text2)
            
            # Compare structures
            return self._compare_json_objects(obj1, obj2)
        except json.JSONDecodeError:
            return False
    
    def _compare_json_objects(self, obj1: Any, obj2: Any) -> bool:
        """
        Compare if two JSON objects have the same structure.
        
        Args:
            obj1: First JSON object
            obj2: Second JSON object
            
        Returns:
            True if the objects have the same structure
        """
        # Compare types
        if type(obj1) != type(obj2):
            return False
        
        # Handle dictionaries
        if isinstance(obj1, dict):
            # Check if keys match
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            
            # Recursively compare each value's structure
            for key in obj1:
                if not self._compare_json_objects(obj1[key], obj2[key]):
                    return False
            
            return True
        
        # Handle lists
        elif isinstance(obj1, list):
            # Empty lists match
            if len(obj1) == 0 and len(obj2) == 0:
                return True
            
            # If lengths differ, try to match structures as much as possible
            # by comparing the first few elements
            match_length = min(len(obj1), len(obj2), 1)  # At least one element
            
            # Compare structure of first elements
            for i in range(match_length):
                if not self._compare_json_objects(obj1[i], obj2[i]):
                    return False
            
            return True
        
        # For primitive values, just check type matching
        else:
            return isinstance(obj1, type(obj2))
    
    def _compare_xml_structures(self, text1: str, text2: str) -> bool:
        """
        Compare if two XML texts have the same structure.
        
        Args:
            text1: First XML text
            text2: Second XML text
            
        Returns:
            True if the texts have the same XML structure
        """
        # Extract tags and attributes
        import re
        
        # Extract tags from both texts
        tags1 = re.findall(r'<([^/\s>]+)[^>]*>', text1)
        tags2 = re.findall(r'<([^/\s>]+)[^>]*>', text2)
        
        # Compare tag sequences
        return tags1 == tags2
    
    def _compare_yaml_structures(self, text1: str, text2: str) -> bool:
        """
        Compare if two YAML texts have the same structure.
        
        Args:
            text1: First YAML text
            text2: Second YAML text
            
        Returns:
            True if the texts have the same YAML structure
        """
        # Try to parse YAML
        try:
            import yaml
            obj1 = yaml.safe_load(text1)
            obj2 = yaml.safe_load(text2)
            
            # Use JSON object comparison
            return self._compare_json_objects(obj1, obj2)
        except (yaml.YAMLError, ImportError):
            return False
    
    def _generate_structure_samples(self, **kwargs) -> None:
        """
        Generate structured output samples during evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if not evaluation or model doesn't support generation
        if not hasattr(self.model, "generate") or self.eval_dataloader is None:
            return
        
        # Get a sample batch from the evaluation dataset
        sample_batch = next(iter(self.eval_dataloader))
        sample_batch = self._batch_to_device(sample_batch)
        
        # Try to generate structured outputs
        try:
            # Get input ids
            input_ids = sample_batch.get("input_ids", None)
            if input_ids is None:
                return
            
            # Select a random sample from the batch
            import random
            sample_idx = random.randint(0, input_ids.size(0) - 1)
            sample_input = input_ids[sample_idx:sample_idx+1]
            
            # Generate structured output
            with torch.no_grad():
                generated = self.model.generate(
                    sample_input,
                    max_length=min(512, sample_input.size(1) + 200),
                    do_sample=False,  # Use greedy decoding for structured output
                    num_return_sequences=1
                )
            
            # Get input text and generated structure
            input_text = ""
            generated_structure = ""
            
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                # Decode input text
                input_text = self.model.tokenizer.decode(
                    sample_input[0], skip_special_tokens=True
                )
                
                # Get generated structure
                generated_structure = self.model.tokenizer.decode(
                    generated[0, sample_input.size(1):], skip_special_tokens=True
                )
            
            # Get reference structure if available
            reference_structure = ""
            if "labels" in sample_batch:
                reference_ids = sample_batch["labels"][sample_idx]
                if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                    reference_structure = self.model.tokenizer.decode(
                        reference_ids, skip_special_tokens=True
                    )
            
            # Validate generated structure
            is_valid = self._validate_structure(generated_structure)
            
            # Log the structure sample
            self._log_structure_sample(
                input_text, 
                generated_structure, 
                reference_structure,
                is_valid
            )
        except Exception as e:
            # Log error but don't interrupt training
            self._log_error(f"Error generating structure sample: {e}")
    
    def _log_structure_sample(
        self,
        input_text: str,
        generated_structure: str,
        reference_structure: str = "",
        is_valid: bool = False
    ) -> None:
        """
        Log structured output sample.
        
        Args:
            input_text: Input text
            generated_structure: Generated structured output
            reference_structure: Reference (ground truth) structured output if available
            is_valid: Whether the generated structure is valid
        """
        self._log_info("\n" + "=" * 40)
        self._log_info(f"STRUCTURED OUTPUT GENERATION SAMPLE ({self.output_format.upper()})")
        self._log_info("=" * 40)
        self._log_info(f"INPUT: {input_text}")
        self._log_info("-" * 40)
        self._log_info(f"GENERATED STRUCTURE (VALID: {is_valid}):")
        self._log_info(generated_structure)
        
        if reference_structure:
            self._log_info("-" * 40)
            self._log_info(f"REFERENCE STRUCTURE:")
            self._log_info(reference_structure)
            
            # Compare structures if both are valid
            if is_valid and self._validate_structure(reference_structure):
                structures_match = self._compare_structures(
                    generated_structure, reference_structure
                )
                self._log_info("-" * 40)
                self._log_info(f"STRUCTURES MATCH: {structures_match}")
        
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
"""
Function Call Trainer for QLLM.

This module provides a specialized trainer for models that generate function
calls, extending the base trainer with functionality specific to training
models that can invoke functions with parameters.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from src.training.base_trainer import BaseTrainer


class FunctionCallTrainer(BaseTrainer):
    """
    Specialized trainer for function call generation models.
    
    This trainer extends the base trainer with functionality specific to
    training models that can generate structured function calls, handling
    both the function name and parameter prediction aspects.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the function call trainer.
        
        Args:
            *args: Positional arguments for the base trainer
            **kwargs: Keyword arguments for the base trainer
        """
        # Extract function call-specific parameters
        self.function_schema = kwargs.pop("function_schema", {})
        self.separate_name_params = kwargs.pop("separate_name_params", True)
        self.name_weight = kwargs.pop("name_weight", 1.0)
        self.params_weight = kwargs.pop("params_weight", 1.0)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Add function call-specific hooks
        self.add_hook("post_step", self._calculate_function_call_metrics)
        self.add_hook("post_eval", self._generate_function_call_samples)
    
    def _forward(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass for function call training.
        
        Args:
            batch: Batch of data
            is_training: Whether this is a training forward pass
            
        Returns:
            Dictionary with model outputs
        """
        # Handle function call-specific format
        if self.separate_name_params and "function_name" in batch and "parameters" in batch:
            # Some models may expect a combined function_call object
            batch["function_call"] = {
                "name": batch["function_name"],
                "parameters": batch["parameters"]
            }
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate custom loss if needed and not already calculated by model
        if "loss" not in outputs and "function_name_logits" in outputs and "parameter_logits" in outputs:
            # Calculate function name loss
            name_loss = 0.0
            if "function_name" in batch:
                name_logits = outputs["function_name_logits"]
                name_labels = batch["function_name"]
                name_loss = F.cross_entropy(
                    name_logits.view(-1, name_logits.size(-1)),
                    name_labels.view(-1)
                )
            
            # Calculate parameters loss
            params_loss = 0.0
            if "parameters" in batch:
                param_logits = outputs["parameter_logits"]
                param_labels = batch["parameters"]
                params_loss = F.cross_entropy(
                    param_logits.view(-1, param_logits.size(-1)),
                    param_labels.view(-1),
                    ignore_index=-100  # Ignore padding
                )
            
            # Combine losses with weights
            outputs["name_loss"] = name_loss
            outputs["params_loss"] = params_loss
            outputs["loss"] = self.name_weight * name_loss + self.params_weight * params_loss
        
        return outputs
    
    def _calculate_function_call_metrics(self, **kwargs) -> None:
        """
        Calculate function call-specific metrics after each step.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        metrics = kwargs.get("metrics", {})
        batch = kwargs.get("batch", {})
        
        # Skip if no metrics available
        if not metrics or not batch:
            return
        
        # Calculate function name accuracy if not already calculated
        if ("function_name_accuracy" not in metrics and 
            "function_name_logits" in metrics and 
            "function_name" in batch):
            
            name_logits = metrics["function_name_logits"]
            name_labels = batch["function_name"]
            
            if isinstance(name_logits, torch.Tensor) and isinstance(name_labels, torch.Tensor):
                # Get predicted function names
                name_preds = torch.argmax(name_logits, dim=-1)
                
                # Calculate accuracy
                name_acc = (name_preds == name_labels).float().mean().item()
                metrics["function_name_accuracy"] = name_acc
        
        # Calculate parameter accuracy if possible
        if ("parameter_accuracy" not in metrics and 
            "parameter_logits" in metrics and 
            "parameters" in batch):
            
            param_logits = metrics["parameter_logits"]
            param_labels = batch["parameters"]
            
            if isinstance(param_logits, torch.Tensor) and isinstance(param_labels, torch.Tensor):
                # Get predicted parameters
                param_preds = torch.argmax(param_logits, dim=-1)
                
                # Create mask for non-padding tokens
                mask = (param_labels != -100)
                
                # Calculate accuracy only on non-padding tokens
                if mask.sum() > 0:
                    param_acc = ((param_preds == param_labels) * mask).sum() / mask.sum()
                    metrics["parameter_accuracy"] = param_acc.item()
        
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
    
    def _generate_function_call_samples(self, **kwargs) -> None:
        """
        Generate function call samples during evaluation.
        
        Args:
            **kwargs: Keyword arguments from the hook
        """
        # Skip if not evaluation or model doesn't support generation
        if not hasattr(self.model, "generate_function_call") or self.eval_dataloader is None:
            return
        
        # Get a sample batch from the evaluation dataset
        sample_batch = next(iter(self.eval_dataloader))
        sample_batch = self._batch_to_device(sample_batch)
        
        # Try to generate function calls
        try:
            # Get input ids
            input_ids = sample_batch.get("input_ids", None)
            if input_ids is None:
                return
            
            # Select a random sample from the batch
            import random
            sample_idx = random.randint(0, input_ids.size(0) - 1)
            sample_input = input_ids[sample_idx:sample_idx+1]
            
            # Generate function call
            with torch.no_grad():
                # Use specialized function call generation if available
                if hasattr(self.model, "generate_function_call"):
                    function_call = self.model.generate_function_call(sample_input)
                else:
                    # Fall back to regular generation and try to parse function call
                    generated = self.model.generate(
                        sample_input,
                        max_length=min(512, sample_input.size(1) + 100),
                        do_sample=False  # Use greedy decoding for function calls
                    )
                    
                    # Try to parse function call from generated text
                    function_call = self._parse_function_call_from_text(generated[0].tolist())
            
            # Get reference function call if available
            reference_function = {}
            if "function_name" in sample_batch and "parameters" in sample_batch:
                reference_function = {
                    "name": self._decode_function_name(sample_batch["function_name"][sample_idx]),
                    "parameters": self._decode_parameters(sample_batch["parameters"][sample_idx])
                }
            
            # Get input text
            input_text = ""
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                input_text = self.model.tokenizer.decode(
                    sample_input[0], skip_special_tokens=True
                )
            
            # Log the function call sample
            self._log_function_call_sample(input_text, function_call, reference_function)
        except Exception as e:
            # Log error but don't interrupt training
            self._log_error(f"Error generating function call sample: {e}")
    
    def _parse_function_call_from_text(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        Parse function call from generated token IDs.
        
        Args:
            token_ids: Generated token IDs
            
        Returns:
            Dictionary with function name and parameters
        """
        # Default empty function call
        function_call = {"name": "", "parameters": {}}
        
        # Skip if no tokenizer
        if not hasattr(self.model, "tokenizer") or self.model.tokenizer is None:
            return function_call
        
        # Decode text
        text = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Try to extract function call with regex patterns
        import re
        
        # Pattern for function name and parameters
        pattern = r'([a-zA-Z0-9_]+)\s*\(\s*(.*)\s*\)'
        match = re.search(pattern, text)
        
        if match:
            function_name = match.group(1)
            params_str = match.group(2)
            
            # Try to parse parameters as JSON
            try:
                # Convert single quotes to double quotes for JSON parsing
                params_str = params_str.replace("'", '"')
                parameters = json.loads('{' + params_str + '}')
            except json.JSONDecodeError:
                # Fallback parsing
                parameters = {}
                param_pattern = r'([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([a-zA-Z0-9_]+))'
                param_matches = re.findall(param_pattern, params_str)
                
                for param_match in param_matches:
                    param_name = param_match[0]
                    # Get the first non-empty value from groups 1-3
                    param_value = next((v for v in param_match[1:] if v), None)
                    parameters[param_name] = param_value
            
            function_call = {"name": function_name, "parameters": parameters}
        
        return function_call
    
    def _decode_function_name(self, function_name_id: torch.Tensor) -> str:
        """
        Decode function name from token ID.
        
        Args:
            function_name_id: Function name ID
            
        Returns:
            Decoded function name
        """
        # Check if we have a function schema with names
        if hasattr(self, "function_schema") and self.function_schema and "names" in self.function_schema:
            function_id = function_name_id.item() if isinstance(function_name_id, torch.Tensor) else function_name_id
            return self.function_schema["names"].get(function_id, f"unknown_{function_id}")
        
        # Default to returning the ID
        return str(function_name_id.item() if isinstance(function_name_id, torch.Tensor) else function_name_id)
    
    def _decode_parameters(self, parameter_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Decode parameters from token IDs.
        
        Args:
            parameter_ids: Parameter token IDs
            
        Returns:
            Decoded parameters dictionary
        """
        # Skip if no tokenizer
        if not hasattr(self.model, "tokenizer") or self.model.tokenizer is None:
            return {}
        
        # Decode parameters text
        params_text = self.model.tokenizer.decode(parameter_ids, skip_special_tokens=True)
        
        # Try to parse as JSON
        try:
            # Clean up params text to be valid JSON
            params_text = params_text.strip()
            if not params_text.startswith('{'):
                params_text = '{' + params_text
            if not params_text.endswith('}'):
                params_text = params_text + '}'
            
            return json.loads(params_text)
        except json.JSONDecodeError:
            # Simple key-value parsing fallback
            params = {}
            import re
            
            # Pattern for key-value pairs
            pattern = r'([a-zA-Z0-9_]+)\s*:\s*([^,}]+)'
            matches = re.findall(pattern, params_text)
            
            for key, value in matches:
                params[key.strip()] = value.strip()
            
            return params
    
    def _log_function_call_sample(
        self, 
        input_text: str, 
        function_call: Dict[str, Any],
        reference_function: Dict[str, Any] = None
    ) -> None:
        """
        Log a function call sample.
        
        Args:
            input_text: Input text that triggered the function call
            function_call: Generated function call
            reference_function: Reference (ground truth) function call if available
        """
        self._log_info("\n" + "=" * 40)
        self._log_info("FUNCTION CALL GENERATION SAMPLE")
        self._log_info("=" * 40)
        self._log_info(f"INPUT: {input_text}")
        self._log_info("-" * 40)
        self._log_info(f"GENERATED FUNCTION CALL:")
        self._log_info(f"  Name: {function_call.get('name', '')}")
        self._log_info(f"  Parameters: {json.dumps(function_call.get('parameters', {}), indent=2)}")
        
        if reference_function:
            self._log_info("-" * 40)
            self._log_info(f"REFERENCE FUNCTION CALL:")
            self._log_info(f"  Name: {reference_function.get('name', '')}")
            self._log_info(f"  Parameters: {json.dumps(reference_function.get('parameters', {}), indent=2)}")
        
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
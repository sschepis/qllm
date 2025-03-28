"""
Dialogue model adapter for the enhanced training system.

This module provides an implementation of the ModelAdapter for dialogue models,
handling initialization, forward passes, and loss computation specific to
dialogue-focused training.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import logging

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.training.model_adapters.base_adapter import ModelAdapter


logger = logging.getLogger("quantum_resonance")


class DialogueModelAdapter(ModelAdapter):
    """
    Model adapter implementation for dialogue models.
    
    This adapter handles model initialization, tokenizer setup with dialogue-specific
    tokens, batch processing for conversations, and loss computation for dialogue training.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dialogue model adapter.
        
        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training
            device: Device to use for model execution
            logger: Logger instance
        """
        super().__init__(model_config, training_config, device)
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Dialogue-specific settings
        self.required_tokens = ["<system>", "<user>", "<assistant>"]
        self.system_prompt = getattr(self.training_config, "system_prompt", "You are a helpful assistant.")
        
        # Ensure model config has necessary parameters for dialogue
        self._ensure_dialogue_config()
    
    def _ensure_dialogue_config(self) -> None:
        """Ensure model config has necessary parameters for dialogue."""
        if not hasattr(self.model_config, "use_dialog_aware_attention") or not self.model_config.use_dialog_aware_attention:
            self.logger.info("Setting use_dialog_aware_attention=True for dialogue model")
            setattr(self.model_config, "use_dialog_aware_attention", True)
    
    def create_model(self) -> nn.Module:
        """
        Create and initialize the dialogue model.
        
        Returns:
            Initialized dialogue model instance
        """
        self.logger.info("Initializing dialogue model...")
        
        # Create model instance with dialogue-specific configurations
        model = SemanticResonanceModel(self.model_config)
        model.to(self.device)
        
        # Log model size
        total_params, trainable_params = self.compute_model_size(model)
        self.logger.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        return model
    
    def create_tokenizer(self) -> Any:
        """
        Create and initialize the tokenizer with dialogue-specific tokens.
        
        Returns:
            Initialized tokenizer with dialogue tokens
        """
        # Find tokenizer name from config
        tokenizer_name = None
        
        # Try to get from model_config
        if self.model_config and hasattr(self.model_config, "tokenizer_name"):
            tokenizer_name = self.model_config.tokenizer_name
        
        # Try to get from training_config
        if tokenizer_name is None and self.training_config:
            if hasattr(self.training_config, "model_adapter") and hasattr(self.training_config.model_adapter, "tokenizer_name"):
                tokenizer_name = self.training_config.model_adapter.tokenizer_name
            elif hasattr(self.training_config, "tokenizer_name"):
                tokenizer_name = self.training_config.tokenizer_name
        
        # Default tokenizer if none specified
        if tokenizer_name is None:
            tokenizer_name = "gpt2"
            self.logger.warning(f"No tokenizer name found in config, using default: {tokenizer_name}")
        
        self.logger.info(f"Loading tokenizer for dialogue: {tokenizer_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set default pad token if not set
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Ensure we have the necessary special tokens for dialogue
        special_tokens = {
            "additional_special_tokens": []
        }
        
        if not tokenizer.sep_token:
            special_tokens["sep_token"] = "<sep>"
        
        # Add speaker tokens if not present
        missing_tokens = [token for token in self.required_tokens if token not in tokenizer.get_vocab()]
        
        if missing_tokens:
            self.logger.info(f"Adding missing dialogue tokens to tokenizer: {missing_tokens}")
            special_tokens["additional_special_tokens"].extend(missing_tokens)
        
        # Resize tokenizer if needed
        if special_tokens["additional_special_tokens"] or "sep_token" in special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            
            # Update vocab size in model config
            self.model_config.vocab_size = len(tokenizer)
            
            # Resize model's token embeddings
            if self.model is not None:
                self.model = self.resize_token_embeddings(self.model, tokenizer)
        
        return tokenizer
    
    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare a dialogue batch for model processing.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Batch prepared for model input
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            prepared_batch = batch
        else:
            # Handle tuple batch (typical for dialogue datasets)
            try:
                if len(batch) >= 3:
                    input_ids, attention_mask, labels = batch[:3]
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                elif len(batch) == 2:
                    input_ids, attention_mask = batch
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    }
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            except Exception as e:
                raise ValueError(f"Error unpacking dialogue batch: {e}")
        
        # Ensure required keys are present
        if "input_ids" not in prepared_batch or "attention_mask" not in prepared_batch:
            raise ValueError("Batch is missing required keys: input_ids or attention_mask")
        
        # If no labels are present, add dummy labels using input_ids
        if "labels" not in prepared_batch:
            self.logger.warning("No labels found in dialogue batch. Using input_ids as labels.")
            prepared_batch["labels"] = prepared_batch["input_ids"].clone()
            
        # Ensure input_ids and labels are LongTensor (required for embedding layers)
        for key in ["input_ids", "labels"]:
            if key in prepared_batch and torch.is_tensor(prepared_batch[key]):
                if prepared_batch[key].dtype != torch.long:
                    self.logger.warning(f"Converting {key} from {prepared_batch[key].dtype} to torch.long")
                    prepared_batch[key] = prepared_batch[key].long()
        
        # Ensure attention_mask has correct dimensions for attention layers
        if "attention_mask" in prepared_batch and torch.is_tensor(prepared_batch["attention_mask"]):
            # Check if attention_mask is 1D but needs to be 2D
            if prepared_batch["attention_mask"].dim() == 1:
                self.logger.warning("Reshaping 1D attention_mask to 2D")
                batch_size = prepared_batch["input_ids"].size(0)
                seq_len = prepared_batch["input_ids"].size(1)
                
                # Try to reshape based on input_ids dimensions
                if prepared_batch["attention_mask"].size(0) == batch_size * seq_len:
                    # Reshape to [batch_size, seq_len]
                    prepared_batch["attention_mask"] = prepared_batch["attention_mask"].view(batch_size, seq_len)
                else:
                    # Create a new attention mask that's all 1s
                    self.logger.warning("Cannot reshape attention_mask, creating new one")
                    prepared_batch["attention_mask"] = torch.ones_like(prepared_batch["input_ids"]).bool()
        
        # NaN prevention: Check if all labels are -100
        if torch.is_tensor(prepared_batch["labels"]) and (prepared_batch["labels"] == -100).all():
            self.logger.warning("All labels are -100! This may lead to NaN loss.")
        
        # Move batch to device
        return self.move_to_device(prepared_batch)
    
    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Perform forward pass with the dialogue model.
        
        Args:
            model: Model instance
            batch: Prepared dialogue input batch
            return_dict: Whether to return outputs as a dictionary
            
        Returns:
            Model outputs (as dict if return_dict=True, otherwise as tuple)
        """
        # Dialogue-specific forward pass additions can be added here
        
        # Forward pass through model
        try:
            # Handle attention mask dimensionality issues
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
                
                # Fix dimensions if needed
                if attention_mask.dim() == 1:
                    bs = batch['input_ids'].size(0)
                    seq_len = batch['input_ids'].size(1)
                    
                    # Reshape or create proper attention mask
                    if attention_mask.size(0) == bs * seq_len:
                        batch['attention_mask'] = attention_mask.view(bs, seq_len)
                    else:
                        # Create new mask of correct size - use float not bool
                        batch['attention_mask'] = torch.ones_like(batch['input_ids'], dtype=torch.float)
                
                # Convert boolean tensor to float tensor for subtraction operations
                if batch['attention_mask'].dtype == torch.bool:
                    self.logger.warning("Converting attention_mask from bool to float")
                    batch['attention_mask'] = batch['attention_mask'].float()
                
                # Check if key_padding_mask is needed instead of attention_mask
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                    # Add key_padding_mask for transformer models
                    if 'key_padding_mask' not in batch:
                        # Make sure this is a float tensor too
                        batch['key_padding_mask'] = batch['attention_mask'].float()
            
            # Forward pass through model
            outputs = model(**batch, return_dict=return_dict)
            
            # DialogueModelAdapter.forward is expected to return (loss, outputs)
            # not just the outputs
            loss = None
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            
            # Return the correct format - some code expects a tuple (loss, outputs)
            if return_dict:
                return outputs
            else:
                return loss, outputs
                
        except Exception as e:
            # Log error and try to recover
            self.logger.error(f"Error in dialogue model forward pass: {e}")
            
            # Create dummy outputs as a fallback
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            
            if return_dict:
                return {'loss': dummy_loss, 'logits': dummy_loss.view(1, 1)}
            else:
                return (dummy_loss, {'logits': dummy_loss.view(1, 1)})
    
    def compute_loss(
        self,
        outputs: Union[Dict[str, Any], Tuple],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss from dialogue model outputs and batch.
        
        Args:
            outputs: Model outputs
            batch: Input batch that generated the outputs
            
        Returns:
            Loss tensor
        """
        # Extract loss from outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # Assume first element is loss in tuple outputs
            loss = outputs[0]
        else:
            raise ValueError("Unable to extract loss from dialogue model outputs")
        
        # Dialogue-specific loss handling
        # Check for NaN/Inf values
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            self.logger.warning("Detected NaN/Inf in dialogue model loss calculation")
            
            # Optional: more robust NaN handling for dialogue models if needed
            # For now, just return the loss as is
            
        return loss
    
    def format_dialogue_prompt(
        self,
        conversation: List[Dict[str, str]],
        tokenizer: Optional[Any] = None
    ) -> str:
        """
        Format a conversation into a dialogue prompt.
        
        Args:
            conversation: List of conversation turns
                Each turn is a dict with 'role' and 'content' keys
            tokenizer: Tokenizer to use (uses self.tokenizer if None)
            
        Returns:
            Formatted dialogue prompt
        """
        tokenizer = tokenizer or self.get_tokenizer()
        
        # Start with system prompt
        formatted = f"<system>{self.system_prompt}</system>"
        
        # Add each turn
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                formatted += f"\n<user>{content}</user>"
            elif role == "assistant":
                formatted += f"\n<assistant>{content}</assistant>"
            elif role == "system":
                # Skip additional system prompts for now
                pass
            else:
                self.logger.warning(f"Unknown dialogue role: {role}")
        
        return formatted
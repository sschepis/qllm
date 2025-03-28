"""
Finetuning strategy for the enhanced training system.

This module implements a specialized training strategy for finetuning
pre-trained models with techniques like layer-wise learning rate decay.
"""

import logging
import math
from typing import Dict, Any, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.strategies.standard_strategy import StandardTrainingStrategy


class FinetuningStrategy(StandardTrainingStrategy):
    """
    Specialized training strategy for finetuning pre-trained models.
    
    This strategy extends the standard training strategy with specific
    techniques for finetuning, such as layer-wise learning rate decay,
    discriminative learning rates, and early stopping.
    """
    
    def __init__(
        self,
        config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the finetuning strategy.
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        
        # Extract finetuning-specific parameters
        self.lr_decay_rate = getattr(config, "lr_decay_rate", 0.8)
        self.use_layerwise_lr = getattr(config, "use_layerwise_lr", True)
        self.freeze_layers = getattr(config, "freeze_layers", None)
        self.freeze_embeddings = getattr(config, "freeze_embeddings", False)
        self.early_stopping_patience = getattr(config, "early_stopping_patience", 3)
        self.early_stopping_threshold = getattr(config, "early_stopping_threshold", 0.01)
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        **kwargs
    ) -> optim.Optimizer:
        """
        Create an optimizer with layer-wise learning rate decay for finetuning.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            **kwargs: Additional optimizer parameters
            
        Returns:
            Initialized optimizer
        """
        # Get parameters from kwargs or default to config
        lr = kwargs.get("lr", self.learning_rate)
        weight_decay = kwargs.get("weight_decay", self.weight_decay)
        
        # Handle layer freezing if specified
        if self.freeze_layers is not None:
            self._freeze_layers(model, self.freeze_layers)
        
        # Handle embedding freezing if specified
        if self.freeze_embeddings:
            self._freeze_embeddings(model)
        
        # If using layer-wise learning rate decay
        if self.use_layerwise_lr:
            # Get parameter groups with decayed learning rates
            parameter_groups = self._get_layerwise_parameters(model, lr, weight_decay)
            self.logger.info(f"Created {len(parameter_groups)} parameter groups with layerwise learning rate decay")
            
            # Create optimizer with parameter groups
            if optimizer_type.lower() == "adamw":
                return optim.AdamW(parameter_groups)
            elif optimizer_type.lower() == "adam":
                return optim.Adam(parameter_groups)
            elif optimizer_type.lower() == "sgd":
                momentum = kwargs.get("momentum", 0.9)
                return optim.SGD(parameter_groups, momentum=momentum)
            else:
                self.logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
                return optim.AdamW(parameter_groups)
        else:
            # Use standard optimizer without layer-wise decay
            return super().create_optimizer(model, optimizer_type, **kwargs)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        # If no early stopping
        if self.early_stopping_patience <= 0:
            return False
        
        # Check if this is the best loss so far
        if val_loss < self.best_val_loss * (1 - self.early_stopping_threshold):
            # New best loss
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            # No improvement
            self.patience_counter += 1
            
            # Check if patience is exhausted
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                return True
            
            self.logger.info(f"No improvement for {self.patience_counter} epochs, best loss: {self.best_val_loss:.6f}")
            return False
    
    def _get_layerwise_parameters(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with layer-wise learning rate decay.
        
        Args:
            model: Model to get parameters for
            lr: Base learning rate
            weight_decay: Weight decay
            
        Returns:
            List of parameter group dictionaries
        """
        parameter_groups = []
        
        # Check if model is a transformers model
        is_transformers_model = self._is_transformers_model(model)
        
        if is_transformers_model:
            # Handle transformers models
            return self._get_transformers_layerwise_parameters(model, lr, weight_decay)
        
        # Try to identify model structure
        layers = self._identify_model_layers(model)
        
        if layers:
            # Create parameter groups for identified layers
            num_layers = len(layers)
            
            for layer_idx, layer in enumerate(layers):
                # Calculate decayed learning rate
                layer_lr = lr * (self.lr_decay_rate ** (num_layers - layer_idx - 1))
                
                # Create parameter group
                parameter_groups.append({
                    'params': layer.parameters(),
                    'lr': layer_lr,
                    'weight_decay': weight_decay
                })
                
                self.logger.debug(f"Layer {layer_idx}: lr={layer_lr:.2e}")
        else:
            # Fallback to simple decay by parameter name pattern
            return self._get_parameters_by_name_pattern(model, lr, weight_decay)
        
        return parameter_groups
    
    def _get_transformers_layerwise_parameters(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for transformers models with layer-wise decay.
        
        Args:
            model: Transformers model
            lr: Base learning rate
            weight_decay: Weight decay
            
        Returns:
            List of parameter group dictionaries
        """
        parameter_groups = []
        
        # Collect parameters by layer depth
        names_to_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                names_to_params[name] = param
        
        # Embeddings typically in transformers models
        emb_params = []
        # Collect embedding parameters
        for name, param in names_to_params.items():
            if 'embed' in name or 'embedding' in name or 'wte' in name or 'wpe' in name:
                emb_params.append(param)
                names_to_params.pop(name)
        
        if emb_params:
            # Lowest learning rate for embeddings
            emb_lr = lr * (self.lr_decay_rate ** 12)  # Assuming 12 layers
            parameter_groups.append({
                'params': emb_params,
                'lr': emb_lr,
                'weight_decay': weight_decay
            })
            self.logger.debug(f"Embeddings: lr={emb_lr:.2e}")
        
        # Collect layer parameters (assuming layers are numbered)
        layer_params = {}
        remaining_params = []
        
        for name, param in names_to_params.items():
            found_layer = False
            
            # Try to extract layer number from name
            for pattern in [r'layer\.(\d+)', r'layers\.(\d+)', r'h\.(\d+)', r'encoder\.(\d+)', r'decoder\.(\d+)']:
                import re
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = []
                    layer_params[layer_idx].append(param)
                    found_layer = True
                    break
            
            if not found_layer:
                remaining_params.append(param)
        
        # Create parameter groups for each layer
        if layer_params:
            num_layers = max(layer_params.keys()) + 1
            
            for layer_idx in sorted(layer_params.keys()):
                # Calculate decayed learning rate
                layer_lr = lr * (self.lr_decay_rate ** (num_layers - layer_idx - 1))
                
                # Create parameter group
                parameter_groups.append({
                    'params': layer_params[layer_idx],
                    'lr': layer_lr,
                    'weight_decay': weight_decay
                })
                
                self.logger.debug(f"Layer {layer_idx}: lr={layer_lr:.2e}")
        
        # Add remaining parameters with base learning rate
        if remaining_params:
            parameter_groups.append({
                'params': remaining_params,
                'lr': lr,
                'weight_decay': weight_decay
            })
            self.logger.debug(f"Remaining params: lr={lr:.2e}")
        
        return parameter_groups
    
    def _get_parameters_by_name_pattern(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups by name pattern.
        
        Args:
            model: Model to get parameters for
            lr: Base learning rate
            weight_decay: Weight decay
            
        Returns:
            List of parameter group dictionaries
        """
        # Define parameter groups with heuristic patterns
        groups = {
            'embedding': {'params': [], 'lr': lr * (self.lr_decay_rate ** 6)},
            'early_layers': {'params': [], 'lr': lr * (self.lr_decay_rate ** 4)},
            'middle_layers': {'params': [], 'lr': lr * (self.lr_decay_rate ** 2)},
            'late_layers': {'params': [], 'lr': lr},
            'other': {'params': [], 'lr': lr}
        }
        
        # Add weight decay to all groups
        for group in groups.values():
            group['weight_decay'] = weight_decay
        
        # Assign parameters to groups based on name patterns
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'embed' in name or 'embedding' in name:
                groups['embedding']['params'].append(param)
            elif any(p in name for p in ['layer.0', 'layer.1', 'layer.2', 'blocks.0', 'blocks.1', 'blocks.2']):
                groups['early_layers']['params'].append(param)
            elif any(p in name for p in ['layer.3', 'layer.4', 'layer.5', 'blocks.3', 'blocks.4', 'blocks.5']):
                groups['middle_layers']['params'].append(param)
            elif any(p in name for p in ['layer.6', 'layer.7', 'layer.8', 'blocks.6', 'blocks.7', 'blocks.8']):
                groups['late_layers']['params'].append(param)
            else:
                groups['other']['params'].append(param)
        
        # Filter out empty groups
        parameter_groups = [group for group in groups.values() if group['params']]
        
        # Log parameter groups
        for name, group in groups.items():
            if group['params']:
                self.logger.debug(f"{name}: {len(group['params'])} params, lr={group['lr']:.2e}")
        
        return parameter_groups
    
    def _identify_model_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Identify model layers for layerwise learning rate decay.
        
        Args:
            model: Model to identify layers in
            
        Returns:
            List of identified layer modules
        """
        layers = []
        
        # Try to find transformer layers or blocks
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            # Transformer encoder layers
            layers = list(model.encoder.layers)
            self.logger.debug(f"Found {len(layers)} transformer encoder layers")
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            # Transformer decoder layers
            layers = list(model.decoder.layers)
            self.logger.debug(f"Found {len(layers)} transformer decoder layers")
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            # GPT-style transformer layers
            layers = list(model.transformer.layers)
            self.logger.debug(f"Found {len(layers)} transformer layers")
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Hugging Face transformer layers
            layers = list(model.transformer.h)
            self.logger.debug(f"Found {len(layers)} Hugging Face transformer layers")
        elif hasattr(model, 'blocks'):
            # Vision transformer blocks
            layers = list(model.blocks)
            self.logger.debug(f"Found {len(layers)} vision transformer blocks")
        elif hasattr(model, 'layers'):
            # Generic layers
            layers = list(model.layers)
            self.logger.debug(f"Found {len(layers)} generic layers")
        
        return layers
    
    def _is_transformers_model(self, model: nn.Module) -> bool:
        """
        Check if the model is from the Hugging Face transformers library.
        
        Args:
            model: Model to check
            
        Returns:
            True if the model is from transformers, False otherwise
        """
        try:
            from transformers import PreTrainedModel
            return isinstance(model, PreTrainedModel)
        except ImportError:
            # If transformers is not available, check the module name
            return 'transformers' in str(type(model).__module__)
    
    def _freeze_layers(self, model: nn.Module, num_layers: Union[int, List[int]]) -> None:
        """
        Freeze specified layers of the model.
        
        Args:
            model: Model to freeze layers in
            num_layers: Number of layers to freeze (from bottom) or list of layer indices
        """
        layers = self._identify_model_layers(model)
        
        if not layers:
            self.logger.warning("Could not identify layers to freeze")
            return
        
        # If num_layers is an integer, freeze that many layers from the bottom
        if isinstance(num_layers, int):
            layers_to_freeze = layers[:num_layers]
            self.logger.info(f"Freezing {num_layers} layers from the bottom")
        else:
            # If num_layers is a list, freeze those specific layers
            layers_to_freeze = [layers[i] for i in num_layers if i < len(layers)]
            self.logger.info(f"Freezing layers at indices: {num_layers}")
        
        # Freeze parameters in selected layers
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _freeze_embeddings(self, model: nn.Module) -> None:
        """
        Freeze embedding layers of the model.
        
        Args:
            model: Model to freeze embeddings in
        """
        # Try to find embedding layers
        embedding_found = False
        
        # Check common embedding layer names
        for name, module in model.named_modules():
            if any(embed_name in name.lower() for embed_name in ['embed', 'embedding', 'wte', 'wpe']):
                for param in module.parameters():
                    param.requires_grad = False
                embedding_found = True
                self.logger.info(f"Froze embedding parameters in {name}")
        
        if not embedding_found:
            self.logger.warning("Could not identify embedding layers to freeze")
"""
Base Extension Module.

This module defines the abstract base class for all extensions in the
Semantic Resonance Language Model.
"""

import abc
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import torch.nn as nn


class BaseExtension(nn.Module, abc.ABC):
    """
    Abstract base class for all model extensions.
    
    All extensions must inherit from this class and implement its abstract methods.
    The base extension provides a common interface for all extension types and
    handles integration with the main model.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the extension.
        
        Args:
            name (str): Unique name for this extension instance
            config (Dict[str, Any]): Configuration dictionary for the extension
        """
        super().__init__()
        self.name = name
        self.config = config
        self.initialized = False
    
    @abc.abstractmethod
    def initialize(self, model: nn.Module) -> None:
        """
        Initialize the extension with the main model.
        
        This method is called once when the extension is attached to the model.
        Implementations should set up any necessary hooks or connections to the main model.
        
        Args:
            model (nn.Module): The main model instance
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, 
                x: torch.Tensor,
                model_outputs: Optional[Dict[str, Any]] = None, 
                extension_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the extension.
        
        Args:
            x (torch.Tensor): Input tensor
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            extension_outputs (Dict[str, Any], optional): Outputs from other extensions
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and extension metadata
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_extension_type(self) -> str:
        """
        Get the type of this extension.
        
        Returns:
            str: Extension type (e.g., "multimodal", "memory", "quantum")
        """
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the extension configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config
    
    def get_name(self) -> str:
        """
        Get the extension name.
        
        Returns:
            str: Extension name
        """
        return self.name
    
    def is_initialized(self) -> bool:
        """
        Check if the extension is initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.initialized
    
    def register_forward_hook(self, model: nn.Module, layer_name: str, hook_fn) -> None:
        """
        Register a forward hook on a specific layer of the model.
        
        Args:
            model (nn.Module): Model to register hook on
            layer_name (str): Name of the layer
            hook_fn: Hook function
        """
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook_fn)
                return
        
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    def get_model_parameter(self, model: nn.Module, param_name: str) -> nn.Parameter:
        """
        Get a parameter from the model by name.
        
        Args:
            model (nn.Module): Model to get parameter from
            param_name (str): Name of the parameter
            
        Returns:
            nn.Parameter: The requested parameter
        """
        for name, param in model.named_parameters():
            if name == param_name:
                return param
        
        raise ValueError(f"Parameter '{param_name}' not found in model")
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of this extension.
        
        Returns:
            int: Output dimension
        """
        return self.config.get("output_dim", 0)
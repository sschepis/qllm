"""
Extension Manager Module.

This module provides a manager class for handling model extensions in the
Semantic Resonance Language Model.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Type, Union
from collections import OrderedDict

import torch
import torch.nn as nn

from .base_extension import BaseExtension
from .extension_config import ExtensionConfig


logger = logging.getLogger(__name__)


class ExtensionManager:
    """
    Manager for model extensions.
    
    The ExtensionManager handles registration, initialization, and execution
    of extensions for the Semantic Resonance Language Model. It ensures proper
    ordering and data flow between extensions and the main model.
    """
    
    def __init__(self, config: ExtensionConfig, parent_model: Optional[nn.Module] = None):
        """
        Initialize the extension manager.
        
        Args:
            config (ExtensionConfig): Configuration for all extensions
            parent_model (nn.Module, optional): Parent model to attach extensions to
        """
        self.config = config
        self.parent_model = parent_model
        
        # Dictionary of registered extensions, grouped by type
        self.extensions: Dict[str, OrderedDict[str, BaseExtension]] = {
            "multimodal": OrderedDict(),
            "memory": OrderedDict(),
            "quantum": OrderedDict(),
            "other": OrderedDict()
        }
        
        # Extension execution order (can be reconfigured)
        self.extension_order: List[Tuple[str, str]] = []  # List of (type, name) tuples
        
        # Tracking enabled/disabled extensions
        self.enabled_extensions: Dict[str, bool] = {}
        
        # Hooks for pre/post processing
        self.pre_extension_hooks: List[callable] = []
        self.post_extension_hooks: List[callable] = []
        
        logger.info(f"Initialized ExtensionManager with configuration: {config}")
    
    def register_extension(self, extension: BaseExtension) -> None:
        """
        Register a new extension with the manager.
        
        Args:
            extension (BaseExtension): Extension to register
        """
        ext_type = extension.get_extension_type()
        ext_name = extension.get_name()
        
        if ext_type not in self.extensions:
            logger.warning(f"Unknown extension type '{ext_type}', registering as 'other'")
            ext_type = "other"
        
        if ext_name in self.extensions[ext_type]:
            logger.warning(f"Extension '{ext_name}' of type '{ext_type}' already registered, replacing")
        
        # Store the extension
        self.extensions[ext_type][ext_name] = extension
        
        # Add to execution order if not already present
        if (ext_type, ext_name) not in self.extension_order:
            self.extension_order.append((ext_type, ext_name))
        
        # Initialize extension if parent model is available
        if self.parent_model is not None and not extension.is_initialized():
            extension.initialize(self.parent_model)
            extension.initialized = True
        
        # Mark as enabled by default
        self.enabled_extensions[f"{ext_type}.{ext_name}"] = True
        
        logger.info(f"Registered extension '{ext_name}' of type '{ext_type}'")
    
    def unregister_extension(self, ext_type: str, ext_name: str) -> None:
        """
        Unregister an extension from the manager.
        
        Args:
            ext_type (str): Type of the extension
            ext_name (str): Name of the extension
        """
        if ext_type in self.extensions and ext_name in self.extensions[ext_type]:
            # Remove from extensions dictionary
            del self.extensions[ext_type][ext_name]
            
            # Remove from execution order
            if (ext_type, ext_name) in self.extension_order:
                self.extension_order.remove((ext_type, ext_name))
            
            # Remove from enabled extensions
            if f"{ext_type}.{ext_name}" in self.enabled_extensions:
                del self.enabled_extensions[f"{ext_type}.{ext_name}"]
            
            logger.info(f"Unregistered extension '{ext_name}' of type '{ext_type}'")
        else:
            logger.warning(f"Extension '{ext_name}' of type '{ext_type}' not found, cannot unregister")
    
    def get_extension(self, ext_type: str, ext_name: str) -> Optional[BaseExtension]:
        """
        Get a registered extension by type and name.
        
        Args:
            ext_type (str): Type of the extension
            ext_name (str): Name of the extension
            
        Returns:
            Optional[BaseExtension]: The extension if found, None otherwise
        """
        if ext_type in self.extensions and ext_name in self.extensions[ext_type]:
            return self.extensions[ext_type][ext_name]
        return None
    
    def enable_extension(self, ext_type: str, ext_name: str) -> None:
        """
        Enable a specific extension.
        
        Args:
            ext_type (str): Type of the extension
            ext_name (str): Name of the extension
        """
        key = f"{ext_type}.{ext_name}"
        self.enabled_extensions[key] = True
        logger.info(f"Enabled extension '{ext_name}' of type '{ext_type}'")
    
    def disable_extension(self, ext_type: str, ext_name: str) -> None:
        """
        Disable a specific extension.
        
        Args:
            ext_type (str): Type of the extension
            ext_name (str): Name of the extension
        """
        key = f"{ext_type}.{ext_name}"
        self.enabled_extensions[key] = False
        logger.info(f"Disabled extension '{ext_name}' of type '{ext_type}'")
    
    def is_extension_enabled(self, ext_type: str, ext_name: str) -> bool:
        """
        Check if a specific extension is enabled.
        
        Args:
            ext_type (str): Type of the extension
            ext_name (str): Name of the extension
            
        Returns:
            bool: True if the extension is enabled, False otherwise
        """
        key = f"{ext_type}.{ext_name}"
        return self.enabled_extensions.get(key, False)
    
    def set_extension_order(self, order: List[Tuple[str, str]]) -> None:
        """
        Set the execution order for extensions.
        
        Args:
            order (List[Tuple[str, str]]): List of (type, name) tuples defining execution order
        """
        # Validate that all extensions exist
        for ext_type, ext_name in order:
            if ext_type not in self.extensions or ext_name not in self.extensions[ext_type]:
                raise ValueError(f"Extension '{ext_name}' of type '{ext_type}' not registered")
        
        self.extension_order = order
        logger.info(f"Set extension execution order: {order}")
    
    def register_pre_extension_hook(self, hook: callable) -> None:
        """
        Register a hook to run before any extensions are executed.
        
        Args:
            hook (callable): Hook function to register
        """
        self.pre_extension_hooks.append(hook)
    
    def register_post_extension_hook(self, hook: callable) -> None:
        """
        Register a hook to run after all extensions are executed.
        
        Args:
            hook (callable): Hook function to register
        """
        self.post_extension_hooks.append(hook)
    
    def attach_to_model(self, model: nn.Module) -> None:
        """
        Attach the extension manager to a model and initialize all extensions.
        
        Args:
            model (nn.Module): Model to attach to
        """
        self.parent_model = model
        
        # Initialize all extensions
        for ext_type, extensions in self.extensions.items():
            for ext_name, extension in extensions.items():
                if not extension.is_initialized():
                    extension.initialize(model)
                    extension.initialized = True
        
        logger.info(f"Attached extension manager to model and initialized all extensions")
    
    def process_extensions(self, 
                          x: torch.Tensor, 
                          model_outputs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through all enabled extensions in the specified order.
        
        Args:
            x (torch.Tensor): Input tensor
            model_outputs (Dict[str, Any], optional): Outputs from the main model
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Modified tensor and combined extension outputs
        """
        if not self.config.extensions_enabled:
            return x, {}
        
        # Initialize extension outputs
        extension_outputs = {}
        
        # Run pre-extension hooks
        for hook in self.pre_extension_hooks:
            x = hook(x, model_outputs, extension_outputs)
        
        # Process through each extension in order
        for ext_type, ext_name in self.extension_order:
            # Skip disabled extensions
            if not self.is_extension_enabled(ext_type, ext_name):
                continue
            
            # Get the extension
            extension = self.get_extension(ext_type, ext_name)
            if extension is None:
                continue
            
            # Process through the extension
            x, ext_output = extension.forward(x, model_outputs, extension_outputs)
            
            # Store the extension output
            extension_outputs[f"{ext_type}.{ext_name}"] = ext_output
        
        # Run post-extension hooks
        for hook in self.post_extension_hooks:
            x = hook(x, model_outputs, extension_outputs)
        
        return x, extension_outputs
    
    def get_all_extensions(self) -> List[BaseExtension]:
        """
        Get a list of all registered extensions.
        
        Returns:
            List[BaseExtension]: All registered extensions
        """
        all_extensions = []
        for ext_type, extensions in self.extensions.items():
            all_extensions.extend(list(extensions.values()))
        return all_extensions
    
    def get_extensions_by_type(self, ext_type: str) -> List[BaseExtension]:
        """
        Get all extensions of a specific type.
        
        Args:
            ext_type (str): Type of extensions to get
            
        Returns:
            List[BaseExtension]: All extensions of the specified type
        """
        if ext_type in self.extensions:
            return list(self.extensions[ext_type].values())
        return []
    
    def get_config(self) -> ExtensionConfig:
        """
        Get the extension configuration.
        
        Returns:
            ExtensionConfig: Extension configuration
        """
        return self.config
    
    def save_state(self, path: str) -> None:
        """
        Save the state of all extensions.
        
        Args:
            path (str): Path to save state to
        """
        states = {}
        for ext_type, extensions in self.extensions.items():
            type_states = {}
            for ext_name, extension in extensions.items():
                if hasattr(extension, 'state_dict'):
                    type_states[ext_name] = extension.state_dict()
            states[ext_type] = type_states
        
        torch.save(states, path)
        logger.info(f"Saved extension states to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load the state of all extensions.
        
        Args:
            path (str): Path to load state from
        """
        states = torch.load(path)
        
        for ext_type, type_states in states.items():
            if ext_type in self.extensions:
                for ext_name, state in type_states.items():
                    if ext_name in self.extensions[ext_type]:
                        extension = self.extensions[ext_type][ext_name]
                        if hasattr(extension, 'load_state_dict'):
                            extension.load_state_dict(state)
        
        logger.info(f"Loaded extension states from {path}")
"""
Extension manager for the enhanced training system.

This module provides a manager for model extensions during training,
coordinating extension initialization, hook registration, and integration
with the training process.
"""

from typing import Dict, Any, List, Callable, Optional, Union, Set, Type
import logging
import importlib
import inspect

import torch
import torch.nn as nn

from src.config.training_config import TrainingConfig
from src.model.extensions.base_extension import BaseExtension
from src.training.extensions.extension_hooks import ExtensionHooks


class ExtensionManager:
    """
    Manager for model extensions during training.
    
    This class coordinates the initialization, configuration, and integration
    of model extensions with the training process, providing a unified
    interface for the trainer to interact with extensions.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the extension manager.
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Initialize extension hooks
        self.hooks = ExtensionHooks(logger=self.logger)
        
        # Tracking for extensions
        self.extensions: Dict[str, BaseExtension] = {}
        self.extension_configs: Dict[str, Dict[str, Any]] = {}
        
        # Get enabled extensions from config
        self.enabled_extensions = getattr(config, "enabled_extensions", [])
        self.extension_configs = getattr(config, "extension_configs", {})
        
        # Load extension registry
        self.extension_registry = self._load_extension_registry()
        
        # Initialize extensions
        self._initialize_extensions()
    
    def _load_extension_registry(self) -> Dict[str, Type[BaseExtension]]:
        """
        Load the extension registry from the extensions package.
        
        Returns:
            Dictionary mapping extension names to extension classes
        """
        registry = {}
        
        try:
            # Try to import the extension registry from the main extensions package
            try:
                from src.model.extensions import EXTENSION_REGISTRY
                registry.update(EXTENSION_REGISTRY)
            except (ImportError, AttributeError):
                self.logger.warning("Main extension registry not found, searching for extensions individually")
            
            # Dynamically search for extension modules
            extension_packages = [
                "src.model.extensions.quantum",
                "src.model.extensions.memory",
                "src.model.extensions.multimodal"
            ]
            
            for package_name in extension_packages:
                try:
                    package = importlib.import_module(package_name)
                    
                    # Look for extension classes in the package
                    for name, obj in inspect.getmembers(package):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseExtension) and 
                            obj is not BaseExtension):
                            
                            # Add to registry using snake_case name
                            import re
                            extension_name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
                            if extension_name.endswith('_extension'):
                                extension_name = extension_name[:-10]  # Remove _extension suffix
                            
                            registry[extension_name] = obj
                            self.logger.debug(f"Found extension: {extension_name} -> {obj.__name__}")
                except ImportError:
                    self.logger.debug(f"Extension package not found: {package_name}")
                except Exception as e:
                    self.logger.warning(f"Error loading extensions from {package_name}: {e}")
            
            self.logger.info(f"Loaded extension registry with {len(registry)} extensions: {list(registry.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error loading extension registry: {e}")
        
        return registry
    
    def _initialize_extensions(self) -> None:
        """Initialize enabled extensions based on configuration."""
        for extension_name in self.enabled_extensions:
            self._initialize_extension(extension_name)
    
    def _initialize_extension(self, extension_name: str) -> None:
        """
        Initialize a specific extension.
        
        Args:
            extension_name: Name of the extension to initialize
        """
        if extension_name in self.extensions:
            # Already initialized
            return
        
        # Check if extension class is available
        if extension_name not in self.extension_registry:
            self.logger.warning(f"Extension '{extension_name}' not found in registry. Available extensions: {list(self.extension_registry.keys())}")
            return
        
        try:
            # Get extension class
            extension_class = self.extension_registry[extension_name]
            
            # Get extension config
            extension_config = self.extension_configs.get(extension_name, {})
            
            # Import specific config class if needed
            config_class = None
            try:
                config_module_name = f"src.model.extensions.{extension_name}.{extension_name}_config"
                config_class_name = f"{extension_name.capitalize()}Config"
                
                # Try to import specific config class
                config_module = importlib.import_module(config_module_name)
                config_class = getattr(config_module, config_class_name)
            except (ImportError, AttributeError):
                self.logger.debug(f"No specific config class found for {extension_name}")
            
            # Create extension instance
            if config_class:
                # Create with typed config
                config_instance = config_class(**extension_config)
                extension_instance = extension_class(config_instance)
            else:
                # Create with dict config
                extension_instance = extension_class(extension_config)
            
            # Store extension
            self.extensions[extension_name] = extension_instance
            
            # Register extension hooks
            self.hooks.register_extension_hooks(extension_instance, extension_name)
            
            self.logger.info(f"Initialized extension: {extension_name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing extension '{extension_name}': {e}")
    
    def register_with_model(self, model: nn.Module) -> nn.Module:
        """
        Register extensions with a model.
        
        Args:
            model: Model instance
            
        Returns:
            Model with extensions integrated
        """
        # Check if model has extension manager
        if hasattr(model, "extension_manager"):
            model_ext_manager = model.extension_manager
            
            # Register each extension with the model
            for name, extension in self.extensions.items():
                try:
                    model_ext_manager.register_extension(extension)
                    self.logger.info(f"Registered extension '{name}' with model")
                except Exception as e:
                    self.logger.error(f"Error registering extension '{name}' with model: {e}")
        else:
            self.logger.warning("Model does not have extension_manager, extensions may not be fully integrated")
            
            # Try to attach extensions directly
            for name, extension in self.extensions.items():
                try:
                    # Check for direct attachment method
                    if hasattr(model, f"attach_{name}_extension"):
                        attach_method = getattr(model, f"attach_{name}_extension")
                        attach_method(extension)
                        self.logger.info(f"Attached extension '{name}' directly to model")
                except Exception as e:
                    self.logger.error(f"Error attaching extension '{name}' directly to model: {e}")
        
        return model
    
    def pre_forward(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute pre-forward hooks.
        
        Args:
            model: Model instance
            batch: Input batch
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_forward", model, batch)
    
    def post_forward(
        self,
        model: nn.Module,
        outputs: Union[Dict[str, Any], torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Execute post-forward hooks.
        
        Args:
            model: Model instance
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_forward", model, outputs, batch)
    
    def pre_backward(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        outputs: Union[Dict[str, Any], torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Execute pre-backward hooks.
        
        Args:
            model: Model instance
            loss: Loss tensor
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_backward", model, loss, outputs, batch)
    
    def post_backward(self, model: nn.Module) -> Dict[str, Any]:
        """
        Execute post-backward hooks.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_backward", model)
    
    def pre_optimizer(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Execute pre-optimizer hooks.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_optimizer", model, optimizer)
    
    def post_optimizer(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Execute post-optimizer hooks.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_optimizer", model, optimizer)
    
    def pre_batch(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute pre-batch hooks.
        
        Args:
            model: Model instance
            batch: Input batch
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_batch", model, batch)
    
    def post_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss: torch.Tensor,
        step: int
    ) -> Dict[str, Any]:
        """
        Execute post-batch hooks.
        
        Args:
            model: Model instance
            batch: Input batch
            loss: Batch loss
            step: Current step
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_batch", model, batch, loss, step)
    
    def pre_epoch(self, model: nn.Module, epoch: int) -> Dict[str, Any]:
        """
        Execute pre-epoch hooks.
        
        Args:
            model: Model instance
            epoch: Current epoch
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_epoch", model, epoch)
    
    def post_epoch(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute post-epoch hooks.
        
        Args:
            model: Model instance
            epoch: Current epoch
            metrics: Epoch metrics
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_epoch", model, epoch, metrics)
    
    def pre_validation(self, model: nn.Module) -> Dict[str, Any]:
        """
        Execute pre-validation hooks.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_validation", model)
    
    def post_validation(
        self,
        model: nn.Module,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute post-validation hooks.
        
        Args:
            model: Model instance
            metrics: Validation metrics
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_validation", model, metrics)
    
    def pre_save(
        self,
        model: nn.Module,
        checkpoint: Dict[str, Any],
        path: str
    ) -> Dict[str, Any]:
        """
        Execute pre-save hooks.
        
        Args:
            model: Model instance
            checkpoint: Checkpoint data
            path: Checkpoint path
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_save", model, checkpoint, path)
    
    def post_save(
        self,
        model: nn.Module,
        path: str
    ) -> Dict[str, Any]:
        """
        Execute post-save hooks.
        
        Args:
            model: Model instance
            path: Checkpoint path
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_save", model, path)
    
    def pre_load(
        self,
        model: nn.Module,
        checkpoint: Dict[str, Any],
        path: str
    ) -> Dict[str, Any]:
        """
        Execute pre-load hooks.
        
        Args:
            model: Model instance
            checkpoint: Checkpoint data
            path: Checkpoint path
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("pre_load", model, checkpoint, path)
    
    def post_load(
        self,
        model: nn.Module,
        path: str
    ) -> Dict[str, Any]:
        """
        Execute post-load hooks.
        
        Args:
            model: Model instance
            path: Checkpoint path
            
        Returns:
            Dictionary mapping extension names to hook results
        """
        return self.hooks.execute_hooks("post_load", model, path)
    
    def get_extension(self, name: str) -> Optional[BaseExtension]:
        """
        Get a specific extension by name.
        
        Args:
            name: Extension name
            
        Returns:
            Extension instance or None if not found
        """
        return self.extensions.get(name)
    
    def get_extensions(self) -> Dict[str, BaseExtension]:
        """
        Get all enabled extensions.
        
        Returns:
            Dictionary mapping extension names to extension instances
        """
        return self.extensions.copy()
    
    def get_extension_names(self) -> List[str]:
        """
        Get names of all enabled extensions.
        
        Returns:
            List of extension names
        """
        return list(self.extensions.keys())
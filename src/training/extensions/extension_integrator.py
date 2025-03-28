"""
Extension integration for the enhanced training system.

This module provides utilities for integrating extensions with models
and training components, handling configuration and extension setup.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Callable, Type

import torch
import torch.nn as nn

from src.training.extensions.extension_manager import ExtensionManager
from src.model.extensions.base_extension import BaseExtension
from src.model.extensions.extension_config import ExtensionConfig


# Get logger
logger = logging.getLogger("quantum_resonance")


class ExtensionIntegrator:
    """
    Integrator for model extensions.
    
    This class handles the integration of model extensions with the
    training system, including configuration, initialization, and
    hooking extensions into models and trainers.
    """
    
    def __init__(
        self,
        extension_manager: Optional[ExtensionManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the extension integrator.
        
        Args:
            extension_manager: Extension manager instance
            config: Extension configuration
        """
        self.extension_manager = extension_manager
        self.config = config or {}
        self.extensions = {}
        self.initialized = False
    
    def integrate_extensions_with_model(
        self,
        model: nn.Module,
        extension_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> nn.Module:
        """
        Integrate extensions with a model.
        
        Args:
            model: Model to integrate extensions with
            extension_configs: Configuration for extensions
            
        Returns:
            Model with integrated extensions
        """
        if self.extension_manager is None:
            logger.warning("No extension manager available, skipping extension integration")
            return model
        
        configs = extension_configs or self.config.get("extension_configs", {})
        enabled_extensions = self.extension_manager.get_active_extensions()
        
        if not enabled_extensions:
            logger.info("No active extensions to integrate")
            return model
        
        logger.info(f"Integrating {len(enabled_extensions)} extensions with model")
        
        # Process extensions in order of initialization
        init_order = self._get_initialization_order(enabled_extensions)
        
        for ext_name in init_order:
            if ext_name not in enabled_extensions:
                continue
                
            logger.info(f"Integrating extension: {ext_name}")
            
            # Get extension instance
            ext_instance = self.extension_manager.get_extension(ext_name)
            
            if ext_instance is None:
                logger.warning(f"Extension {ext_name} not found, skipping")
                continue
            
            # Get extension-specific config
            ext_config = configs.get(ext_name, {})
            
            # Integrate with model
            try:
                if hasattr(ext_instance, "integrate_with_model"):
                    model = ext_instance.integrate_with_model(model, ext_config)
                    logger.info(f"Successfully integrated extension {ext_name}")
                else:
                    logger.warning(f"Extension {ext_name} does not implement integrate_with_model")
            except Exception as e:
                logger.error(f"Error integrating extension {ext_name}: {e}")
        
        self.initialized = True
        return model
    
    def initialize_model_extensions(
        self,
        model: nn.Module,
        extension_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Initialize extensions for a model.
        
        Args:
            model: Model to initialize extensions for
            extension_configs: Configuration for extensions
        """
        if self.extension_manager is None:
            logger.warning("No extension manager available, skipping extension initialization")
            return
        
        configs = extension_configs or self.config.get("extension_configs", {})
        enabled_extensions = self.extension_manager.get_active_extensions()
        
        if not enabled_extensions:
            logger.info("No active extensions to initialize")
            return
        
        logger.info(f"Initializing {len(enabled_extensions)} model extensions")
        
        # Process extensions in order of initialization
        init_order = self._get_initialization_order(enabled_extensions)
        
        for ext_name in init_order:
            if ext_name not in enabled_extensions:
                continue
                
            logger.info(f"Initializing extension: {ext_name}")
            
            # Get extension instance
            ext_instance = self.extension_manager.get_extension(ext_name)
            
            if ext_instance is None:
                logger.warning(f"Extension {ext_name} not found, skipping")
                continue
            
            # Get extension-specific config
            ext_config = configs.get(ext_name, {})
            
            # Initialize extension
            try:
                if hasattr(ext_instance, "initialize"):
                    ext_instance.initialize(model, ext_config)
                    logger.info(f"Successfully initialized extension {ext_name}")
            except Exception as e:
                logger.error(f"Error initializing extension {ext_name}: {e}")
    
    def add_extension_to_model(
        self,
        model: nn.Module,
        extension_name: str,
        extension_config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Add a single extension to a model.
        
        Args:
            model: Model to add extension to
            extension_name: Name of the extension to add
            extension_config: Configuration for the extension
            
        Returns:
            Model with added extension
        """
        if self.extension_manager is None:
            logger.warning("No extension manager available, skipping extension addition")
            return model
        
        # Get extension instance
        ext_instance = self.extension_manager.get_extension(extension_name)
        
        if ext_instance is None:
            logger.warning(f"Extension {extension_name} not found, skipping")
            return model
        
        # Get extension-specific config
        ext_config = extension_config or self.config.get("extension_configs", {}).get(extension_name, {})
        
        # Add to model
        try:
            if hasattr(ext_instance, "integrate_with_model"):
                model = ext_instance.integrate_with_model(model, ext_config)
                logger.info(f"Successfully added extension {extension_name}")
            else:
                logger.warning(f"Extension {extension_name} does not implement integrate_with_model")
        except Exception as e:
            logger.error(f"Error adding extension {extension_name}: {e}")
        
        return model
    
    def create_extension_configuration(
        self,
        extension_name: str,
        config_params: Dict[str, Any]
    ) -> Optional[ExtensionConfig]:
        """
        Create configuration for an extension.
        
        Args:
            extension_name: Extension name
            config_params: Configuration parameters
            
        Returns:
            ExtensionConfig instance or None if creation fails
        """
        try:
            # Get extension class
            extension_class = self._get_extension_class(extension_name)
            
            if extension_class is None:
                logger.warning(f"Extension class for {extension_name} not found")
                return None
            
            # Create configuration instance
            if hasattr(extension_class, "create_config"):
                return extension_class.create_config(**config_params)
            elif hasattr(extension_class, "Config"):
                return extension_class.Config(**config_params)
            else:
                logger.warning(f"No configuration class found for extension {extension_name}")
                return ExtensionConfig(**config_params)
        
        except Exception as e:
            logger.error(f"Error creating configuration for extension {extension_name}: {e}")
            return None
    
    def get_extension_configurations(self) -> Dict[str, ExtensionConfig]:
        """
        Get configurations for all active extensions.
        
        Returns:
            Dictionary of extension configurations
        """
        if self.extension_manager is None:
            return {}
        
        configs = {}
        enabled_extensions = self.extension_manager.get_active_extensions()
        
        for ext_name in enabled_extensions:
            ext_instance = self.extension_manager.get_extension(ext_name)
            
            if ext_instance is None:
                continue
            
            if hasattr(ext_instance, "get_config"):
                configs[ext_name] = ext_instance.get_config()
            elif hasattr(ext_instance, "config"):
                configs[ext_name] = ext_instance.config
        
        return configs
    
    def _get_initialization_order(
        self,
        extensions: List[str]
    ) -> List[str]:
        """
        Get the order in which to initialize extensions.
        
        Args:
            extensions: List of extension names
            
        Returns:
            Ordered list of extension names
        """
        # Check if explicit order is specified
        if self.config and "extension_init_order" in self.config:
            init_order = self.config["extension_init_order"]
            
            # Filter to only include enabled extensions
            ordered = [ext for ext in init_order if ext in extensions]
            
            # Add any remaining extensions not in the order
            remaining = [ext for ext in extensions if ext not in ordered]
            return ordered + remaining
        
        # Get dependency graph
        dependencies = self._get_extension_dependencies(extensions)
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(ext):
            if ext in temp_visited:
                # Cyclic dependency, break the cycle
                return
            if ext in visited:
                return
            
            temp_visited.add(ext)
            
            for dep in dependencies.get(ext, []):
                if dep in extensions:
                    visit(dep)
            
            temp_visited.remove(ext)
            visited.add(ext)
            order.append(ext)
        
        for ext in extensions:
            if ext not in visited:
                visit(ext)
        
        # Reverse to get correct order
        return list(reversed(order))
    
    def _get_extension_dependencies(
        self,
        extensions: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get dependencies between extensions.
        
        Args:
            extensions: List of extension names
            
        Returns:
            Dictionary mapping extension names to their dependencies
        """
        dependencies = {}
        
        for ext_name in extensions:
            ext_instance = self.extension_manager.get_extension(ext_name)
            
            if ext_instance is None:
                continue
            
            # Get dependencies from extension
            if hasattr(ext_instance, "get_dependencies"):
                ext_deps = ext_instance.get_dependencies()
                dependencies[ext_name] = ext_deps
            else:
                dependencies[ext_name] = []
        
        return dependencies
    
    def _get_extension_class(
        self,
        extension_name: str
    ) -> Optional[Type[BaseExtension]]:
        """
        Get extension class by name.
        
        Args:
            extension_name: Extension name
            
        Returns:
            Extension class or None if not found
        """
        # Try to import from extensions
        try:
            # First try to get from extension manager
            if self.extension_manager is not None:
                ext_instance = self.extension_manager.get_extension(extension_name)
                if ext_instance is not None:
                    return ext_instance.__class__
            
            # Fall back to import
            module_path = f"src.model.extensions.{extension_name}"
            class_name = "".join(word.capitalize() for word in extension_name.split("_"))
            
            # Try direct import
            try:
                module = __import__(module_path, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError:
                pass
            
            # Try different module paths
            for category in ["memory", "multimodal", "quantum"]:
                try:
                    module_path = f"src.model.extensions.{category}.{extension_name}"
                    module = __import__(module_path, fromlist=[class_name])
                    return getattr(module, class_name)
                except ImportError:
                    pass
            
            # Try generic extension name
            try:
                module_path = f"src.model.extensions.{extension_name}"
                module = __import__(module_path, fromlist=[f"{class_name}Extension"])
                return getattr(module, f"{class_name}Extension")
            except ImportError:
                pass
        
        except Exception as e:
            logger.error(f"Error getting extension class {extension_name}: {e}")
        
        return None
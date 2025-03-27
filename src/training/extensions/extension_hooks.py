"""
Extension hook system for the enhanced training framework.

This module provides a hook system that allows model extensions
to integrate with the training process at specific points.
"""

from typing import Dict, Any, List, Callable, Optional, Union
import logging

import torch
import torch.nn as nn


class ExtensionHooks:
    """
    Manages hook points for model extensions in the training process.
    
    This class provides a system for extensions to register callbacks
    at specific points in the training process, allowing for custom
    behavior without modifying the core training code.
    """
    
    # Predefined hook points in the training process
    HOOK_POINTS = [
        # Forward pass hooks
        "pre_forward",      # Before model forward pass
        "post_forward",     # After model forward pass
        
        # Backward pass hooks
        "pre_backward",     # Before loss.backward()
        "post_backward",    # After loss.backward()
        
        # Optimization hooks
        "pre_optimizer",    # Before optimizer.step()
        "post_optimizer",   # After optimizer.step()
        
        # Batch processing hooks
        "pre_batch",        # Before batch processing
        "post_batch",       # After batch processing
        
        # Epoch hooks
        "pre_epoch",        # Before epoch starts
        "post_epoch",       # After epoch ends
        
        # Validation hooks
        "pre_validation",   # Before validation
        "post_validation",  # After validation
        
        # Checkpoint hooks
        "pre_save",         # Before saving checkpoint
        "post_save",        # After saving checkpoint
        "pre_load",         # Before loading checkpoint
        "post_load",        # After loading checkpoint
    ]
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the extension hooks system.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Initialize empty hook dictionaries
        self.hooks = {hook_point: [] for hook_point in self.HOOK_POINTS}
        
        # Track registered extensions
        self.registered_extensions = set()
    
    def register_hook(
        self,
        hook_point: str,
        callback: Callable,
        extension_name: Optional[str] = None,
        priority: int = 0
    ) -> None:
        """
        Register a callback function for a specific hook point.
        
        Args:
            hook_point: The hook point to register for
            callback: The callback function to execute
            extension_name: Name of the extension registering the hook
            priority: Hook execution priority (higher means earlier execution)
            
        Raises:
            ValueError: If hook_point is not valid
        """
        if hook_point not in self.HOOK_POINTS:
            raise ValueError(f"Invalid hook point: {hook_point}. "
                            f"Valid hook points: {self.HOOK_POINTS}")
        
        # Add hook with metadata
        hook_entry = {
            "callback": callback,
            "extension_name": extension_name or "unknown",
            "priority": priority
        }
        
        self.hooks[hook_point].append(hook_entry)
        
        # Sort hooks by priority (higher priority first)
        self.hooks[hook_point] = sorted(
            self.hooks[hook_point],
            key=lambda h: h["priority"],
            reverse=True
        )
        
        # Log registration
        self.logger.debug(f"Registered {extension_name or 'unknown'} hook for {hook_point} "
                         f"with priority {priority}")
        
        # Track extension registration
        if extension_name:
            self.registered_extensions.add(extension_name)
    
    def register_extension_hooks(
        self,
        extension: Any,
        extension_name: Optional[str] = None
    ) -> None:
        """
        Register all hooks from an extension.
        
        Args:
            extension: Extension instance with get_hooks method
            extension_name: Name of the extension (uses extension.name if available)
        """
        # Get extension name
        name = extension_name
        if name is None and hasattr(extension, "name"):
            name = extension.name
        
        # Check if extension has get_hooks method
        if not hasattr(extension, "get_hooks") or not callable(extension.get_hooks):
            self.logger.warning(f"Extension {name or 'unknown'} does not have get_hooks method")
            return
        
        # Get hooks from extension
        try:
            hooks = extension.get_hooks()
            
            # Register each hook
            for hook_point, callback in hooks.items():
                # Get priority if available
                priority = 0
                if isinstance(callback, dict) and "callback" in callback and "priority" in callback:
                    priority = callback["priority"]
                    callback = callback["callback"]
                
                self.register_hook(hook_point, callback, name, priority)
            
            self.logger.info(f"Registered hooks for extension: {name or 'unknown'}")
            
        except Exception as e:
            self.logger.error(f"Error registering hooks for extension {name or 'unknown'}: {e}")
    
    def execute_hooks(
        self,
        hook_point: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute all hooks for a specific hook point.
        
        Args:
            hook_point: The hook point to execute
            *args: Arguments to pass to the hooks
            **kwargs: Keyword arguments to pass to the hooks
            
        Returns:
            Dictionary mapping extension names to their hook results
        """
        if hook_point not in self.HOOK_POINTS:
            self.logger.warning(f"Attempted to execute invalid hook point: {hook_point}")
            return {}
        
        results = {}
        
        # Execute all registered hooks for this point
        for hook in self.hooks[hook_point]:
            extension_name = hook["extension_name"]
            callback = hook["callback"]
            
            try:
                # Execute the hook and store the result
                result = callback(*args, **kwargs)
                results[extension_name] = result
            except Exception as e:
                self.logger.error(f"Error executing {hook_point} hook for {extension_name}: {e}")
                results[extension_name] = None
        
        return results
    
    def has_hooks_for(self, hook_point: str) -> bool:
        """
        Check if there are any hooks registered for a specific hook point.
        
        Args:
            hook_point: The hook point to check
            
        Returns:
            True if hooks are registered, False otherwise
        """
        if hook_point not in self.HOOK_POINTS:
            return False
        
        return len(self.hooks[hook_point]) > 0
    
    def get_registered_extensions(self) -> List[str]:
        """
        Get list of registered extension names.
        
        Returns:
            List of extension names
        """
        return list(self.registered_extensions)
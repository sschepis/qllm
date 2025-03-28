"""
Extension integration system for the enhanced training framework.

This package provides components for integrating model extensions with the
training process, enabling custom behaviors during different phases of training.
"""

from src.training.extensions.extension_hooks import ExtensionHooks
from src.training.extensions.extension_manager import ExtensionManager

__all__ = [
    'ExtensionHooks',
    'ExtensionManager',
]
"""
Extended Semantic Resonance Model with support for extensions.

This module provides an extended version of the SemanticResonanceModel
that supports various extensions including multimodal, memory, and quantum
enhancements. It has been refactored to reduce code duplication.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Type, Callable

from src.model.semantic_resonance_model import SemanticResonanceModel
from src.config.model_config import ModelConfig


class SemanticResonanceModelWithExtensions(SemanticResonanceModel):
    """
    Semantic Resonance Model with extension support.
    
    This model extends the base SemanticResonanceModel to add support for
    various extensions that enhance the model's capabilities. It supports:
    - Memory extensions for improved long-term context
    - Multimodal extensions for handling different modalities
    - Quantum extensions for quantum computing integration
    """
    
    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        """
        Initialize the model with extensions.
        
        Args:
            config: Model configuration as ModelConfig object or dictionary
        """
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        
        # Call parent constructor
        super().__init__(config)
        
        # Extension registry for dynamically loaded extensions
        self.extensions = {}
        
        # Set up extensions based on configuration
        self.extensions_enabled = True
        self._setup_extensions()
    
    def _setup_extensions(self) -> None:
        """Set up model extensions based on configuration."""
        # Set up memory extension
        if self.config.has_extension("memory"):
            self._setup_memory_extension()
        
        # Set up multimodal extension
        if self.config.has_extension("multimodal"):
            self._setup_multimodal_extension()
        
        # Set up quantum extension
        if self.config.has_extension("quantum"):
            self._setup_quantum_extension()
        
        # Set up additional extensions
        extension_config = getattr(self.config, "extensions", {})
        if isinstance(extension_config, dict):
            for ext_name, ext_config in extension_config.items():
                if ext_config.get("enabled", False) and ext_name not in ["memory", "multimodal", "quantum"]:
                    self._setup_custom_extension(ext_name, ext_config)
    
    def _setup_memory_extension(self) -> None:
        """Set up memory extension."""
        try:
            # Import memory extension
            from src.model.extensions.memory import MemoryExtension
            
            # Get memory configuration
            memory_config = self.config.get_extension_config("memory")
            
            # Create memory extension
            memory_extension = MemoryExtension(
                model=self,
                hidden_dim=self.hidden_dim,
                config=memory_config
            )
            
            # Register memory extension
            self.memory_extension = memory_extension
            self.extensions["memory"] = memory_extension
            
            print(f"Memory extension enabled: {memory_config}")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error loading memory extension: {e}")
            self.memory_extension = None
    
    def _setup_multimodal_extension(self) -> None:
        """Set up multimodal extension."""
        try:
            # Import multimodal extension
            from src.model.extensions.multimodal import MultimodalExtension
            
            # Get multimodal configuration
            multimodal_config = self.config.get_extension_config("multimodal")
            
            # Create multimodal extension
            multimodal_extension = MultimodalExtension(
                model=self,
                hidden_dim=self.hidden_dim,
                config=multimodal_config
            )
            
            # Register multimodal extension
            self.multimodal_extension = multimodal_extension
            self.extensions["multimodal"] = multimodal_extension
            
            print(f"Multimodal extension enabled: {multimodal_config}")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error loading multimodal extension: {e}")
            self.multimodal_extension = None
    
    def _setup_quantum_extension(self) -> None:
        """Set up quantum extension."""
        try:
            # Import quantum extension
            from src.model.extensions.quantum import QuantumExtension
            
            # Get quantum configuration
            quantum_config = self.config.get_extension_config("quantum")
            
            # Create quantum extension
            quantum_extension = QuantumExtension(
                model=self,
                hidden_dim=self.hidden_dim,
                config=quantum_config
            )
            
            # Register quantum extension
            self.quantum_extension = quantum_extension
            self.extensions["quantum"] = quantum_extension
            
            print(f"Quantum extension enabled: {quantum_config}")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error loading quantum extension: {e}")
            self.quantum_extension = None
    
    def _setup_custom_extension(self, ext_name: str, ext_config: Dict[str, Any]) -> None:
        """
        Set up a custom extension.
        
        Args:
            ext_name: Name of the extension
            ext_config: Extension configuration
        """
        try:
            # Import extension class from the specified module
            module_path = ext_config.get("module", f"src.model.extensions.{ext_name}")
            class_name = ext_config.get("class", f"{ext_name.capitalize()}Extension")
            
            # Dynamically import the extension class
            import importlib
            module = importlib.import_module(module_path)
            extension_class = getattr(module, class_name)
            
            # Create extension instance
            extension = extension_class(
                model=self,
                hidden_dim=self.hidden_dim,
                config=ext_config
            )
            
            # Register extension
            self.extensions[ext_name] = extension
            setattr(self, f"{ext_name}_extension", extension)
            
            print(f"Custom extension '{ext_name}' enabled: {ext_config}")
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            print(f"Error loading custom extension '{ext_name}': {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_metadata: bool = False,
        extension_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with extension support.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for computing loss
            return_dict: Whether to return output as dictionary
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_metadata: Whether to return metadata from layers
            extension_kwargs: Additional keyword arguments for extensions
            **kwargs: Additional keyword arguments
            
        Returns:
            Model outputs
        """
        # Initialize extension inputs
        ext_inputs = {}
        
        # Process extension inputs for multimodal
        if self.multimodal_extension is not None and extension_kwargs:
            # Extract multimodal inputs (images, audio, etc.)
            multimodal_inputs = extension_kwargs.get("multimodal", {})
            ext_inputs["multimodal"] = multimodal_inputs
        
        # Process memory extension
        if self.memory_extension is not None and extension_kwargs:
            memory_kwargs = extension_kwargs.get("memory", {})
            ext_inputs["memory"] = memory_kwargs
        
        # Process quantum extension
        if self.quantum_extension is not None and extension_kwargs:
            quantum_kwargs = extension_kwargs.get("quantum", {})
            ext_inputs["quantum"] = quantum_kwargs
        
        # Process other extensions
        for ext_name, extension in self.extensions.items():
            if ext_name not in ["multimodal", "memory", "quantum"] and extension_kwargs:
                ext_inputs[ext_name] = extension_kwargs.get(ext_name, {})
        
        # Apply extensions pre-forward hooks
        for ext_name, extension in self.extensions.items():
            if hasattr(extension, "pre_forward"):
                # Apply pre-forward hook for the extension
                input_ids, attention_mask, position_ids, kwargs = extension.pre_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    ext_inputs=ext_inputs.get(ext_name, {}),
                    **kwargs
                )
        
        # Call parent forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,  # Always use dict for extensions
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_metadata=return_metadata,
            **kwargs
        )
        
        # Apply extensions post-forward hooks
        for ext_name, extension in self.extensions.items():
            if hasattr(extension, "post_forward"):
                # Apply post-forward hook for the extension
                outputs = extension.post_forward(
                    outputs=outputs,
                    ext_inputs=ext_inputs.get(ext_name, {})
                )
        
        # Return outputs in the requested format
        if not return_dict:
            return outputs.get("loss", None) if labels is not None else outputs.get("logits", None)
        
        return outputs
    
    def generate(
        self,
        input_ids: Union[torch.Tensor, str],
        attention_mask: Optional[torch.Tensor] = None,
        extension_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, str]:
        """
        Generate text with extension support.
        
        Args:
            input_ids: Input token IDs or text string
            attention_mask: Attention mask
            extension_kwargs: Additional keyword arguments for extensions
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs or text
        """
        # Initialize extension inputs
        ext_inputs = {}
        if extension_kwargs:
            ext_inputs = extension_kwargs
        
        # Apply extensions pre-generate hooks
        for ext_name, extension in self.extensions.items():
            if hasattr(extension, "pre_generate"):
                # Apply pre-generate hook for the extension
                input_ids, attention_mask, kwargs = extension.pre_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ext_inputs=ext_inputs.get(ext_name, {}),
                    **kwargs
                )
        
        # Call parent generate method
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply extensions post-generate hooks
        for ext_name, extension in self.extensions.items():
            if hasattr(extension, "post_generate"):
                # Apply post-generate hook for the extension
                outputs = extension.post_generate(
                    outputs=outputs,
                    ext_inputs=ext_inputs.get(ext_name, {}),
                    **kwargs
                )
        
        return outputs
    
    def add_extension(self, name: str, extension: Any) -> None:
        """
        Add a new extension to the model.
        
        Args:
            name: Name of the extension
            extension: Extension object
        """
        self.extensions[name] = extension
        setattr(self, f"{name}_extension", extension)
        
        # Update the config to include the new extension
        if not hasattr(self.config, "extensions"):
            self.config.extensions = {}
        
        self.config.extensions[name] = {"enabled": True}
    
    def remove_extension(self, name: str) -> bool:
        """
        Remove an extension from the model.
        
        Args:
            name: Name of the extension
            
        Returns:
            True if extension was removed, False if not found
        """
        if name in self.extensions:
            # Get the extension
            extension = self.extensions[name]
            
            # Call cleanup if available
            if hasattr(extension, "cleanup"):
                extension.cleanup()
            
            # Remove from extensions dict
            del self.extensions[name]
            
            # Remove attribute if it exists
            if hasattr(self, f"{name}_extension"):
                delattr(self, f"{name}_extension")
            
            # Update config
            if hasattr(self.config, "extensions") and name in self.config.extensions:
                self.config.extensions[name]["enabled"] = False
            
            return True
        
        return False
    
    def get_extension(self, name: str) -> Optional[Any]:
        """
        Get an extension by name.
        
        Args:
            name: Name of the extension
            
        Returns:
            Extension object or None if not found
        """
        return self.extensions.get(name)
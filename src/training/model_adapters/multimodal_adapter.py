"""
Multimodal model adapter for the enhanced training system.

This module provides an implementation of the ModelAdapter for multimodal models
with vision capabilities, handling initialization, forward passes, and loss
computation specific to text-image multimodal training.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.model.semantic_resonance_model import SemanticResonanceModel
from src.training.model_adapters.base_adapter import ModelAdapter
from src.model.extensions.multimodal.vision_extension import VisionExtension


logger = logging.getLogger("quantum_resonance")


class MultimodalModelAdapter(ModelAdapter):
    """
    Model adapter implementation for multimodal models with vision capabilities.
    
    This adapter handles model initialization, tokenizer setup, vision encoder
    integration, batch processing for text-image pairs, and loss computation
    for multimodal training.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize multimodal model adapter.
        
        Args:
            model_config: Configuration for the model architecture
            training_config: Configuration for training
            device: Device to use for model execution
            logger: Logger instance
        """
        super().__init__(model_config, training_config, device)
        self.logger = logger or logging.getLogger("quantum_resonance")
        
        # Multimodal-specific settings
        self.vision_encoder_type = getattr(self.model_config, "vision_encoder", "resnet50")
        self.image_size = getattr(self.model_config, "image_size", 224)
        self.patch_size = getattr(self.model_config, "patch_size", 16)
        
        # Ensure model config has necessary parameters for multimodal
        self._ensure_multimodal_config()
        
        # Vision-specific components
        self.vision_extension = None
    
    def _ensure_multimodal_config(self) -> None:
        """Ensure model config has necessary parameters for multimodal."""
        if not hasattr(self.model_config, "use_multimodal"):
            self.logger.info("Setting use_multimodal=True for multimodal model")
            setattr(self.model_config, "use_multimodal", True)
        
        if not hasattr(self.model_config, "vision_encoder"):
            self.logger.info(f"Setting vision_encoder={self.vision_encoder_type} for multimodal model")
            setattr(self.model_config, "vision_encoder", self.vision_encoder_type)
        
        if not hasattr(self.model_config, "image_size"):
            self.logger.info(f"Setting image_size={self.image_size} for multimodal model")
            setattr(self.model_config, "image_size", self.image_size)
        
        if not hasattr(self.model_config, "patch_size"):
            self.logger.info(f"Setting patch_size={self.patch_size} for multimodal model")
            setattr(self.model_config, "patch_size", self.patch_size)
    
    def create_model(self) -> nn.Module:
        """
        Create and initialize the multimodal model.
        
        Returns:
            Initialized multimodal model instance
        """
        self.logger.info("Initializing multimodal model...")
        
        # Create model instance with multimodal-specific configurations
        model = SemanticResonanceModel(self.model_config)
        
        # Initialize vision extension
        self._initialize_vision_extension(model)
        
        # Move model to device
        model.to(self.device)
        
        # Log model size
        total_params, trainable_params = self.compute_model_size(model)
        self.logger.info(f"Multimodal model size: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        return model
    
    def _initialize_vision_extension(self, model: nn.Module) -> None:
        """
        Initialize vision extension for the model.
        
        Args:
            model: Model instance to attach vision extension to
        """
        self.logger.info(f"Initializing vision extension with encoder: {self.vision_encoder_type}")
        
        # Create vision extension configuration
        from src.model.extensions.multimodal.multimodal_config import MultimodalConfig
        vision_config = MultimodalConfig(
            vision_encoder=self.vision_encoder_type,
            image_size=self.image_size,
            patch_size=self.patch_size,
            enabled=True
        )
        
        # Create and attach vision extension
        try:
            self.vision_extension = VisionExtension(vision_config)
            
            # If the model has an extension_manager, register the vision extension
            if hasattr(model, "extension_manager"):
                model.extension_manager.register_extension(self.vision_extension)
                self.logger.info("Vision extension registered with model extension manager")
            else:
                self.logger.warning("Model does not have extension_manager, vision capabilities may be limited")
                
                # Attach vision encoder directly if needed
                if hasattr(model, "attach_vision_encoder"):
                    model.attach_vision_encoder(self.vision_extension.vision_encoder)
                    self.logger.info("Vision encoder attached directly to model")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vision extension: {e}")
            raise
    
    def create_tokenizer(self) -> Any:
        """
        Create and initialize the tokenizer with multimodal tokens.
        
        Returns:
            Initialized tokenizer with multimodal tokens
        """
        self.logger.info(f"Loading tokenizer for multimodal model: {self.model_config.tokenizer_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer_name)
        
        # Set default pad token if not set
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add multimodal-specific tokens
        special_tokens = {
            "additional_special_tokens": []
        }
        
        # Add image tokens if not present
        multimodal_tokens = ["<image>", "<image_features>", "</image>"]
        missing_tokens = [token for token in multimodal_tokens if token not in tokenizer.get_vocab()]
        
        if missing_tokens:
            self.logger.info(f"Adding missing multimodal tokens to tokenizer: {missing_tokens}")
            special_tokens["additional_special_tokens"].extend(missing_tokens)
        
        # Resize tokenizer if needed
        if special_tokens["additional_special_tokens"]:
            tokenizer.add_special_tokens(special_tokens)
            
            # Update vocab size in model config
            self.model_config.vocab_size = len(tokenizer)
            
            # Resize model's token embeddings
            if self.model is not None:
                self.model = self.resize_token_embeddings(self.model, tokenizer)
        
        return tokenizer
    
    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare a multimodal batch for model processing.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Batch prepared for model input
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            prepared_batch = batch.copy()
        else:
            # Handle tuple batch (typical for multimodal datasets)
            try:
                if len(batch) >= 4:  # input_ids, attention_mask, labels, images
                    input_ids, attention_mask, labels, images = batch[:4]
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "images": images
                    }
                elif len(batch) == 3:  # input_ids, attention_mask, images
                    input_ids, attention_mask, images = batch
                    prepared_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "images": images,
                        "labels": input_ids.clone()  # Use inputs as labels
                    }
                else:
                    raise ValueError(f"Unexpected multimodal batch format with {len(batch)} elements")
            except Exception as e:
                raise ValueError(f"Error unpacking multimodal batch: {e}")
        
        # Ensure required keys are present
        required_keys = ["input_ids", "attention_mask"]
        for key in required_keys:
            if key not in prepared_batch:
                raise ValueError(f"Multimodal batch is missing required key: {key}")
        
        # Process images if present
        if "images" in prepared_batch:
            prepared_batch = self._process_images(prepared_batch)
        
        # If no labels are present, add dummy labels using input_ids
        if "labels" not in prepared_batch:
            self.logger.warning("No labels found in multimodal batch. Using input_ids as labels.")
            prepared_batch["labels"] = prepared_batch["input_ids"].clone()
        
        # Move batch to device
        return self.move_to_device(prepared_batch)
    
    def _process_images(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process images in the batch.
        
        Args:
            batch: Batch containing images
            
        Returns:
            Batch with processed images
        """
        images = batch["images"]
        
        # Handle different image formats
        if isinstance(images, torch.Tensor):
            # Images already in tensor format
            batch["image_features"] = images
        elif isinstance(images, np.ndarray):
            # Convert numpy array to tensor
            batch["image_features"] = torch.from_numpy(images)
        elif isinstance(images, list):
            if all(isinstance(img, str) for img in images):
                # List of image paths
                self.logger.warning("Processing image paths in batch - this may be slow")
                # This would require loading and processing the images
                # For production, preprocess images and provide tensors directly
                batch["image_features"] = self._load_images_from_paths(images)
            else:
                # Assume list of tensors or numpy arrays
                batch["image_features"] = torch.stack([
                    torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                    for img in images
                ])
        else:
            self.logger.warning(f"Unknown image format in batch: {type(images)}")
            batch["image_features"] = None
        
        # Delete original images to save memory if image_features were created
        if "image_features" in batch and batch["image_features"] is not None:
            del batch["images"]
        
        return batch
    
    def _load_images_from_paths(self, image_paths: List[str]) -> torch.Tensor:
        """
        Load and process images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tensor of processed images
        """
        # This is a placeholder implementation
        # In a real implementation, you would use PIL or cv2 to load images and process them
        
        self.logger.warning("Loading images from paths is not fully implemented")
        
        # Check if vision extension is available for preprocessing
        if self.vision_extension is not None and hasattr(self.vision_extension, "preprocess_images"):
            return self.vision_extension.preprocess_images(image_paths)
        
        # Placeholder implementation
        batch_size = len(image_paths)
        return torch.zeros((batch_size, 3, self.image_size, self.image_size))
    
    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Perform forward pass with the multimodal model.
        
        Args:
            model: Model instance
            batch: Prepared multimodal input batch
            return_dict: Whether to return outputs as a dictionary
            
        Returns:
            Model outputs (as dict if return_dict=True, otherwise as tuple)
        """
        # Process image features if present
        if "image_features" in batch:
            # Check if model has vision extension or capability
            if hasattr(model, "process_images") and callable(model.process_images):
                # Let model handle image processing
                outputs = model(
                    **{k: v for k, v in batch.items() if k != "image_features"},
                    images=batch["image_features"],
                    return_dict=return_dict
                )
            elif self.vision_extension is not None:
                # Use vision extension to process images
                with torch.no_grad():
                    image_embeddings = self.vision_extension.encode_images(batch["image_features"])
                
                # Add image embeddings to batch
                vision_batch = {k: v for k, v in batch.items() if k != "image_features"}
                vision_batch["image_embeddings"] = image_embeddings
                
                # Forward pass with image embeddings
                outputs = model(**vision_batch, return_dict=return_dict)
            else:
                # No vision capability, ignore images
                self.logger.warning("Model does not have vision capability, ignoring images")
                outputs = model(
                    **{k: v for k, v in batch.items() if k != "image_features"},
                    return_dict=return_dict
                )
        else:
            # Standard forward pass without images
            outputs = model(**{k: v for k, v in batch.items() if k != "images"}, return_dict=return_dict)
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Union[Dict[str, Any], Tuple],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss from multimodal model outputs and batch.
        
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
            raise ValueError("Unable to extract loss from multimodal model outputs")
        
        # Multimodal-specific loss handling
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            self.logger.warning("Detected NaN/Inf in multimodal model loss calculation")
            
            # For debugging purposes in multimodal models
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                self.logger.warning(f"Logits shape: {logits.shape}, "
                                   f"min: {logits.min().item()}, "
                                   f"max: {logits.max().item()}, "
                                   f"mean: {logits.mean().item()}, "
                                   f"has NaN: {torch.isnan(logits).any().item()}")
        
        return loss
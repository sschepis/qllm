"""
Tensor collate functions for QLLM dataloaders.

This module provides custom collate functions for different types of data
to properly batch them for training and evaluation.
"""

import torch
from typing import Dict, List, Any, Tuple, Union


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Default collate function for text data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch as a dictionary of tensors
    """
    # Initialize the output dictionary
    collated_batch = {}
    
    # Process all keys in the first sample
    if not batch:
        return collated_batch
        
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    for key in keys:
        if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            # Pad sequences of different lengths
            if batch[0][key].dim() >= 1:
                # Get the maximum sequence length in this batch
                max_len = max(sample[key].size(0) for sample in batch if key in sample)
                
                # Create a list to store padded tensors
                padded_tensors = []
                
                # Pad each tensor to the maximum length
                for sample in batch:
                    if key not in sample:
                        continue
                        
                    tensor = sample[key]
                    
                    if tensor.size(0) < max_len:
                        # Create padding tensor
                        pad_size = list(tensor.size())
                        pad_size[0] = max_len - tensor.size(0)
                        
                        # Create padding with zeros or -100 for labels
                        if key == "labels":
                            padding = torch.full(pad_size, -100, dtype=tensor.dtype, device=tensor.device)
                        else:
                            padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
                        
                        # Concatenate original tensor with padding
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                        padded_tensors.append(padded_tensor)
                    else:
                        padded_tensors.append(tensor)
                
                # Stack the padded tensors
                collated_batch[key] = torch.stack(padded_tensors)
            else:
                # For scalar tensors, just stack them
                try:
                    collated_batch[key] = torch.stack([sample[key] for sample in batch if key in sample])
                except:
                    # If stacking fails, use a list
                    collated_batch[key] = [sample[key] for sample in batch if key in sample]
        else:
            # Non-tensor values are collected into a list
            collated_batch[key] = [sample[key] for sample in batch if key in sample]
    
    return collated_batch


def dialogue_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for dialogue data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch as a dictionary of tensors
    """
    # Initialize the output dictionary
    collated_batch = {}
    
    if not batch:
        return collated_batch
    
    # Special handling for dialogue format
    if "input_ids" in batch[0] and "response_ids" in batch[0]:
        # Process input parts
        input_ids = [sample["input_ids"] for sample in batch]
        input_attention_mask = [sample.get("input_attention_mask", None) for sample in batch]
        
        # Process response parts
        response_ids = [sample["response_ids"] for sample in batch]
        response_attention_mask = [sample.get("response_attention_mask", None) for sample in batch]
        
        # Get the maximum lengths
        max_input_len = max(t.size(0) for t in input_ids)
        max_response_len = max(t.size(0) for t in response_ids)
        
        # Pad all sequences
        padded_input_ids = []
        padded_input_masks = []
        padded_response_ids = []
        padded_response_masks = []
        input_lengths = []
        
        # Pad inputs
        for i, tensor in enumerate(input_ids):
            pad_length = max_input_len - tensor.size(0)
            if pad_length > 0:
                padding = torch.zeros(pad_length, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, padding], dim=0)
                padded_input_ids.append(padded_tensor)
                
                # Pad attention mask if it exists
                if input_attention_mask[i] is not None:
                    mask_padding = torch.zeros(pad_length, dtype=input_attention_mask[i].dtype, device=input_attention_mask[i].device)
                    padded_mask = torch.cat([input_attention_mask[i], mask_padding], dim=0)
                    padded_input_masks.append(padded_mask)
            else:
                padded_input_ids.append(tensor)
                if input_attention_mask[i] is not None:
                    padded_input_masks.append(input_attention_mask[i])
                    
            # Store original input length
            input_lengths.append(tensor.size(0))
        
        # Pad responses
        for i, tensor in enumerate(response_ids):
            pad_length = max_response_len - tensor.size(0)
            if pad_length > 0:
                padding = torch.zeros(pad_length, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, padding], dim=0)
                padded_response_ids.append(padded_tensor)
                
                # Pad attention mask if it exists
                if response_attention_mask[i] is not None:
                    mask_padding = torch.zeros(pad_length, dtype=response_attention_mask[i].dtype, device=response_attention_mask[i].device)
                    padded_mask = torch.cat([response_attention_mask[i], mask_padding], dim=0)
                    padded_response_masks.append(padded_mask)
            else:
                padded_response_ids.append(tensor)
                if response_attention_mask[i] is not None:
                    padded_response_masks.append(response_attention_mask[i])
        
        # Stack tensors
        collated_batch["input_ids"] = torch.stack(padded_input_ids)
        if padded_input_masks:
            collated_batch["input_attention_mask"] = torch.stack(padded_input_masks)
        collated_batch["response_ids"] = torch.stack(padded_response_ids)
        if padded_response_masks:
            collated_batch["response_attention_mask"] = torch.stack(padded_response_masks)
        collated_batch["input_lengths"] = torch.tensor(input_lengths, dtype=torch.long)
        
        return collated_batch
    
    # Fall back to default collate for other formats
    return default_collate_fn(batch)


def function_calling_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for function calling data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch as a dictionary of tensors
    """
    # Initialize the output dictionary
    collated_batch = {}
    
    if not batch:
        return collated_batch
    
    # Use the default collate function for most fields
    collated_batch = default_collate_fn(batch)
    
    # Special handling for function name and parameters
    if "function_name" in batch[0] and isinstance(batch[0]["function_name"], torch.Tensor):
        function_names = [sample["function_name"] for sample in batch]
        collated_batch["function_name"] = torch.stack(function_names)
    
    # Additional processing specific to function calling data
    if "parameters" in collated_batch and isinstance(collated_batch["parameters"], torch.Tensor):
        # Ensure parameters have consistent tensor shapes
        if collated_batch["parameters"].dim() == 2:
            # Already properly shaped, no action needed
            pass
        elif collated_batch["parameters"].dim() == 1:
            # Convert to a batch dimension of 1
            collated_batch["parameters"] = collated_batch["parameters"].unsqueeze(0)
    
    return collated_batch


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multimodal data.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch as a dictionary of tensors
    """
    # Initialize the output dictionary
    collated_batch = {}
    
    if not batch:
        return collated_batch
    
    # Use the default collate function for text fields
    text_keys = ["input_ids", "attention_mask", "labels"]
    text_batch = [{k: v for k, v in sample.items() if k in text_keys} for sample in batch]
    text_collated = default_collate_fn(text_batch)
    collated_batch.update(text_collated)
    
    # Special handling for image data
    if "image" in batch[0] and batch[0]["image"] is not None:
        # Collect all valid images (some samples might have missing images)
        valid_images = [sample["image"] for sample in batch if "image" in sample and sample["image"] is not None]
        
        if valid_images:
            # Stack images into a single tensor
            try:
                collated_batch["images"] = torch.stack(valid_images)
                collated_batch["has_image"] = torch.tensor([
                    1 if ("image" in sample and sample["image"] is not None) else 0
                    for sample in batch
                ], dtype=torch.bool)
            except Exception as e:
                # If images have different shapes, we may need more complex handling
                collated_batch["images"] = valid_images
                collated_batch["has_image"] = [
                    ("image" in sample and sample["image"] is not None)
                    for sample in batch
                ]
    
    # Include metadata fields with "metadata_" prefix
    metadata_keys = [k for k in batch[0].keys() if k.startswith("metadata_")]
    for key in metadata_keys:
        collated_batch[key] = [sample.get(key) for sample in batch]
    
    return collated_batch
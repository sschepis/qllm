"""
Dialogue dataset for the Quantum Resonance Language Model.

This module provides a dataset for dialogue data, supporting
conversational training with multiple speakers.
"""

import os
import json
import torch
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

# Set up logging
logger = logging.getLogger("dialogue_dataset")

class DialogueDataset(Dataset):
    """
    Dataset for dialogue data.
    
    This dataset is designed for conversational data with multiple turns.
    It supports various dialogue formats and includes special tokens for
    different speakers.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[str] = None,
        dialogues: Optional[List[List[Dict[str, str]]]] = None,
        max_length: int = 512,
        system_prompt: Optional[str] = None,
        role_mapping: Optional[Dict[str, str]] = None,
        add_eos_token: bool = True,
        add_special_tokens: bool = True,
        processing_batch_size: int = 8,  # Reduced batch size
        assistant_token_id: Optional[int] = None,  # Pre-computed assistant token ID
        role_token_ids: Optional[Set[int]] = None  # Pre-computed role token IDs
    ):
        """
        Initialize the dialogue dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding texts
            data_path: Path to the dialogue data file (JSON)
            dialogues: Pre-loaded dialogues (if data_path is not provided)
            max_length: Maximum sequence length
            system_prompt: Optional system prompt to add at start of dialogues
            role_mapping: Mapping of role names to special tokens
            add_eos_token: Whether to add EOS token after each message
            add_special_tokens: Whether to add role tokens
            processing_batch_size: Batch size for dialogue processing
            assistant_token_id: Pre-computed assistant token ID for optimization
            role_token_ids: Pre-computed set of role token IDs for optimization
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.system_prompt = system_prompt or ""
        self.add_eos_token = add_eos_token
        self.add_special_tokens = add_special_tokens
        self.processing_batch_size = processing_batch_size
        
        logger.info(f"Creating DialogueDataset (max_length={max_length}, batch_size={processing_batch_size})")
        
        # Default role mapping if not provided
        self.role_mapping = role_mapping or {
            "system": "<system>",
            "user": "<user>",
            "assistant": "<assistant>",
            "human": "<user>",
            "bot": "<assistant>"
        }
        
        # Pre-compute token IDs for roles if not provided
        self._pre_compute_token_ids(assistant_token_id, role_token_ids)
        
        # Performance tracking
        total_start = time.time()
        
        # Load dialogue data - either from provided dialogues or from file
        if dialogues is not None:
            logger.info(f"Using {len(dialogues)} pre-loaded dialogues")
            self.dialogues = dialogues
        else:
            logger.info(f"Loading dialogues from file: {data_path}")
            self.dialogues = self._load_dialogues()
            
        logger.info(f"Loaded {len(self.dialogues)} dialogues")
        
        # Prepare tokenized dialogues with progress bar
        self.tokenized_dialogues = self._prepare_dialogues()
        
        logger.info(f"DialogueDataset initialized in {time.time() - total_start:.2f} seconds")
    
    def _pre_compute_token_ids(self, assistant_token_id=None, role_token_ids=None):
        """
        Pre-compute token IDs for roles for faster processing.
        
        Args:
            assistant_token_id: Pre-computed assistant token ID
            role_token_ids: Pre-computed set of role token IDs
        """
        # Use pre-computed values if provided
        if assistant_token_id is not None and role_token_ids is not None:
            self.assistant_token_id = assistant_token_id
            self.role_token_ids = role_token_ids
            logger.info("Using pre-computed token IDs")
            return
            
        # Compute from scratch
        logger.info("Pre-computing role token IDs")
        
        # Get assistant token ID
        assistant_token = self.role_mapping.get("assistant", "<assistant>")
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(assistant_token)
        
        # Check if assistant token is recognized
        if self.assistant_token_id == self.tokenizer.unk_token_id:
            logger.warning(f"Assistant token '{assistant_token}' not found in tokenizer vocabulary!")
            # Force add it to tokenizer if needed
            if self.add_special_tokens and assistant_token not in self.tokenizer.get_vocab():
                logger.info(f"Adding assistant token '{assistant_token}' to tokenizer")
                self.tokenizer.add_special_tokens({"additional_special_tokens": [assistant_token]})
                self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(assistant_token)
        
        # Get all role token IDs
        self.role_token_ids = set()
        for role, token in self.role_mapping.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                self.role_token_ids.add(token_id)
            else:
                logger.warning(f"Role token '{token}' not found in tokenizer vocabulary!")
                # Force add it to tokenizer if needed
                if self.add_special_tokens and token not in self.tokenizer.get_vocab():
                    logger.info(f"Adding role token '{token}' to tokenizer")
                    self.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    self.role_token_ids.add(token_id)
        
        logger.info(f"Pre-computed {len(self.role_token_ids)} role token IDs")
        
        # Verify assistant token is in role tokens
        if self.assistant_token_id not in self.role_token_ids and self.assistant_token_id != self.tokenizer.unk_token_id:
            logger.warning(f"Assistant token ID {self.assistant_token_id} not in role token IDs! Adding it.")
            self.role_token_ids.add(self.assistant_token_id)
    
    def _load_dialogues(self) -> List[List[Dict[str, str]]]:
        """
        Load dialogue data from file.
        
        Returns:
            List of dialogues, where each dialogue is a list of turns
        """
        if not self.data_path:
            raise ValueError("Either data_path or dialogues must be provided")
            
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, list):
            # Check if it's a list of dialogues or a list of messages
            if data and isinstance(data[0], list):
                # List of dialogues
                dialogues = data
            elif data and isinstance(data[0], dict) and "role" in data[0]:
                # Single dialogue as a list of messages
                dialogues = [data]
            else:
                # Try to parse as a list of dialogue objects
                dialogues = []
                for item in data:
                    if isinstance(item, dict) and "messages" in item:
                        dialogues.append(item["messages"])
                    elif isinstance(item, dict) and "conversation" in item:
                        dialogues.append(item["conversation"])
                    elif isinstance(item, dict) and "dialogue" in item:
                        dialogues.append(item["dialogue"])
        elif isinstance(data, dict):
            # Single dialogue object
            if "messages" in data:
                dialogues = [data["messages"]]
            elif "conversation" in data:
                dialogues = [data["conversation"]]
            elif "dialogue" in data:
                dialogues = [data["dialogue"]]
            elif "dialogues" in data:
                dialogues = data["dialogues"]
            else:
                raise ValueError(f"Unsupported data format in {self.data_path}")
        else:
            raise ValueError(f"Unsupported data format in {self.data_path}")
        
        # Validate and normalize dialogues
        normalized_dialogues = []
        for dialogue in dialogues:
            normalized_dialogue = []
            for turn in dialogue:
                if isinstance(turn, dict):
                    # Check if we have role and content
                    if "role" in turn and "content" in turn:
                        normalized_dialogue.append({
                            "role": turn["role"],
                            "content": turn["content"]
                        })
                    # Check if we have speaker and text
                    elif "speaker" in turn and "text" in turn:
                        normalized_dialogue.append({
                            "role": turn["speaker"],
                            "content": turn["text"]
                        })
                    # Check if we have role and message
                    elif "role" in turn and "message" in turn:
                        normalized_dialogue.append({
                            "role": turn["role"],
                            "content": turn["message"]
                        })
                elif isinstance(turn, list) and len(turn) == 2:
                    # Handle format [role, content]
                    normalized_dialogue.append({
                        "role": turn[0],
                        "content": turn[1]
                    })
            
            # Only add if dialogue is valid
            if normalized_dialogue:
                normalized_dialogues.append(normalized_dialogue)
        
        return normalized_dialogues
    
    def _prepare_dialogues(self) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare dialogues for training.
        
        This function tokenizes the dialogues and formats them for training.
        
        Returns:
            List of tokenized dialogues
        """
        tokenized_dialogues = []
        
        # Create mini-batches for efficient processing
        batch_size = self.processing_batch_size
        num_batches = (len(self.dialogues) + batch_size - 1) // batch_size
        
        logger.info(f"Tokenizing {len(self.dialogues)} dialogues in {num_batches} batches")
        
        # Process in batches with a progress bar
        for batch_idx in tqdm(range(num_batches), desc="Tokenizing dialogues"):
            batch_start_time = time.time()
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.dialogues))
            batch_dialogues = self.dialogues[start_idx:end_idx]
            
            batch_formatted_texts = []
            
            # Format each dialogue
            for dialogue in batch_dialogues:
                # Add system prompt if provided
                formatted_text = ""
                if self.system_prompt:
                    formatted_text += self._format_message("system", self.system_prompt)
                
                # Format dialogue turns
                for turn in dialogue:
                    role = turn["role"].lower()
                    content = turn["content"]
                    formatted_text += self._format_message(role, content)
                
                batch_formatted_texts.append(formatted_text)
            
            # Tokenize the batch
            batch_encodings = self.tokenizer(
                batch_formatted_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Process each item in the batch using optimized method
            for i in range(len(batch_formatted_texts)):
                input_ids = batch_encodings["input_ids"][i]
                attention_mask = batch_encodings["attention_mask"][i]
                
                # Create labels using vectorized operations
                labels = self._create_assistant_only_labels(input_ids)
                
                # Add to tokenized dialogues
                tokenized_dialogues.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })
            
            if batch_idx % 5 == 0:  # Log every 5 batches
                logger.info(f"Processed batch {batch_idx+1}/{num_batches} in {time.time() - batch_start_time:.2f}s")
        
        logger.info(f"Finished tokenizing {len(tokenized_dialogues)} dialogue examples")
        return tokenized_dialogues
    
    def _create_assistant_only_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels that only compute loss on assistant tokens.
        This is an optimized version that avoids nested loops.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Labels tensor with non-assistant tokens set to -100
        """
        # Create initial labels
        labels = input_ids.clone()
        
        # If assistant token ID is unknown, return original labels
        if self.assistant_token_id == self.tokenizer.unk_token_id:
            # For safety, set all labels to -100 to avoid NaN issues
            # We'll compute loss on all tokens in this case
            return labels
            
        # Find positions of assistant tokens
        is_assistant = (input_ids == self.assistant_token_id)
        
        if not is_assistant.any():
            # No assistant tokens found
            # For safety, return regular language modeling labels (without masking)
            # instead of setting all to -100, which could cause NaN loss
            return labels
            
        # Initialize mask for keeping assistant's text (all False initially)
        keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Find role token positions (including assistant)
        is_role_token = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(len(input_ids)):
            if input_ids[i].item() in self.role_token_ids:
                is_role_token[i] = True
        
        # Process each assistant token
        assistant_positions = is_assistant.nonzero().squeeze(1)
        
        # Make sure assistant_positions is always properly shaped
        if assistant_positions.dim() == 0:
            # Single position was found, reshape to 1D tensor
            assistant_positions = assistant_positions.unsqueeze(0)
        
        for pos in assistant_positions:
            # Get the next role token position after this assistant token
            next_positions = (is_role_token & (torch.arange(len(input_ids), device=input_ids.device) > pos))
            next_role_pos = next_positions.nonzero()
            
            # Determine end position (next role token or end of sequence)
            if next_role_pos.numel() > 0:  # Check if tensor is not empty
                # Take the first position found
                end_pos = next_role_pos[0].item()
            else:
                # No more role tokens, go until padding or end
                end_pos = len(input_ids)
                
                # Adjust for padding
                pad_positions = (input_ids == self.tokenizer.pad_token_id)
                if pad_positions.any():
                    first_pad = pad_positions.nonzero()[0].item()
                    if first_pad < end_pos:
                        end_pos = first_pad
            
            # Mark tokens to keep (the assistant's response)
            if pos < len(input_ids) - 1:  # Avoid out of bounds
                if pos + 1 < end_pos:  # Only set if range is valid
                    keep_mask[pos+1:end_pos] = True
        
        # Set non-assistant tokens to -100
        labels[~keep_mask] = -100
        
        # Verify we didn't set ALL tokens to -100, which would cause NaN loss
        if (labels == -100).all():
            # If all tokens were masked (which would cause NaN loss),
            # revert to standard language modeling
            logger.warning("All tokens were masked! Reverting to standard language modeling to avoid NaN loss.")
            return input_ids.clone()
            
        return labels
    
    def _format_message(self, role: str, content: str) -> str:
        """
        Format a message with role tokens.
        
        Args:
            role: Role (system, user, assistant, etc.)
            content: Message content
            
        Returns:
            Formatted message with role tokens
        """
        role = role.lower()
        role_token = self.role_mapping.get(role, f"<{role}>")
        
        if self.add_special_tokens:
            formatted = f"{role_token} {content}"
        else:
            formatted = content
            
        if self.add_eos_token:
            formatted += f" {self.tokenizer.eos_token}"
            
        return formatted
    
    def __len__(self) -> int:
        """Get the number of dialogues."""
        return len(self.tokenized_dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized dialogue."""
        return self.tokenized_dialogues[idx]


def dialogue_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for dialogue batches.
    
    Args:
        batch: List of tokenized dialogues
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
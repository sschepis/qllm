"""
Pretraining strategy for language models.

This module implements a strategy for pretraining language models with
techniques like masked language modeling and next sentence prediction.
"""

import logging
import random
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.strategies.base_strategy import TrainingStrategy
from src.training.model_adapters.base_adapter import ModelAdapter


# Get logger
logger = logging.getLogger("quantum_resonance")


class PretrainingStrategy(TrainingStrategy):
    """
    Strategy for pretraining language models.
    
    This strategy implements pretraining techniques such as masked language
    modeling and next sentence prediction for language models.
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        mlm_probability: float = 0.15,
        pretraining_type: str = "mlm",
        whole_word_masking: bool = False,
        max_predictions_per_seq: int = 20,
        nsp_probability: float = 0.5
    ):
        """
        Initialize the pretraining strategy.
        
        Args:
            config: Training configuration
            mlm_probability: Probability of masking tokens for MLM
            pretraining_type: Type of pretraining ("mlm", "clm", "mlm_nsp")
            whole_word_masking: Whether to mask whole words instead of tokens
            max_predictions_per_seq: Maximum number of masked tokens per sequence
            nsp_probability: Probability of choosing a random next sentence for NSP
        """
        super().__init__(config)
        
        self.mlm_probability = mlm_probability
        self.pretraining_type = pretraining_type.lower()
        self.whole_word_masking = whole_word_masking
        self.max_predictions_per_seq = max_predictions_per_seq
        self.nsp_probability = nsp_probability
        
        # Get values from config if available
        if isinstance(config, dict):
            self.mlm_probability = config.get("mlm_probability", self.mlm_probability)
            self.pretraining_type = config.get("pretraining_type", self.pretraining_type).lower()
            self.whole_word_masking = config.get("whole_word_masking", self.whole_word_masking)
            self.max_predictions_per_seq = config.get("max_predictions_per_seq", self.max_predictions_per_seq)
            self.nsp_probability = config.get("nsp_probability", self.nsp_probability)
        else:
            # Try to get from object attributes
            self.mlm_probability = getattr(config, "mlm_probability", self.mlm_probability)
            self.pretraining_type = getattr(config, "pretraining_type", self.pretraining_type).lower()
            self.whole_word_masking = getattr(config, "whole_word_masking", self.whole_word_masking)
            self.max_predictions_per_seq = getattr(config, "max_predictions_per_seq", self.max_predictions_per_seq)
            self.nsp_probability = getattr(config, "nsp_probability", self.nsp_probability)
    
    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        model_adapter: ModelAdapter,
        global_step: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a training step.
        
        Args:
            model: Model to train
            batch: Input batch
            model_adapter: Model adapter instance
            global_step: Current global step
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Prepare batch for model
        prepared_batch = model_adapter.prepare_batch(batch)
        prepared_batch = model_adapter.move_to_device(prepared_batch)
        
        # Apply pretraining modifications to the batch
        if self.pretraining_type == "mlm":
            training_batch = self._prepare_mlm_batch(prepared_batch, model_adapter)
        elif self.pretraining_type == "clm":
            # For causal language modeling, no special preparation needed
            training_batch = prepared_batch
        elif self.pretraining_type == "mlm_nsp":
            training_batch = self._prepare_mlm_nsp_batch(prepared_batch, model_adapter)
        else:
            logger.warning(f"Unknown pretraining type: {self.pretraining_type}, using default")
            training_batch = prepared_batch
        
        # Forward pass
        outputs = model_adapter.forward(model, training_batch)
        
        # Compute loss
        loss = model_adapter.compute_loss(outputs, training_batch)
        
        # Extract metrics
        metrics = {"loss": loss.item()}
        
        # Add specific metrics if available
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.endswith("_loss") and isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
        
        return loss, metrics
    
    def validation_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        model_adapter: ModelAdapter
    ) -> Dict[str, float]:
        """
        Perform a validation step.
        
        Args:
            model: Model to validate
            batch: Input batch
            model_adapter: Model adapter instance
            
        Returns:
            Dictionary of validation metrics
        """
        # Prepare batch for model
        prepared_batch = model_adapter.prepare_batch(batch, is_train=False)
        prepared_batch = model_adapter.move_to_device(prepared_batch)
        
        # For validation, we use original unmasked inputs
        # Forward pass
        with torch.no_grad():
            outputs = model_adapter.forward(model, prepared_batch)
            loss = model_adapter.compute_loss(outputs, prepared_batch)
        
        # Extract metrics
        metrics = {"loss": loss.item()}
        
        # Add specific metrics if available
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.endswith("_loss") and isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
        
        return metrics
    
    def _prepare_mlm_batch(
        self,
        batch: Dict[str, torch.Tensor],
        model_adapter: ModelAdapter
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for masked language modeling.
        
        Args:
            batch: Input batch
            model_adapter: Model adapter instance
            
        Returns:
            Batch modified for MLM
        """
        # Create a new batch dictionary to avoid modifying the original
        mlm_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get tokenizer from model adapter
        tokenizer = model_adapter.get_tokenizer()
        
        if tokenizer is None:
            logger.warning("No tokenizer available for MLM, skipping masking")
            return mlm_batch
        
        # Get special token ids
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            mask_token = getattr(tokenizer, "mask_token", "[MASK]")
            mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        
        if mask_token_id is None:
            logger.warning("No mask token ID available, using a default value")
            mask_token_id = 103  # Default for BERT
        
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0  # Default padding token ID
        
        # Get input ids and create labels
        input_ids = mlm_batch["input_ids"].clone()
        
        # Create labels - copy input_ids
        labels = input_ids.clone()
        
        # Create attention mask if not present
        if "attention_mask" not in mlm_batch:
            attention_mask = (input_ids != pad_token_id).long()
            mlm_batch["attention_mask"] = attention_mask
        else:
            attention_mask = mlm_batch["attention_mask"]
        
        # Only mask where attention mask is 1
        masked_indices = torch.bernoulli(
            torch.full(input_ids.shape, self.mlm_probability)
        ).bool() & (attention_mask == 1)
        
        # Limit the number of masked tokens per sequence
        if self.max_predictions_per_seq > 0:
            for i in range(input_ids.size(0)):
                indices = masked_indices[i].nonzero(as_tuple=True)[0]
                if len(indices) > self.max_predictions_per_seq:
                    # Randomly select max_predictions_per_seq indices
                    indices = indices[torch.randperm(len(indices))[:self.max_predictions_per_seq]]
                    masked_indices[i] = torch.zeros_like(masked_indices[i])
                    masked_indices[i, indices] = True
        
        # Whole word masking if enabled
        if self.whole_word_masking and "word_ids" in mlm_batch:
            word_ids = mlm_batch["word_ids"]
            for i in range(input_ids.size(0)):
                for word_id in torch.unique(word_ids[i]):
                    if word_id == -100:  # Special tokens
                        continue
                    
                    # Get indices of tokens for this word
                    word_idx = (word_ids[i] == word_id).nonzero(as_tuple=True)[0]
                    
                    # Determine if any token in this word is masked
                    if masked_indices[i, word_idx].any():
                        # Mask all tokens in this word
                        masked_indices[i, word_idx] = True
        
        # Get values to replace masked tokens with
        # - 80% of the time, replace with [MASK]
        # - 10% of the time, replace with random word
        # - 10% of the time, keep as is (but still compute loss)
        rand = torch.rand(input_ids.shape, device=input_ids.device)
        
        # Mask tokens - set to [MASK] token id
        mask_indices = masked_indices & (rand < 0.8)
        mlm_batch["input_ids"][mask_indices] = mask_token_id
        
        # Replace with random token 10% of the remaining time
        random_indices = masked_indices & (rand >= 0.8) & (rand < 0.9)
        random_words = torch.randint(
            low=0, high=tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 30000, 
            size=input_ids.shape, device=input_ids.device
        )
        mlm_batch["input_ids"][random_indices] = random_words[random_indices]
        
        # Rest 10% remains the same but still included in loss
        
        # Set labels: -100 for non-masked tokens
        labels[~masked_indices] = -100
        mlm_batch["labels"] = labels
        
        return mlm_batch
    
    def _prepare_mlm_nsp_batch(
        self,
        batch: Dict[str, torch.Tensor],
        model_adapter: ModelAdapter
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for masked language modeling and next sentence prediction.
        
        Args:
            batch: Input batch
            model_adapter: Model adapter instance
            
        Returns:
            Batch modified for MLM and NSP
        """
        # First apply MLM preprocessing
        mlm_batch = self._prepare_mlm_batch(batch, model_adapter)
        
        # Check if the batch has next sentence prediction data
        if "next_sentence_label" not in batch and "token_type_ids" not in batch:
            logger.warning("Batch does not contain NSP data, skipping NSP preprocessing")
            return mlm_batch
        
        # For NSP, we need token_type_ids and next_sentence_label
        # If token_type_ids is present but next_sentence_label is not, create random labels
        if "token_type_ids" in batch and "next_sentence_label" not in batch:
            batch_size = batch["input_ids"].size(0)
            # Generate random NSP labels: 0 = next sentence, 1 = random sentence
            nsp_labels = torch.randint(0, 2, (batch_size,), device=batch["input_ids"].device)
            mlm_batch["next_sentence_label"] = nsp_labels
        
        return mlm_batch
    
    def collate_fn(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for pretraining data.
        
        Args:
            examples: List of examples to collate
            tokenizer: Tokenizer to use
            
        Returns:
            Collated batch
        """
        # Choose collation based on pretraining type
        if self.pretraining_type == "mlm_nsp":
            return self._collate_mlm_nsp(examples, tokenizer)
        elif self.pretraining_type == "mlm":
            return self._collate_mlm(examples, tokenizer)
        else:
            # Default collation
            return self._default_collate(examples, tokenizer)
    
    def _collate_mlm(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for MLM pretraining.
        
        Args:
            examples: List of examples to collate
            tokenizer: Tokenizer to use
            
        Returns:
            Collated batch
        """
        # Extract text from examples
        if all(isinstance(ex, dict) and "text" in ex for ex in examples):
            texts = [ex["text"] for ex in examples]
        elif all(isinstance(ex, str) for ex in examples):
            texts = examples
        else:
            # Try to handle various formats
            texts = []
            for ex in examples:
                if isinstance(ex, str):
                    texts.append(ex)
                elif isinstance(ex, dict):
                    text = ex.get("text", ex.get("input_text", ex.get("source", str(ex))))
                    texts.append(text)
                else:
                    texts.append(str(ex))
        
        # Tokenize
        batch = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True if self.pretraining_type == "mlm_nsp" else False
        )
        
        # For whole word masking, add word IDs
        if self.whole_word_masking:
            word_ids = []
            for text in texts:
                tokens = tokenizer.tokenize(text)
                word_id = 0
                ids = []
                
                for token in tokens:
                    # Check if this is a new word
                    if token.startswith("##") or not token.isalnum():
                        # Continuation of previous word
                        ids.append(word_id)
                    else:
                        # New word
                        word_id += 1
                        ids.append(word_id)
                
                # Pad to match sequence length
                while len(ids) < len(batch["input_ids"][0]):
                    ids.append(-100)  # Special value for padding
                
                word_ids.append(ids[:len(batch["input_ids"][0])])
            
            batch["word_ids"] = torch.tensor(word_ids, dtype=torch.long)
        
        return batch
    
    def _collate_mlm_nsp(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for MLM+NSP pretraining.
        
        Args:
            examples: List of examples to collate
            tokenizer: Tokenizer to use
            
        Returns:
            Collated batch
        """
        # For NSP, we need pairs of sentences
        sentence_pairs = []
        nsp_labels = []
        
        # Process examples to create sentence pairs and NSP labels
        for ex in examples:
            if isinstance(ex, dict) and "sentences" in ex:
                # Example has explicit sentences
                sentences = ex["sentences"]
                if len(sentences) >= 2:
                    if random.random() < self.nsp_probability:
                        # Use a random sentence as the second sentence
                        sentence_a = sentences[0]
                        random_idx = random.randint(0, len(examples) - 1)
                        random_ex = examples[random_idx]
                        if isinstance(random_ex, dict) and "sentences" in random_ex:
                            random_sentences = random_ex["sentences"]
                            if random_sentences:
                                sentence_b = random.choice(random_sentences)
                            else:
                                sentence_b = sentences[1]
                        else:
                            sentence_b = sentences[1]
                        nsp_label = 1  # Not next sentence
                    else:
                        # Use the actual next sentence
                        sentence_a = sentences[0]
                        sentence_b = sentences[1]
                        nsp_label = 0  # Is next sentence
                    
                    sentence_pairs.append((sentence_a, sentence_b))
                    nsp_labels.append(nsp_label)
            elif isinstance(ex, str):
                # Split example into sentences
                text_sentences = ex.split('.')
                if len(text_sentences) >= 2:
                    sentence_a = text_sentences[0] + '.'
                    if random.random() < self.nsp_probability:
                        # Use a random sentence as the second sentence
                        random_idx = random.randint(0, len(examples) - 1)
                        random_ex = examples[random_idx]
                        if isinstance(random_ex, str):
                            random_sentences = random_ex.split('.')
                            if random_sentences:
                                sentence_b = random.choice(random_sentences) + '.'
                            else:
                                sentence_b = text_sentences[1] + '.'
                        else:
                            sentence_b = text_sentences[1] + '.'
                        nsp_label = 1  # Not next sentence
                    else:
                        # Use the actual next sentence
                        sentence_b = text_sentences[1] + '.'
                        nsp_label = 0  # Is next sentence
                    
                    sentence_pairs.append((sentence_a, sentence_b))
                    nsp_labels.append(nsp_label)
        
        # Tokenize sentence pairs
        if sentence_pairs:
            batch = tokenizer(
                [pair[0] for pair in sentence_pairs],
                [pair[1] for pair in sentence_pairs],
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=True
            )
            
            # Add NSP labels
            batch["next_sentence_label"] = torch.tensor(nsp_labels, dtype=torch.long)
            
            return batch
        else:
            # Fall back to MLM only
            logger.warning("Could not create sentence pairs for NSP, falling back to MLM only")
            return self._collate_mlm(examples, tokenizer)
    
    def _default_collate(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Default collate function for pretraining.
        
        Args:
            examples: List of examples to collate
            tokenizer: Tokenizer to use
            
        Returns:
            Collated batch
        """
        # Extract text from examples
        texts = []
        for ex in examples:
            if isinstance(ex, str):
                texts.append(ex)
            elif isinstance(ex, dict):
                text = ex.get("text", ex.get("input_text", ex.get("source", str(ex))))
                texts.append(text)
            else:
                texts.append(str(ex))
        
        # Tokenize
        batch = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        return batch
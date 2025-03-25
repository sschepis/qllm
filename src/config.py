"""
Configuration for the Semantic Resonance Language Model.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Dict


@dataclass
class ModelConfig:
    # General model settings
    vocab_size: int = 30000
    max_seq_length: int = 512
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1
    
    # Prime Hilbert Encoder settings
    primes: List[int] = None
    base_dim: int = 768
    
    # Resonance Block settings
    max_iterations: int = 10
    entropy_threshold: float = 0.1
    use_prime_mask: bool = True
    
    # HCW (Self-Evolving Memory) settings
    memory_size: int = 1000
    memory_key_dim: int = 128
    enable_hcw: bool = True
    
    # Pre-Manifest Resonance Layer settings
    pre_manifest_iterations: int = 5
    pre_manifest_entropy_threshold: float = 0.05
    
    def __post_init__(self):
        if self.primes is None:
            # Default set of primes for subspace decomposition
            self.primes = [7, 11, 13, 17, 19]
        
        # Calculate the total embedding dimension from prime subspaces
        self.embedding_dim = sum(self.primes)


@dataclass
class TrainingConfig:
    # Training hyperparameters
    batch_size: int = 32
    accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    
    # Optimizer settings
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Loss settings
    entropy_regularization_weight: float = 0.01
    adapter_l2_penalty: float = 0.001
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Training device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0


@dataclass
class DataConfig:
    # Dataset settings
    dataset_name: str = "wikitext-103-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Preprocessing settings
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    stride: int = 256
    
    # Caching
    cache_dir: str = ".cache"
    preprocessing_num_workers: int = 4


@dataclass
class EvalConfig:
    # Evaluation settings
    batch_size: int = 16
    metrics: List[str] = None
    
    # Baseline model for comparison
    baseline_model: str = "gpt2"
    baseline_model_size: str = "small"  # small, medium, or large
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["perplexity", "accuracy"]


def get_default_configs():
    """Return default configurations for the Semantic Resonance model."""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "eval": EvalConfig()
    }
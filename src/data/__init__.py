"""
Data package for QLLM.

This package contains datasets, dataloaders, and utilities for
handling data in the Quantum Resonance Language Model.
"""

from src.data.dataloaders import get_wikitext_dataloaders
from src.data.dialogue_dataset import DialogueDataset
from src.data.tensor_collate import (
    dialogue_collate_fn,
    default_collate_fn,
    tensor_collate_fn
)

__all__ = [
    'get_wikitext_dataloaders',
    'DialogueDataset',
    'dialogue_collate_fn',
    'default_collate_fn',
    'tensor_collate_fn',
]
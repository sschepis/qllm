"""
Memory Extension Types Module.

This module provides common type definitions for memory extensions.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
from collections import OrderedDict
import torch

# Type aliases
MemoryConfig = Dict[str, Any]
MemoryStats = Dict[str, Any]
MemoryIndices = torch.Tensor
MemoryValues = torch.Tensor
MemoryMetadata = Optional[Dict[str, Any]]
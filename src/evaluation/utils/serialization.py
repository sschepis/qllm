"""
Serialization utilities for QLLM evaluation.

This module provides functions for serializing evaluation results
and ensuring they can be properly stored in JSON format.
"""

import datetime
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union


def make_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    elif hasattr(obj, "__dict__"):
        return make_serializable(obj.__dict__)
    elif hasattr(obj, "__iter__"):
        return [make_serializable(item) for item in obj]
    else:
        return str(obj)


def save_results_to_json(results: Dict[str, Any], file_path: str) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results
        file_path: Path to save results
    """
    import os
    import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert results to serializable format
    serializable_results = make_serializable(results)
    
    # Add timestamp
    serializable_results["timestamp"] = datetime.datetime.now().isoformat()
    
    # Write to file
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {file_path}")


def load_results_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded results
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def create_timestamped_filename(base_dir: str, prefix: str, extension: str = "json") -> str:
    """
    Create a filename with a timestamp.
    
    Args:
        base_dir: Base directory
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Path with timestamp
    """
    import os
    import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create filename
    filename = f"{prefix}_{timestamp}.{extension}"
    
    return os.path.join(base_dir, filename)
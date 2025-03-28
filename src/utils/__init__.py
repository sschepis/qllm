"""
Utility modules for QLLM.

This package provides various utility functions for the QLLM framework,
organized into specialized modules.
"""

# Import and export key utilities from compression module
from src.utils.compression import (
    create_prime_resonance_mask,
    prime_importance_pruning,
    compress_model,
    load_compressed_model,
    compare_models
)

# Import and export key utilities from config module
from src.utils.config import (
    load_config_file,
    save_config_file,
    merge_configs,
    validate_config_schema,
    get_config_path,
    update_nested_dict
)

# Import and export key utilities from device module
from src.utils.device import (
    get_default_device,
    get_device,
    get_device_info,
    print_device_info,
    move_to_device,
    get_memory_usage,
    set_cuda_device_environment
)

# Import and export key utilities from logging module
from src.utils.logging import (
    setup_logger,
    get_logger,
    TrainingLogger,
    log_training_progress,
    log_evaluation_results
)

# Import and export key utilities from checkpoint module
from src.utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    get_checkpoint_type,
    create_model_archive
)

# Import and export key utilities from generation module
from src.utils.generation_utils import (
    generate_text,
    beam_search,
    sample_with_temperature,
    nucleus_sampling,
    token_probabilities,
    top_k_top_p_filtering
)

# Import and export key utilities from visualization module
from src.utils.visualization_utils import (
    setup_plotting,
    create_figure,
    save_figure,
    plot_training_curves,
    plot_confusion_matrix,
    plot_attention_heatmap,
    plot_embedding_clusters,
    create_interactive_dashboard
)

__all__ = [
    # Compression utilities
    'create_prime_resonance_mask',
    'prime_importance_pruning',
    'compress_model',
    'load_compressed_model',
    'compare_models',
    
    # Config utilities
    'load_config_file',
    'save_config_file',
    'merge_configs',
    'validate_config_schema', 
    'get_config_path',
    'update_nested_dict',
    
    # Device utilities
    'get_default_device',
    'get_device',
    'get_device_info',
    'print_device_info',
    'move_to_device',
    'get_memory_usage',
    'set_cuda_device_environment',
    
    # Logging utilities
    'setup_logger',
    'get_logger',
    'TrainingLogger',
    'log_training_progress',
    'log_evaluation_results',
    
    # Checkpoint utilities
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'get_checkpoint_type',
    'create_model_archive',
    
    # Generation utilities
    'generate_text',
    'beam_search',
    'sample_with_temperature',
    'nucleus_sampling',
    'token_probabilities',
    'top_k_top_p_filtering',
    
    # Visualization utilities
    'setup_plotting',
    'create_figure',
    'save_figure',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_attention_heatmap',
    'plot_embedding_clusters',
    'create_interactive_dashboard'
]
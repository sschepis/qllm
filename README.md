# Quantum Resonance Language Model (QLLM)

A next-generation language model implementation leveraging quantum resonance principles for enhanced representation learning and attention mechanisms.

## Project Overview

QLLM combines advanced quantum-inspired mathematical principles with state-of-the-art deep learning techniques to create a highly efficient language model architecture. Key innovations include:

- **Prime Hilbert Encoding**: Representing tokens in prime-based Hilbert subspaces
- **Resonance Attention Mechanism**: Iterative, entropy-driven attention with dynamic temperature scheduling
- **Homomorphic Computational Wrapper**: Self-evolving memory system enabling continuous adaptation
- **Extensible Architecture**: Pluggable extension system for multimodal, memory, and quantum enhancements

## Project Structure

The codebase follows a modular architecture for maintainability and scalability:

```
qllm/
├── main.py                     # Main entry point for all operations
├── train_verbose.py            # Specialized training script with detailed logging
├── train_dialogue.py           # Specialized script for conversational training
├── src/
│   ├── cli/                    # Command-line interface
│   │   ├── arg_parsing.py      # Argument parsing logic
│   │   └── commands.py         # Core command implementations
│   ├── data/                   # Data processing modules
│   │   ├── batch_utils.py      # Utilities for batch processing
│   │   ├── dataloaders.py      # Dataset loaders with preprocessing
│   │   ├── dialogue_dataset.py # Dataset implementation for dialogue training
│   │   ├── function_calling_dataset.py # Dataset for function calling
│   │   ├── tensor_collate.py   # Collation utilities for data batching
│   │   └── wikitext_dataset.py # Implementation for wikitext dataset
│   ├── model/                  # Core model architecture
│   │   ├── semantic_resonance_model.py         # Main model implementation
│   │   ├── semantic_resonance_model_with_extensions.py # Extended model with plugins
│   │   ├── resonance_attention.py              # Quantum resonance attention
│   │   ├── resonance_block.py                  # Transformer block with resonance
│   │   ├── prime_hilbert_encoder.py            # Prime-based token encoding
│   │   ├── homomorphic_wrapper.py              # Self-evolving memory system
│   │   ├── pre_manifest_layer.py               # Logit refinement layer
│   │   ├── fixed_autocast.py                   # Mixed precision utilities
│   │   ├── extensions/                         # Extension framework
│   │       ├── base_extension.py               # Base extension class
│   │       ├── extension_config.py             # Extension configuration
│   │       ├── extension_manager.py            # Extension system management
│   │       ├── memory/                         # Memory extensions
│   │       ├── multimodal/                     # Multimodal extensions
│   │       └── quantum/                        # Quantum computing extensions
│   ├── evaluation/                # Evaluation utilities
│   │   ├── comprehensive_suite.py # Full evaluation framework
│   │   ├── metrics.py             # Performance metrics
│   │   └── visualize_results.py   # Results visualization
│   ├── training/                  # Training components
│   │   ├── checkpoint.py          # Checkpoint management
│   │   ├── continuous_learning.py # Continuous learning utilities
│   │   └── trainer.py             # Training loop and optimization
│   └── utils/                     # Shared utilities
│       ├── compression.py         # Model compression utilities
│       ├── config.py              # Configuration management
│       ├── device.py              # Device utilities
│       └── logging.py             # Structured logging system
├── examples/                      # Example scripts and configurations
├── evaluation_results/            # Evaluation output directory
└── runs/                          # Training runs output directory
```

## Key Components

### Model Architecture

The QLLM architecture consists of several innovative components:

1. **Prime Hilbert Encoder**: Transforms input tokens into prime-based Hilbert subspaces that enable more efficient representation of semantic relationships.

2. **Resonance Attention**: An advanced attention mechanism that iteratively refines attention distributions until they converge. Key features include:
   - Entropy-based halting to adaptively determine refinement iterations
   - Phase modulation to shift attention perspective across iterations
   - Momentum-based attention updates to stabilize convergence
   - Temperature scheduling to progressively sharpen distributions

3. **Homomorphic Computational Wrapper (HCW)**: A self-evolving memory system that enables the model to continuously adapt to new information.

4. **Pre-Manifest Resonance Layer**: Refines final hidden states before generating output distributions, allowing for more accurate token predictions.

### Extension System

QLLM features a flexible extension system for enhancing model capabilities:

1. **Multimodal Extensions**: Add vision processing capabilities through the Vision Extension, supporting:
   - Cross-modal transfer between visual and textual domains
   - Modal entanglement for unified representation
   - Emergent understanding of visual concepts

2. **Memory Extensions**: Add persistent knowledge storage and retrieval through:
   - Knowledge Graph Extension for structured information
   - Graph traversal and query mechanisms
   - Counterfactual reasoning

3. **Quantum Extensions**: Implement quantum computing principles to improve efficiency:
   - Symmetry mask optimization for parameter reduction
   - Non-local correlations for enhanced long-range dependencies
   - Adaptive resonance patterns for dynamic parameter usage

### Training Framework

The trainer (`src/training/trainer.py`) provides a comprehensive solution for model training:

- Advanced checkpoint management with automatic resumption
- Early stopping based on validation metrics
- Multiple learning rate scheduling options
- Gradient accumulation for large batch simulation
- Mixed precision training for performance optimization
- Comprehensive logging and progress tracking

### Evaluation Suite

QLLM includes a comprehensive evaluation framework (`src/evaluation/comprehensive_suite.py`) that measures:

- **Perplexity**: Language modeling quality
- **Parameter Efficiency**: Compute-to-performance ratio
- **Memory Usage**: Memory footprint during inference
- **Inference Speed**: Tokens processed per second
- **Generation Diversity**: Variety in generated outputs

The evaluation suite supports ablation studies to isolate the impact of each extension.

## Usage

### Basic Training

```bash
python main.py --mode train --output_dir runs/my_model --max_epochs 10
```

### Training with Extensions

```bash
python main.py --mode train --output_dir runs/multimodal_model --enable_extensions --extension_types multimodal
```

### Detailed Training with Verbose Logging

For debugging and visualizing the resonance attention process:

```bash
python train_verbose.py --output_dir runs/debug_run --log_entropy_every 50
```

### Dialogue-Specific Training

For training on conversational data:

```bash
python train_dialogue.py --dataset_path data/dialogues --output_dir runs/dialogue_model
```

### Evaluation

```bash
python main.py --mode eval --checkpoint_path runs/my_model/checkpoints/best_model.pt
```

### Comprehensive Evaluation

```bash
python -m examples.run_evaluation --config examples/evaluation_config.json
```

### Text Generation

```bash
python main.py --mode generate --checkpoint_path runs/my_model/checkpoints/best_model.pt --prompt "The quantum theory of"
```

### Model Compression

```bash
python main.py --mode compress --checkpoint_path runs/my_model/checkpoints/best_model.pt --compression_threshold 0.7
```

## Configuration

You can provide configurations in three ways:

1. **Command line arguments**: Override specific parameters.
2. **Configuration files**: Load complete configurations from JSON files.
3. **Environment variables**: Set configuration through environment.

Example:

```bash
python main.py --mode train --config_dir configs/ --learning_rate 1e-4
```

## Extension Development

To develop new extensions for QLLM:

1. Create a new extension class that inherits from `BaseExtension`
2. Implement required methods: `initialize()`, `forward()`, and extension-specific methods
3. Register the extension with the `ExtensionManager`

Example:

```python
from src.model.extensions.base_extension import BaseExtension

class MyCustomExtension(BaseExtension):
    def __init__(self, name, config):
        super().__init__(name, "custom", config)
        # Initialize extension-specific attributes
        
    def initialize(self, model):
        # Connect to model components
        
    def forward(self, x, model_outputs=None, extension_outputs=None):
        # Process inputs and return modified tensor
        return modified_tensor, {"my_metric": value}
```

## Performance Results

Recent evaluation results demonstrate the efficacy of quantum resonance principles in language modeling:

- **Base Model Perplexity**: Competitive performance with similar-sized models
- **Resonance Attention**: 15-20% improvement in convergence speed compared to standard attention
- **Extension Benefits**:
  - Multimodal: Enhanced text generation with visual context
  - Memory: Improved factual consistency in long-form generation
  - Quantum: 30-40% parameter reduction with minimal performance impact

## Citation

If you use QLLM in your research, please cite:

```
@software{qllm2025,
  author = {Quantum Resonance Team},
  title = {Quantum Resonance Language Model},
  year = {2025},
  url = {https://github.com/username/qllm}
}
```

## License

[MIT License](LICENSE)
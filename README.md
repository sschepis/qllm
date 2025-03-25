# Quantum Resonance Language Model (QLLM)

A language model implementation that uses quantum resonance principles for improved attention mechanisms.

## Project Structure

The codebase has been refactored into a modular structure for better organization and maintainability:

```
qllm/
├── main.py                 # Main entrypoint for all operations
├── train_verbose.py        # Specialized training script with detailed logging
├── src/
│   ├── cli/                # Command-line interface utilities
│   │   ├── arg_parsing.py  # Argument parsing for all commands
│   │   └── commands.py     # Command implementations
│   ├── data/               # Data handling
│   │   └── dataloaders.py  # Dataset loading and processing
│   ├── model/              # Model architecture
│   │   ├── semantic_resonance_model.py
│   │   └── resonance_attention.py
│   ├── training/           # Training utilities
│   │   ├── checkpoint.py   # Robust checkpoint handling
│   │   ├── trainer.py      # Core training loop
│   │   └── evaluator.py    # Model evaluation
│   └── utils/              # Shared utilities
│       ├── config.py       # Configuration management
│       ├── device.py       # Device utilities
│       ├── logging.py      # Structured logging
│       └── compression.py  # Model compression utilities
└── runs/                   # Training runs output directory
```

## Key Components

### Command Line Interface

The CLI system is designed to handle four primary operations:

- **Training**: `python main.py --mode train`
- **Evaluation**: `python main.py --mode eval`
- **Text Generation**: `python main.py --mode generate`
- **Model Compression**: `python main.py --mode compress`

### Model Architecture

The model uses a transformer architecture with specialized quantum resonance attention mechanisms:

- **Semantic Resonance Model**: Main model implementation
- **Resonance Attention**: Custom attention mechanism that iteratively refines attention distributions

### Trainer

The trainer (`src/training/trainer.py`) provides a unified interface for model training with:

- Checkpoint management
- Early stopping
- Learning rate scheduling
- Mixed precision training
- Progress tracking

### Configuration System

Configurations use typed dataclasses for type safety and validation:

- **ModelConfig**: Architecture parameters
- **TrainingConfig**: Training hyperparameters
- **DataConfig**: Dataset and tokenization settings 
- **GenerationConfig**: Text generation parameters

## Usage

### Basic Training

```bash
python main.py --mode train --output_dir runs/my_model --max_epochs 10
```

### Detailed Training with Verbose Logging

For debugging and visualizing the resonance attention process:

```bash
python train_verbose.py --output_dir runs/debug_run --log_entropy_every 50
```

### Evaluation

```bash
python main.py --mode eval --checkpoint_path runs/my_model/checkpoints/best_model.pt
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

## Checkpoint Management

The project includes robust checkpoint handling that can:

- Automatically find and load the latest checkpoint
- Resume training from interruptions
- Handle disk space errors with fallback mechanisms
- Load partial checkpoints (model-only)

## Extending the Codebase

To add new features:

1. **New model architectures**: Add to `src/model/`
2. **Custom datasets**: Extend `src/data/dataloaders.py`
3. **Training variations**: Create specialized trainers in `src/training/`
4. **CLI commands**: Add to `src/cli/commands.py` and update argument parsing
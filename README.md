# Semantic Resonance Language Model

This repository contains a PyTorch implementation of the Semantic Resonance Language Model, a next-generation language model architecture that integrates semantic resonance principles with transformer-based language modeling.

## Overview

The Semantic Resonance Language Model introduces several innovative components:

1. **Prime Hilbert Encoder**: Maps tokens and positions into prime-based subspaces for efficient representation
2. **Resonance Blocks**: Implements iterative attention with entropy-based halting
3. **Self-Evolving Memory (HCW)**: Enables continuous adaptation without catastrophic forgetting
4. **Model Compression**: Uses prime resonance masking for parameter-efficient models

The model aims to achieve:
- **High Learning Speed**: Requiring fewer parameters/data to reach low perplexity
- **Continuous Self-Evolution**: Adapting to new information on the fly
- **Compression**: Reducing model size while maintaining performance

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Datasets
- TensorBoard

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semantic-resonance-lm.git
   cd semantic-resonance-lm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
semantic-resonance-lm/
│
├── main.py                 # Main entry point for all operations
├── src/
│   ├── config.py           # Configuration classes
│   ├── train.py            # Training script
│   ├── model/              # Model components
│   │   ├── prime_hilbert_encoder.py
│   │   ├── resonance_attention.py
│   │   ├── resonance_block.py
│   │   ├── homomorphic_wrapper.py
│   │   ├── pre_manifest_layer.py
│   │   └── semantic_resonance_model.py
│   ├── data/               # Data processing
│   │   └── wikitext_dataset.py
│   ├── training/           # Training infrastructure
│   │   └── trainer.py
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py
│   └── utils/              # Utility functions
│       └── compression.py
└── requirements.txt        # Project dependencies
```

## Usage

The implementation provides a unified interface through `main.py` with different modes of operation:

### Training

Train a new model on the WikiText-103 dataset:

```bash
python main.py --mode train --output_dir runs/my_model --batch_size 32 --max_epochs 10
```

Advanced configuration:

```bash
python main.py --mode train --hidden_dim 768 --num_layers 6 --num_heads 12 \
    --primes 7 11 13 17 19 --learning_rate 5e-5 --batch_size 32 \
    --enable_hcw --output_dir runs/custom_model
```

Resume training from a checkpoint:

```bash
python main.py --mode train --resume --model_path runs/my_model/checkpoints/best_model.pt \
    --output_dir runs/my_model_continued
```

### Evaluation

Evaluate a trained model on the validation or test set:

```bash
python main.py --mode eval --model_path runs/my_model --eval_split validation
```

### Model Compression

Apply prime resonance compression to a trained model:

```bash
python main.py --mode compress --model_path runs/my_model \
    --compression_threshold 0.8 --compression_method both \
    --output_dir runs/compressed_model
```

### Text Generation

Generate text using a trained model:

```bash
python main.py --mode generate --model_path runs/my_model \
    --prompt "Once upon a time" --max_length 100 \
    --temperature 0.7 --top_k 50 --top_p 0.95
```

## Key Components

### Prime Hilbert Encoder

The Prime Hilbert Encoder maps tokens and positions into prime-based subspaces. Each token and position is represented across multiple subspaces with dimensions defined by distinct primes.

```python
# Example:
encoder = PrimeHilbertEncoder(
    vocab_size=30000,
    primes=[7, 11, 13, 17, 19],
    base_dim=768
)
```

### Resonance Attention

Implements multi-head attention with iterative refinement, stopping when the entropy of the attention distribution falls below a threshold.

```python
# Example:
attention = ResonanceAttention(
    hidden_dim=768,
    num_heads=12,
    max_iterations=10,
    epsilon=0.1
)
```

### Homomorphic Computational Wrapper (HCW)

Enables the model to update its knowledge continuously by generating contextual weight deltas based on new information.

```python
# Example:
hcw = HomomorphicComputationalWrapper(
    hidden_dim=768,
    memory_size=1000,
    key_dim=128
)
```

### Prime Resonance Compression

Reduces model size by applying structured sparsity guided by prime-based frequency analysis.

```python
# Apply compression:
compressed_model, ratio = compress_model(
    model,
    {"method": "both", "primes": [7, 11, 13, 17, 19], "threshold": 0.8}
)
```

## Performance

The implementation aims to achieve:

- Comparable perplexity to larger models with significantly fewer parameters
- Faster convergence during training
- Continuous adaptation to new data with minimal forgetting

## Citations

If you use this implementation in your research, please cite:

```
@article{semantic_resonance_lm,
  title={A Next-Generation Knowledge Model Leveraging Semantic Resonance for Efficient Language Understanding},
  author={Schepis, Sebastian},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
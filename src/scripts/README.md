# QLLM Training Script

This directory contains scripts for training, evaluating, and running QLLM models.

## Training Script (`train.py`)

The `train.py` script provides a comprehensive training pipeline for QLLM models, supporting:
- Standard text generation training
- Dialogue capabilities
- Function calling
- Memory extensions
- Multimodal capabilities

### Basic Usage

```bash
python src/scripts/train.py --train-data path/to/data --model-size mini --training-mode dialogue
```

### Common Training Configurations

#### Training a Mini Model for Dialogue

```bash
python src/scripts/train.py \
  --train-data data/dialogue_dataset.json \
  --eval-data data/dialogue_eval.json \
  --model-size mini \
  --training-mode dialogue \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --output-dir runs/dialogue_model
```

#### Training with Function Calling Support

```bash
python src/scripts/train.py \
  --train-data data/function_calling_dataset.json \
  --model-size small \
  --training-mode function_call \
  --function-schema-path data/function_schema.json \
  --batch-size 8 \
  --epochs 5 \
  --output-dir runs/function_model
```

#### Training a Large Model with Extensions

```bash
python src/scripts/train.py \
  --train-data data/combined_dataset.json \
  --model-size large \
  --training-mode unified \
  --enable-extensions \
  --enable-memory \
  --enable-multimodal \
  --extension-config configs/extensions.json \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --fp16 \
  --output-dir runs/large_model
```

### Key Parameters

#### Model Configuration
- `--model-size`: Size of the model to train (mini, small, medium, large, xlarge)
- `--model-type`: Type of model architecture (semantic_resonance, semantic_resonance_with_extensions)
- `--vocab-size`: Vocabulary size for the model
- `--max-seq-length`: Maximum sequence length for the model
- `--checkpoint-path`: Path to a checkpoint to load the model from

#### Training Configuration
- `--training-mode`: Training mode to use (text, dialogue, function_call, unified)
- `--batch-size`: Batch size for training
- `--eval-batch-size`: Batch size for evaluation
- `--epochs`: Number of epochs to train for
- `--learning-rate`: Learning rate for training
- `--weight-decay`: Weight decay for training
- `--warmup-steps`: Number of warmup steps for learning rate scheduler
- `--gradient-accumulation-steps`: Number of gradient accumulation steps
- `--save-steps`: Save checkpoint every X steps
- `--eval-steps`: Evaluate every X steps
- `--device`: Device to use for training (auto, cuda, cpu, mps)
- `--fp16`: Enable mixed precision training
- `--output-dir`: Directory to save checkpoints and logs

#### Data Configuration
- `--train-data`: Path to training data file or directory
- `--eval-data`: Path to evaluation data file or directory
- `--dataset-type`: Type of dataset to use (auto, text, dialogue, function_calling)
- `--max-history-turns`: Maximum number of history turns for dialogue data
- `--function-schema-path`: Path to function schema JSON file for function calling

#### Extension Configuration
- `--enable-extensions`: Enable model extensions
- `--enable-memory`: Enable memory extension
- `--enable-multimodal`: Enable multimodal extension
- `--enable-quantum`: Enable quantum extension
- `--extension-config`: Path to extension configuration JSON file

### Example Extension Configuration File

Create a JSON file (e.g., `configs/extensions.json`) with the following structure:

```json
{
  "multimodal": {
    "enabled": true,
    "vision_encoder_type": "resnet",
    "vision_encoder_model": "resnet50",
    "vision_embedding_dim": 768,
    "fusion_type": "attention"
  },
  "memory": {
    "enabled": true,
    "memory_size": 10000,
    "memory_key_dim": 128,
    "use_graph_structure": true,
    "persistence_enabled": true,
    "persistence_path": "memory/knowledge_graph.pkl"
  },
  "quantum": {
    "enabled": false
  },
  "feature_flags": {
    "multimodal.vision": true,
    "multimodal.audio": false,
    "memory.graph": true,
    "memory.persistence": true
  }
}
```

## Output and Checkpoints

The training script will create the following in your output directory:
- Model checkpoints saved at the frequency specified by `--save-steps`
- Training logs in `train.log`
- Configuration settings in `config.json`
- Training results in `training_results.json`

## Using the Trained Model

After training, you can use the model for:
1. Dialogue interactions
2. Function calling
3. Multimodal processing (if enabled)

Load the model and use it as follows:

```python
import torch
from src.model.semantic_resonance_model_with_extensions import SemanticResonanceModelWithExtensions

# Load the trained model
model = torch.load("runs/my_model/checkpoint-final.pt")

# Generate text
outputs = model.generate(
    "What is the capital of France?", 
    max_length=100,
    num_return_sequences=1
)

# For models with extensions
outputs = model.generate(
    "Describe this image:", 
    extension_kwargs={
        "multimodal": {
            "images": [image_tensor]
        }
    },
    max_length=200
)
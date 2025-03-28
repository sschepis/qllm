#!/bin/bash
# Examples of training QLLM models with different configurations

# Ensure script is run from project root directory
if [ ! -d "src" ] || [ ! -d "configs" ]; then
    echo "Error: This script must be run from the project root directory (where src/ exists)"
    echo "Please run this script as: ./train_examples.sh"
    exit 1
fi

# Check for Python and required packages
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Create the necessary directories
mkdir -p runs/mini_dialogue
mkdir -p runs/small_function
mkdir -p runs/medium_multimodal
mkdir -p memory

# Function to run a training example with error handling
run_example() {
    echo "=== $1 ==="
    if python3 "$2" "${@:3}"; then
        echo "✅ Example completed successfully"
    else
        echo "❌ Example failed with exit code $?"
    fi
    echo ""
}

# Example 1: Mini dialogue model with extensions
run_example "Training a mini dialogue model with extensions" src/scripts/train.py \
  --train-data examples/simple_dialogue.json \
  --model-size mini \
  --model-type semantic_resonance_with_extensions \
  --training-mode dialogue \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --output-dir runs/mini_dialogue

# # Example 2: Small model with function calling
# run_example "Training a small model with function calling" src/scripts/train.py \
#   --train-data examples/function_calling_data.json \
#   --model-size small \
#   --training-mode function_call \
#   --function-schema-path examples/function_calling_data.json \
#   --batch-size 8 \
#   --epochs 5 \
#   --output-dir runs/small_function

# # Example 3: Medium model with multimodal and memory extensions
# run_example "Training a medium model with multimodal and memory extensions" src/scripts/train.py \
#   --train-data examples/simple_dialogue.json \
#   --model-size medium \
#   --training-mode unified \
#   --enable-extensions \
#   --enable-memory \
#   --enable-multimodal \
#   --extension-config configs/extensions.json \
#   --batch-size 4 \
#   --gradient-accumulation-steps 4 \
#   --fp16 \
#   --output-dir runs/medium_multimodal

echo "All examples completed!"
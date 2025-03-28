#!/bin/bash
# Train a model using the Daily Dialog dataset

# Make sure script is executable
chmod +x train_daily_dialog.py

# Create output directory
mkdir -p runs/daily_dialog_mini

# Run the training script
python train_daily_dialog.py \
  --model-size mini \
  --batch-size 4 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --max-length 256 \
  --max-history-turns 3 \
  --system-prompt "You are a helpful assistant." \
  --device cpu \
  --output-dir runs/daily_dialog_mini
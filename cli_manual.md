# QLLM CLI User Manual

## Introduction

The Quantum Resonance Language Model Command Line Interface (QLLM CLI) provides a user-friendly, menu-driven interface for configuring, training, evaluating, and utilizing the QLLM. This manual covers the key features and usage instructions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages installed via `pip install -r requirements.txt`

### Starting the CLI

To start the CLI, run:

```bash
./cli.py
```

This will launch the main menu with arrow key navigation.

## Navigation

The CLI uses a modern, arrow key-based navigation system:

- **Up/Down Arrow Keys (↑/↓)** - Navigate through menu options
- **Enter Key** - Select the highlighted option
- **Number Keys (1-9)** - Quick select options by their number
- **Ctrl+C** - Exit any screen or menu

Each menu displays a blue highlight on your current selection to make navigation clear.

## Main Menu

The main menu provides access to all QLLM functionality:

1. **Training** - Configure and run model training
2. **Evaluation** - Evaluate model performance
3. **Generation** - Generate text with trained models
4. **Extensions** - Configure model extensions
5. **Compression** - Compress models for efficiency
6. **Exit** - Exit the CLI

## Training

The Training menu allows you to configure and run training jobs:

### Configure Training

This option launches the configuration wizard where you can set various training parameters:

- **Model Configuration** - Hidden dimensions, layers, attention heads, etc.
- **Training Configuration** - Batch size, learning rate, epochs, optimizer, etc.
- **Data Configuration** - Dataset, tokenizer, sequence length, etc.

Navigate through options using arrow keys and modify values as needed.

### Load Configuration

This option allows you to load a previously saved configuration:

1. Configurations are stored in the `configs/` directory
2. Select a configuration file from the list
3. The configuration will be loaded and ready for use

### Start Training

This option starts a training run with the current configuration:

1. The system will display a summary of the training settings
2. Confirm to start training
3. Training progress will be displayed in real-time with loss metrics
4. Training checkpoints will be saved based on your configuration

Training is synchronous and runs in the foreground with live updates.

### Resume Training

This option allows you to resume a previous training run:

1. Select a training run directory
2. Select a specific checkpoint to resume from
3. Training will continue from the chosen checkpoint

## Evaluation

The Evaluation menu allows you to assess model performance:

### Configure Evaluation

Set evaluation parameters such as metrics, test data, and batch size.

### Load Model

Load a trained model for evaluation.

### Run Evaluation

Run the evaluation with the loaded model and configured parameters.

### Visualize Results

Display evaluation results with various visualizations.

## Extensions

Extensions enhance model capabilities:

### Configure Extensions

Enable or disable extensions like Memory, Multimodal, and Quantum extensions.

### Memory Extensions

Configure memory-related parameters for enhanced context handling.

### Multimodal Extensions

Configure multimodal capabilities for handling different data types.

### Quantum Extensions

Configure quantum parameters for resonance optimization.

### List Enabled Extensions

Display all currently enabled extensions and their configurations.

## Example Workflows

### Basic Training Workflow

1. Start the CLI with `./cli.py`
2. Select **Training** from the main menu
3. Select **Configure Training** to set up parameters
4. After configuration, select **Start Training**
5. Review the training progress as it runs
6. When training completes, find model checkpoints in the specified output directory

### Using a Pre-Configured Training Setup

1. Start the CLI with `./cli.py`
2. Select **Training** from the main menu
3. Select **Load Configuration**
4. Choose `dialogue_config.json` for dialogue training
5. Select **Start Training** to begin training with this configuration

### Enabling Extensions

1. Start the CLI with `./cli.py`
2. Select **Extensions** from the main menu
3. Select **Configure Extensions**
4. Toggle extensions on/off as needed
5. Configure specific parameters for each enabled extension
6. Return to the main menu and proceed with training

## Configuration Files

QLLM uses JSON configuration files with three main sections:

### Model Configuration

Controls the model architecture:

```json
"model": {
  "hidden_dim": 768,
  "num_layers": 12,
  "num_heads": 12,
  "dropout": 0.1,
  "max_seq_length": 1024,
  ...
}
```

### Training Configuration

Controls the training process:

```json
"training": {
  "batch_size": 16,
  "eval_batch_size": 16,
  "learning_rate": 5e-5,
  "weight_decay": 0.01,
  "max_epochs": 3,
  "training_type": "standard",
  ...
}
```

### Data Configuration

Controls the dataset and preprocessing:

```json
"data": {
  "dataset_name": "wikitext",
  "tokenizer_name": "gpt2",
  "max_length": 512,
  ...
}
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Error: `ImportError: No module named X`
   - Solution: Run `pip install -r requirements.txt`

2. **CUDA Out of Memory**
   - Error: `RuntimeError: CUDA out of memory`
   - Solution: Decrease batch size or model size in configuration

3. **Dataset Not Found**
   - Error: `FileNotFoundError: No data file found`
   - Solution: Check the data file path in your configuration

### Getting Help

If you encounter any other issues, check the log files in your output directory for detailed error messages and stack traces.

## Advanced Usage

### Custom Datasets

To use custom datasets:

1. Place your data files in a known location
2. In the Training Configuration wizard, set the appropriate paths
3. For dialogue datasets, use the JSON format as shown in `examples/simple_dialogue.json`

### Extensions Development

Extensions can be customized by editing their configuration parameters. For advanced customization, refer to the extension-specific documentation in the `src/model/extensions/` directory.

## Conclusion

The QLLM CLI provides a comprehensive interface for working with the Quantum Resonance Language Model. By leveraging the menu-driven interface with arrow key navigation, you can efficiently configure, train, and utilize the model for various tasks.
import tensorflow as tf
import numpy as np
import logging
import os
import sys # Import sys
import datetime # Needed for TensorBoard log directory timestamp
from datasets import load_dataset
from tokenizers import Tokenizer
# Callbacks no longer needed for custom loop metrics
# from tensorflow.keras.callbacks import TensorBoard, Callback
import tensorflow as tf # Ensure tf is available for summary writing
from tqdm import tqdm # For progress bars
import argparse # Add argparse for command-line arguments

# --- Add project root to sys.path ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(script_dir)
# Add project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Modification ---


# Import model components using absolute paths from project root
from python.model.resonant_knowledge_model import ResonantKnowledgeModel
from python.model.custom_loss import custom_loss
from python.monad.manipulation_monad import ManipulationMonad # Import Monad
# dataset_utils might not be directly used if we use HF datasets pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Model Hyperparameters (adjust as needed)
MODEL_CONFIG = {
    # vocab_size will be determined by the tokenizer
    'sequence_length': 128, # Max sequence length for padding/truncation
    'embedding_dim': 256,
    'num_layers': 8,
    'num_heads': 8,
    'primes': [3, 5, 7, 11, 13, 17, 19, 23], # Example primes
    'beta': 1.0,
    'memory_size': 64,
    'ffn_dim_multiplier': 4,
    'dropout_rate': 0.1,
    # Loss hyperparameters
    'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.2, 'lambda4': 0.1,
}

# Training Parameters
TRAINING_CONFIG = {
    'tokenizer_path': 'daily_dialog_tokenizer.json', # Path relative to project root
    'dataset_name': 'daily_dialog',
    'text_column': 'dialog', # Column containing the dialogue list
    'batch_size': 8,        # Adjust based on GPU memory
    'epochs': 10,             # Number of training epochs
    'learning_rate': 5e-5,
    'buffer_size': 10000,    # Shuffle buffer size
    'validation_split': 0.1 # Use 10% of training data for validation if no val split exists
}

def preprocess_dataset(dataset, tokenizer, max_length):
    """
    Tokenizes, chunks, and prepares the dataset for language modeling.
    Each example will be a sequence of max_length tokens.
    """
    all_token_ids = []
    # Concatenate all dialogues and tokenize
    logger.info("Tokenizing dataset...")
    for example in dataset:
        dialogue_text = "\n".join(example[TRAINING_CONFIG['text_column']]) # Join turns
        # Add EOS token if tokenizer has one, otherwise handle separation implicitly
        encoding = tokenizer.encode(dialogue_text)
        all_token_ids.extend(encoding.ids)

    logger.info(f"Total tokens: {len(all_token_ids)}")

    # Chunk the tokens into sequences of max_length + 1
    # We need max_length+1 to create input/target pairs
    chunk_size = max_length + 1
    total_len = len(all_token_ids)
    # Drop the last partial chunk
    total_len = (total_len // chunk_size) * chunk_size
    all_token_ids = all_token_ids[:total_len]

    # Reshape into chunks
    chunks = np.array(all_token_ids).reshape(-1, chunk_size)
    logger.info(f"Created {chunks.shape[0]} chunks of size {chunk_size}")

    # Create input/target pairs
    input_sequences = chunks[:, :-1]
    target_sequences = chunks[:, 1:] # Target is the next token

    return input_sequences, target_sequences

def create_tf_dataset(inputs, targets, batch_size, seq_len, vocab_size, buffer_size):
    """
    Creates a tf.data.Dataset for training, yielding (inputs_dict, targets).
    """
    def generator():
        for input_seq, target_seq in zip(inputs, targets):
            input_tensor = tf.constant(input_seq, dtype=tf.int32)
            target_indices = tf.constant(target_seq, dtype=tf.int32)
            # One-hot encode targets
            target_one_hot = tf.one_hot(target_indices, depth=vocab_size, dtype=tf.float32)
            # Generate position inputs
            positions = tf.range(seq_len, dtype=tf.int32)
            # Create input dictionary
            inputs_dict = {
                'input_tokens': input_tensor,
                'positions_input': positions,
                'target_one_hot': target_one_hot # Pass targets in dict for model internal loss calc
            }
            # Yield (inputs, targets) tuple for Keras fit/compile with external metrics
            yield inputs_dict, target_one_hot

    # Define the output signature for the (inputs_dict, target_one_hot) tuple
    output_signature = (
        { # Input dictionary signature
            'input_tokens': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            'positions_input': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            'target_one_hot': tf.TensorSpec(shape=(seq_len, vocab_size), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(seq_len, vocab_size), dtype=tf.float32) # Target signature
    )

    # Create dataset with the generator and updated signature
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# --- Metrics Calculation Function ---
# Now only calculates and returns values, logging happens in the main loop
def calculate_metrics(y_true, y_pred_dict, model_config):
    """Helper function to calculate metrics."""
    metrics = {}
    if y_pred_dict is None or y_true is None:
        return metrics # Return empty dict if data is missing

    # Ensure y_pred is the dictionary output
    if not isinstance(y_pred_dict, dict):
        logger.warning(f"calculate_metrics: y_pred is not a dict.")
        return metrics # Return empty dict

    try:
        # --- Calculate Metrics ---
        # 1. CE Loss
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        ce_loss_per_sample = cce(y_true, y_pred_dict['logits'])
        metrics['ce_loss'] = tf.reduce_mean(ce_loss_per_sample)

        # 2. Entropy Penalty (lambda1)
        lambda1 = model_config.get('lambda1', 0.1)
        logits = y_pred_dict['logits']
        logits_l2 = tf.reduce_mean(tf.reduce_sum(tf.square(logits), axis=-1))
        metrics['entropy_penalty'] = tf.multiply(logits_l2, tf.constant(lambda1 * 0.01, dtype=logits.dtype))

        # 3. Dispersion Penalty (lambda2)
        lambda2 = model_config.get('lambda2', 0.1)
        hidden_states = y_pred_dict['final_hidden_states']
        hidden_l2 = tf.reduce_mean(tf.reduce_sum(tf.square(hidden_states), axis=-1))
        metrics['dispersion_penalty'] = tf.multiply(hidden_l2, tf.constant(lambda2 * 0.001, dtype=hidden_states.dtype))

        # 4. Alignment Penalty (lambda3) & Raw Alignment
        lambda3 = model_config.get('lambda3', 0.2)
        alignment = y_pred_dict['observer_alignment']
        alignment_mean = tf.reduce_mean(alignment)
        one_minus_alignment_mean = tf.subtract(tf.constant(1.0, dtype=alignment_mean.dtype), alignment_mean)
        metrics['alignment_penalty'] = tf.multiply(one_minus_alignment_mean, tf.constant(lambda3, dtype=alignment_mean.dtype))
        metrics['observer_alignment_mean'] = alignment_mean # Log raw mean too

        # 5. Gamma Mean
        metrics['gamma_mean'] = tf.reduce_mean(y_pred_dict['gamma'])

        # 6. Observer State Norm
        observer_state = y_pred_dict['observer_state']
        metrics['observer_state_norm'] = tf.reduce_mean(tf.norm(observer_state, axis=-1))

        # 7. Embedding Norms
        prime_emb = y_pred_dict['projected_prime']
        metrics['prime_emb_norm'] = tf.reduce_mean(tf.norm(prime_emb, axis=-1))
        phase_emb = y_pred_dict['projected_phase']
        metrics['phase_emb_norm'] = tf.reduce_mean(tf.norm(phase_emb, axis=-1))

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        # Return potentially partial metrics calculated so far
        return metrics

    return metrics


def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Train Resonant Knowledge Model on Daily Dialog dataset.')
    parser.add_argument(
        '--num_records',
        type=int,
        default=0,
        help='Number of training records to use (0 for all). Default: 0'
    )
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    # --- Load Tokenizer ---
    # Tokenizer path is now relative to project root
    tokenizer_path = os.path.join(project_root, TRAINING_CONFIG['tokenizer_path'])
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found at: {tokenizer_path}")
        return
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    MODEL_CONFIG['vocab_size'] = vocab_size # Set vocab size in model config
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")

    # --- Load Dataset ---
    logger.info(f"Loading dataset: {TRAINING_CONFIG['dataset_name']}")
    # Use cache_dir to potentially speed up subsequent loads
    dataset = load_dataset(TRAINING_CONFIG['dataset_name'], cache_dir="./hf_cache")
    logger.info(f"Dataset loaded: {dataset}")

    # --- Preprocess Data ---
    # Assuming 'train' split exists. Handle validation split.
    if 'validation' not in dataset:
        logger.warning("No validation split found. Creating one from the training set.")
        train_test_split = dataset['train'].train_test_split(test_size=TRAINING_CONFIG['validation_split'])
        train_dataset_hf = train_test_split['train']
        val_dataset_hf = train_test_split['test']
    else:
        train_dataset_hf = dataset['train']
        val_dataset_hf = dataset['validation']

    # --- Select subset of training data if specified ---
    if args.num_records > 0:
        logger.info(f"Using the first {args.num_records} records for training.")
        if args.num_records < len(train_dataset_hf):
             train_dataset_hf = train_dataset_hf.select(range(args.num_records))
             logger.info(f"Training dataset size after selection: {len(train_dataset_hf)}")
        else:
             logger.warning(f"Requested {args.num_records} records, but dataset only has {len(train_dataset_hf)}. Using all available training records.")
    else:
        logger.info("Using all available training records.")


    logger.info("Preprocessing training data...")
    train_inputs, train_targets = preprocess_dataset(
        train_dataset_hf, tokenizer, MODEL_CONFIG['sequence_length']
    )
    logger.info("Preprocessing validation data...")
    val_inputs, val_targets = preprocess_dataset(
        val_dataset_hf, tokenizer, MODEL_CONFIG['sequence_length']
    )

    # --- Create TensorFlow Datasets ---
    logger.info("Creating TensorFlow training dataset...")
    tf_train_dataset = create_tf_dataset(
        train_inputs, train_targets,
        TRAINING_CONFIG['batch_size'],
        MODEL_CONFIG['sequence_length'],
        MODEL_CONFIG['vocab_size'],
        TRAINING_CONFIG['buffer_size']
    )
    logger.info("Creating TensorFlow validation dataset...")
    tf_val_dataset = create_tf_dataset(
        val_inputs, val_targets,
        TRAINING_CONFIG['batch_size'],
        MODEL_CONFIG['sequence_length'],
        MODEL_CONFIG['vocab_size'],
        TRAINING_CONFIG['buffer_size'] # No need to shuffle validation data usually
    )

    # --- Initialize Model ---
    logger.info("Initializing the ResonantKnowledgeModel...")
    model = ResonantKnowledgeModel(config=MODEL_CONFIG)

    # --- Optimizer ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate'])

    # --- Initialize Monad ---
    # Pass relevant parts of model config if needed by Monad's init
    monad_config = {
        'primes': MODEL_CONFIG['primes'],
        # Add other monad-specific configs from MODEL_CONFIG if they exist
        'sigma_source_tensor': 'final_hidden_states' # Specify source tensor
    }
    monad = ManipulationMonad(config=monad_config)
    logger.info("ManipulationMonad initialized.")

    # --- tf.py_function setup for Monad ---
    # Define the Python function to be wrapped
    def _update_monad_py(source_tensor_np):
        # The monad instance is captured from the outer scope
        monad.update({'final_hidden_states': source_tensor_np}) # Use the correct key
        entropy = monad.get_symbolic_entropy()
        resonance = monad.resonance
        parity = 1.0 if monad.parity == 'even' else 0.0
        collapsed = 1.0 if monad.is_collapsed() else 0.0
        # Return values must be numpy types compatible with Tout
        return np.float32(entropy), np.float32(resonance), np.float32(parity), np.float32(collapsed)

    # Define the wrapper function using tf.py_function
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)]) # Input is the source tensor
    def update_monad_tf(source_tensor_tf):
        entropy, resonance, parity, collapsed = tf.py_function(
            func=_update_monad_py,
            inp=[source_tensor_tf],
            # Output types must match the numpy types returned by _update_monad_py
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
        )
        # Set shapes for the output tensors (optional but good practice)
        entropy.set_shape(())
        resonance.set_shape(())
        parity.set_shape(())
        collapsed.set_shape(())
        return entropy, resonance, parity, collapsed

    # --- TensorBoard Setup ---
    log_dir = "logs/rkm_baseline/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'validation'))
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Keras Metrics Setup ---
    # Define metric names for easier management
    metric_names = [
        'loss_total_observed', 'loss_internal', 'loss_monad', 'ce_loss',
        'entropy_penalty', 'dispersion_penalty', 'alignment_penalty',
        'observer_alignment_mean', 'gamma_mean', 'observer_state_norm',
        'prime_emb_norm', 'phase_emb_norm', 'symbolic_entropy',
        'resonance', 'parity_even', 'is_collapsed'
    ]
    # Create dictionaries to hold Mean metric objects for train and validation
    train_metrics = {name: tf.keras.metrics.Mean(name=f'train_{name}') for name in metric_names}
    val_metrics = {name: tf.keras.metrics.Mean(name=f'val_{name}') for name in metric_names}
    # Note: We will update all val metrics for potential analysis, even if only
    # internal loss is the primary validation target.

    # --- Custom Training Loop ---
    logger.info("Starting custom training loop...")
    global_step = 0
    for epoch in range(TRAINING_CONFIG['epochs']):
        logger.info(f"Starting Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")

        # Reset metrics at the start of each epoch
        for metric in train_metrics.values():
            metric.reset_state()
        for metric in val_metrics.values():
            metric.reset_state()

        # --- Training Phase ---
        progress_bar_train = tqdm(tf_train_dataset, desc=f"Epoch {epoch+1} Training", unit="batch")
        for step, (inputs_dict, y_true) in enumerate(progress_bar_train):
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred_dict = model(inputs_dict, training=True)
                # Get internal loss (sum of losses added via model.add_loss)
                # Ensure model.losses is cleared before the forward pass if necessary,
                # although typically Keras handles this per step in custom loops.
                # If losses accumulate unexpectedly, manual clearing might be needed.
                if not model.losses:
                     # If no loss was added (e.g., if target_one_hot wasn't in inputs_dict)
                     # This shouldn't happen with the current dataset structure
                     logger.error(f"No loss found in model.losses at training step {global_step}. Skipping batch.")
                     continue # Skip gradient calculation if loss is missing

                # The internal loss is the sum of losses added via model.add_loss()
                internal_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)

                # --- Monad Update and Loss Calculation (for logging only) ---
                source_tensor = y_pred_dict['final_hidden_states']
                # Ensure source tensor has float32 dtype for py_function
                source_tensor_float32 = tf.cast(source_tensor, tf.float32)

                symbolic_entropy_tensor, resonance_tensor, parity_tensor, collapse_tensor = update_monad_tf(source_tensor_float32)

                lambda4 = MODEL_CONFIG.get('lambda4', 0.1)
                monad_loss = tf.multiply(symbolic_entropy_tensor, tf.constant(lambda4, dtype=tf.float32))

                # Calculate total observed loss (for logging)
                total_observed_loss = internal_loss + monad_loss
                # --- End Monad Integration ---

            # Calculate gradients using ONLY the differentiable internal_loss
            gradients = tape.gradient(internal_loss, model.trainable_variables)

            # Check for None gradients (can happen if variables are not used in loss)
            filtered_grads_and_vars = []
            for grad, var in zip(gradients, model.trainable_variables):
                if grad is not None:
                    filtered_grads_and_vars.append((grad, var))
                # else:
                #     logger.warning(f"Gradient for variable {var.name} is None.")

            if not filtered_grads_and_vars:
                 logger.error(f"No gradients computed for any trainable variables at step {global_step}. Skipping optimizer step.")
                 # Potentially log more details about the loss and model state here
            else:
                 # Apply gradients
                 optimizer.apply_gradients(filtered_grads_and_vars)


            # Update Keras Mean metrics for training
            train_metrics['loss_total_observed'].update_state(total_observed_loss)
            train_metrics['loss_internal'].update_state(internal_loss)
            train_metrics['loss_monad'].update_state(monad_loss)
            train_metrics['symbolic_entropy'].update_state(symbolic_entropy_tensor)
            train_metrics['resonance'].update_state(resonance_tensor)
            train_metrics['parity_even'].update_state(parity_tensor)
            train_metrics['is_collapsed'].update_state(collapse_tensor)

            # Calculate base metrics for this batch
            batch_metrics = calculate_metrics(y_true, y_pred_dict, MODEL_CONFIG)
            # Update base metrics
            for name, value in batch_metrics.items():
                 if name in train_metrics:
                     train_metrics[name].update_state(value)

            # Log per-step internal loss directly for debugging
            with train_summary_writer.as_default(step=global_step):
                 tf.summary.scalar('debug/batch_internal_loss_direct', internal_loss)

            progress_bar_train.set_postfix(loss=f"{train_metrics['loss_total_observed'].result():.4f}")
            global_step += 1

        # Log average training metrics for the epoch
        with train_summary_writer.as_default(step=epoch):
            for name, metric in train_metrics.items():
                tf.summary.scalar(f'epoch_{name}', metric.result())
        train_summary_writer.flush() # Explicitly flush after epoch summary
        # Log to console
        train_log_str = f"Epoch {epoch+1} Training -"
        for name, metric in train_metrics.items():
             train_log_str += f" {name}: {metric.result():.4f}"
        logger.info(train_log_str)


        # --- Validation Phase ---
        logger.info(f"Starting Epoch {epoch+1} Validation")
        progress_bar_val = tqdm(tf_val_dataset, desc=f"Epoch {epoch+1} Validation", unit="batch")
        for inputs_dict, y_true in progress_bar_val:
            # Forward pass (no gradient tape needed)
            y_pred_dict = model(inputs_dict, training=False)

            # Get internal loss
            if not model.losses:
                 logger.warning(f"No loss found in model.losses during validation. Skipping batch.")
                 continue
            internal_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)

            # --- Monad Update for Validation (Logging Only) ---
            source_tensor_val = y_pred_dict['final_hidden_states']
            source_tensor_val_float32 = tf.cast(source_tensor_val, tf.float32)
            # Note: This updates the *same* monad instance state. Consider implications.
            symbolic_entropy_val, resonance_val, parity_val, collapse_val = update_monad_tf(source_tensor_val_float32)
            monad_loss_val = tf.multiply(symbolic_entropy_val, tf.constant(MODEL_CONFIG.get('lambda4', 0.1), dtype=tf.float32))
            total_observed_loss_val = internal_loss + monad_loss_val

            # Update Keras Mean metrics for validation
            val_metrics['loss_internal'].update_state(internal_loss)
            val_metrics['loss_monad'].update_state(monad_loss_val)
            val_metrics['loss_total_observed'].update_state(total_observed_loss_val)
            val_metrics['symbolic_entropy'].update_state(symbolic_entropy_val)
            val_metrics['resonance'].update_state(resonance_val)
            val_metrics['parity_even'].update_state(parity_val)
            val_metrics['is_collapsed'].update_state(collapse_val)

            # Calculate base metrics for this batch
            batch_metrics = calculate_metrics(y_true, y_pred_dict, MODEL_CONFIG) # Use the correct function name

            # Log Monad state and observed losses during validation (per step)
            with val_summary_writer.as_default(step=global_step): # Log against global step
                 tf.summary.scalar('loss/batch_internal', internal_loss)
                 tf.summary.scalar('loss/batch_monad', monad_loss_val)
                 tf.summary.scalar('loss/batch_total_observed', total_observed_loss_val)
                 tf.summary.scalar('monad/symbolic_entropy', symbolic_entropy_val)
                 tf.summary.scalar('monad/resonance', resonance_val)
                 tf.summary.scalar('monad/parity_even', parity_val)
                 tf.summary.scalar('monad/is_collapsed', collapse_val)

            # Update base metrics
            for name, value in batch_metrics.items():
                 if name in val_metrics:
                     val_metrics[name].update_state(value)

            # Log per-step internal loss directly during validation for debugging
            with val_summary_writer.as_default(step=global_step):
                 tf.summary.scalar('debug/batch_internal_loss_direct', internal_loss)


            progress_bar_val.set_postfix(loss=f"{val_metrics['loss_internal'].result():.4f}") # Show running avg internal loss

        # Log average validation metrics for the epoch
        with val_summary_writer.as_default(step=epoch):
            for name, metric in val_metrics.items():
                tf.summary.scalar(f'epoch_{name}', metric.result())
        val_summary_writer.flush() # Explicitly flush after epoch summary
        # Log to console
        val_log_str = f"Epoch {epoch+1} Validation -"
        for name, metric in val_metrics.items():
             val_log_str += f" {name}: {metric.result():.4f}"
        logger.info(val_log_str)


    logger.info("Custom training loop finished.")

    # Final flush just in case
    train_summary_writer.close()
    val_summary_writer.close()

    # --- Save Model ---
    save_path = os.path.join(project_root, 'rkm_daily_dialog_final.weights.h5')
    model.save_weights(save_path)
    logger.info(f"Final model weights saved to {save_path}")

if __name__ == "__main__":
    main()
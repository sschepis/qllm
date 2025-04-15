import tensorflow as tf
import numpy as np
import os
from tokenizers import Tokenizer
import logging

# Import the model definition (adjust path if necessary)
from .model.resonant_knowledge_model import ResonantKnowledgeModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use the same core model config as training, but adjust sequence length if needed for chat
# Ensure these match the parameters used during training for the loaded weights
MODEL_CONFIG = {
    # vocab_size will be determined by the tokenizer
    'sequence_length': 128, # Max sequence length for generation context
    'embedding_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'primes': [3, 5, 7, 11, 13],
    'beta': 1.0,
    'memory_size': 64,
    'ffn_dim_multiplier': 4,
    'dropout_rate': 0.1, # Dropout is typically inactive during inference
    # Loss hyperparameters are not needed for inference
}

CHAT_CONFIG = {
    'tokenizer_path': '../daily_dialog_tokenizer.json', # Relative path from chat.py
    'weights_path': '../rkm_daily_dialog_final.weights.h5', # Relative path from chat.py
    'max_generate_length': 50, # Max number of tokens to generate per turn
    'temperature': 0.7, # Sampling temperature (higher = more random)
    'top_k': 40, # Consider only top_k logits for sampling
}

def load_model_and_tokenizer():
    """Loads the tokenizer and the trained model."""
    # --- Load Tokenizer ---
    tokenizer_path_abs = os.path.join(os.path.dirname(__file__), CHAT_CONFIG['tokenizer_path'])
    if not os.path.exists(tokenizer_path_abs):
        logger.error(f"Tokenizer file not found at: {tokenizer_path_abs}")
        return None, None
    logger.info(f"Loading tokenizer from: {tokenizer_path_abs}")
    tokenizer = Tokenizer.from_file(tokenizer_path_abs)
    vocab_size = tokenizer.get_vocab_size()
    MODEL_CONFIG['vocab_size'] = vocab_size
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")

    # --- Initialize Model ---
    logger.info("Initializing the ResonantKnowledgeModel...")
    model = ResonantKnowledgeModel(config=MODEL_CONFIG)

    # --- Build Model ---
    # Build the model by calling it with dummy data matching the expected input spec
    # Note: We don't need target_one_hot for inference
    logger.info("Building model structure...")
    dummy_tokens = np.zeros((1, MODEL_CONFIG['sequence_length']), dtype=np.int32)
    dummy_positions = np.arange(MODEL_CONFIG['sequence_length'], dtype=np.int32).reshape(1, -1)
    dummy_input_dict = {'input_tokens': dummy_tokens, 'positions_input': dummy_positions}
    _ = model(dummy_input_dict, training=False) # Call with training=False
    model.summary() # Print model summary

    # --- Load Weights ---
    weights_path_abs = os.path.join(os.path.dirname(__file__), CHAT_CONFIG['weights_path'])
    if not os.path.exists(weights_path_abs):
        logger.error(f"Weights file not found at: {weights_path_abs}")
        logger.error("Please ensure the model has been trained and weights are saved.")
        return None, None
    logger.info(f"Loading weights from: {weights_path_abs}")
    model.load_weights(weights_path_abs)
    logger.info("Model weights loaded successfully.")

    return model, tokenizer

def generate_response(model, tokenizer, input_text, max_length, temperature, top_k):
    """Generates a response from the model given input text."""
    sequence_length = model.config['sequence_length']
    eos_token_id = tokenizer.token_to_id("[EOS]") # Assuming EOS token exists

    # Encode the input text
    input_encoding = tokenizer.encode(input_text)
    input_ids = input_encoding.ids
    
    # Truncate if necessary to fit within sequence length minus 1 (for next token)
    if len(input_ids) >= sequence_length:
        input_ids = input_ids[-(sequence_length - 1):]

    generated_ids = list(input_ids) # Start generation with input

    logger.info(f"Starting generation with IDs: {generated_ids}")

    for _ in range(max_length):
        # Prepare model input
        current_sequence = generated_ids[-(sequence_length):] # Get the last part of the sequence
        padded_sequence = np.pad(current_sequence, (sequence_length - len(current_sequence), 0), 'constant')
        
        input_tokens = np.array([padded_sequence], dtype=np.int32)
        positions = np.arange(sequence_length, dtype=np.int32).reshape(1, -1)
        
        input_dict = {'input_tokens': input_tokens, 'positions_input': positions}

        # Get logits from the model
        logits = model(input_dict, training=False) # Shape: (1, seq_len, vocab_size)

        # Get logits for the *next* token prediction (at the end of the current sequence)
        next_token_logits = logits[0, len(current_sequence) - 1, :] # Logits for the position after the last input token

        # Apply temperature scaling
        scaled_logits = next_token_logits / temperature

        # Apply Top-K sampling
        if top_k > 0:
            values, indices = tf.math.top_k(scaled_logits, k=top_k)
            # Create a mask for the logits, setting others to -inf
            k_mask = tf.scatter_nd(indices[:, tf.newaxis], tf.ones_like(values), shape=tf.shape(scaled_logits))
            k_mask = tf.cast(k_mask, dtype=tf.bool)
            scaled_logits = tf.where(k_mask, scaled_logits, tf.fill(tf.shape(scaled_logits), -float('inf')))

        # Sample the next token ID
        next_token_id = tf.random.categorical(scaled_logits[tf.newaxis, :], num_samples=1)[0, 0].numpy()

        # Append the generated token
        generated_ids.append(next_token_id)
        logger.debug(f"Generated token ID: {next_token_id} ({tokenizer.id_to_token(next_token_id)})")


        # Check for EOS token
        if eos_token_id is not None and next_token_id == eos_token_id:
            logger.info("EOS token generated. Stopping.")
            break

    # Decode the generated sequence (excluding the initial input)
    response_ids = generated_ids[len(input_ids):]
    response_text = tokenizer.decode(response_ids)
    
    return response_text

def main():
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return

    print("\nModel loaded. Type 'quit' to exit.")
    print("-" * 30)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            if not user_input:
                continue

            response = generate_response(
                model,
                tokenizer,
                user_input,
                max_length=CHAT_CONFIG['max_generate_length'],
                temperature=CHAT_CONFIG['temperature'],
                top_k=CHAT_CONFIG['top_k']
            )
            print(f"Bot: {response}")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            print("An error occurred. Please try again.")

    print("\nGoodbye!")

if __name__ == "__main__":
    # Add python directory to path if running script directly for imports
    # This is a common workaround but using `python -m python.chat` is preferred
    import sys
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    # Re-import now that path is set (if needed, depends on execution method)
    # from python.model.resonant_knowledge_model import ResonantKnowledgeModel

    main()
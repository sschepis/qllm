import tensorflow as tf
import math
# Import using absolute path from rkm package root
from python.layers.trig_activations import SineActivation
from python.layers.scale_layer import ScaleLayer

def prime_hilbert_embedding(config, inputs, positions, layers):
    """
    Implements Prime Hilbert Embedding using pre-instantiated Keras layers.
    Projects token embeddings into prime-dimensional subspaces with position-dependent phase shifts.
    Formalism: E(w, n) = ⊕ (P_pi(e_w) * sin(2πn/pi))

    Args:
        config (dict): Configuration object containing 'primes' (list of ints)
                       and potentially 'sequence_length'.
        inputs (tf.Tensor): Input token tensor, shape [batch, seq_len], containing token indices.
        positions (tf.Tensor): Position tensor, shape [batch, seq_len], containing position indices.
                                Expected to be castable to float for calculations.
        layers (dict): A dictionary containing pre-instantiated Keras layer instances.
                       Expected keys:
                       'embedding_layer': tf.keras.layers.Embedding
                       'reshape_layer': tf.keras.layers.Reshape (target_shape=[seq_len, 1])
                       'projection_layers': dict {prime: tf.keras.layers.Dense}
                       'scale_layers': dict {prime: ScaleLayer}
                       'sine_layers': dict {prime: SineActivation}
                       'multiply_layers': dict {prime: tf.keras.layers.Multiply}
                       'concat_layer': tf.keras.layers.Concatenate (axis=-1)

    Returns:
        tf.Tensor: The combined prime embedding tensor, shape [batch, seq_len, sum_of_primes].
    """
    primes = config.get('primes')
    # sequence_length = config.get('sequence_length') # Not directly used here

    if not primes:
        raise ValueError("Missing 'primes' list in config dictionary")
    if not isinstance(layers, dict):
         raise ValueError("'layers' argument must be a dictionary")

    # --- Check for required top-level layers ---
    required_top_layers = ['embedding_layer', 'reshape_layer', 'projection_layers',
                           'scale_layers', 'sine_layers', 'multiply_layers', 'concat_layer']
    for key in required_top_layers:
        if key not in layers:
            raise ValueError(f"Missing required layer key '{key}' in layers dictionary")
    if not isinstance(layers['projection_layers'], dict) or \
       not isinstance(layers['scale_layers'], dict) or \
       not isinstance(layers['sine_layers'], dict) or \
       not isinstance(layers['multiply_layers'], dict):
        raise ValueError("Layer group keys ('projection_layers', etc.) must map to dictionaries")


    # --- Base Embedding ---
    base_embedding = layers['embedding_layer'](inputs) # Shape: [batch, seq, embed_dim]

    # --- Reshape Positions ---
    # Ensure positions are float for scaling
    positions_float = tf.cast(positions, dtype=base_embedding.dtype)
    # Reshape positions to [batch, seq_len, 1] for broadcasting/scaling
    reshaped_positions = layers['reshape_layer'](positions_float) # Shape: [batch, seq, 1]

    embedding_parts = []

    # --- Project into Prime Subspaces ---
    for prime in primes:
        prime_key = int(prime) # Ensure key is int if primes are not already

        # Check for prime-specific layers
        if prime_key not in layers['projection_layers']:
            raise ValueError(f"Missing projection layer for prime {prime_key}")
        if prime_key not in layers['scale_layers']:
             raise ValueError(f"Missing scale layer for prime {prime_key}")
        if prime_key not in layers['sine_layers']:
             raise ValueError(f"Missing sine layer for prime {prime_key}")
        if prime_key not in layers['multiply_layers']:
             raise ValueError(f"Missing multiply layer for prime {prime_key}")

        # Project base embedding to prime dimension
        projection_layer = layers['projection_layers'][prime_key]
        projection = projection_layer(base_embedding) # Shape: [batch, seq, prime]

        # Calculate angle: 2πn / p
        # Assumes scale_layer for this prime was initialized with scale_factor = (2 * math.pi) / prime
        scale_layer = layers['scale_layers'][prime_key]
        angle = scale_layer(reshaped_positions) # Shape: [batch, seq, 1]

        # Apply sine activation: sin(2πn / p)
        sine_layer = layers['sine_layers'][prime_key]
        sin_angle = sine_layer(angle) # Shape: [batch, seq, 1]

        # Multiply projection by sinAngle (element-wise, broadcasting sin_angle)
        # P_pi(e_w) * sin(2πn/pi)
        multiply_layer = layers['multiply_layers'][prime_key]
        pos_encoding = multiply_layer([projection, sin_angle]) # Shape: [batch, seq, prime]

        embedding_parts.append(pos_encoding)

    # --- Concatenate ---
    # Concatenate all embedding parts along the last dimension
    if not embedding_parts:
        # Handle case with empty primes list, though config check should prevent this
        # Returning zeros matching expected output shape structure if possible, or raise error
        # This depends on how downstream layers expect the output.
        # For now, let's assume primes list is non-empty based on earlier check.
         raise ValueError("No embedding parts generated, primes list might be empty.")

    embedding = layers['concat_layer'](embedding_parts) # Shape: [batch, seq, sum_of_primes]

    return embedding

# Example Layer Instantiation (Illustrative)
# vocab_size = 10000
# embedding_dim = 256
# sequence_length = 128
# primes_list = [2, 3, 5, 7] # Example primes
# layers_dict = {
#     'embedding_layer': tf.keras.layers.Embedding(vocab_size, embedding_dim),
#     'reshape_layer': tf.keras.layers.Reshape((sequence_length, 1)),
#     'projection_layers': {p: tf.keras.layers.Dense(p, name=f'proj_{p}') for p in primes_list},
#     'scale_layers': {p: ScaleLayer(scale_factor=(2 * math.pi / p), name=f'scale_{p}') for p in primes_list},
#     'sine_layers': {p: SineActivation(name=f'sine_{p}') for p in primes_list},
#     'multiply_layers': {p: tf.keras.layers.Multiply(name=f'mult_{p}') for p in primes_list},
#     'concat_layer': tf.keras.layers.Concatenate(axis=-1)
# }
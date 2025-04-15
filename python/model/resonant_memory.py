import tensorflow as tf

def resonant_memory(config, inputs, memory_index, layers, training=False):
    """
    Implements a simplified Entropy-Modulated Resonant Memory using pre-instantiated Keras layers.
    Provides a 'modulation' term based on memory projection.

    Args:
        config (dict): Configuration object (potentially containing embedding_dim, memory_size).
        inputs (tf.Tensor): Input hidden states tensor, shape [batch, seq_len, embedding_dim].
        memory_index (int): Index for selecting the correct layers if stored per index (unused in this simplified version).
        layers (dict): A dictionary containing pre-instantiated Keras layer instances for this memory block.
                       Expected keys: 'projection_dense', 'attention_activation', 'values_dense',
                       'norm_layer', 'modulation_dense'.
        training (bool): Whether in training mode (unused in this function).

    Returns:
        dict: A dictionary containing:
              'modulation': The calculated memory modulation tensor, shape [batch, seq_len, embedding_dim].
              'attractors': Proxy for memory attractors (normalized values), shape [batch, seq_len, embedding_dim].
              'similarity': Proxy for similarity scores (attention weights), shape [batch, seq_len, memory_size].
    """
    # embedding_dim = config.get('embedding_dim') # Not directly used here
    # memory_size = config.get('memory_size', 128) # Ensure this matches projectionDense units

    # --- Check for required layers ---
    required_layers = [
        'projection_dense', 'attention_activation', 'values_dense',
        'norm_layer', 'modulation_dense'
    ]
    for key in required_layers:
        if key not in layers:
            raise ValueError(f"Missing required layer '{key}' in memory layers dictionary for index {memory_index}")

    # --- Memory Mechanism ---
    # Project hidden states into memory space
    # Input: [batch, seq, embed_dim] -> Output: [batch, seq, memory_size]
    memory_projection = layers['projection_dense'](inputs)

    # Calculate attention weights over memory slots (softmax activation)
    # Input: [batch, seq, memory_size] -> Output: [batch, seq, memory_size]
    memory_attention = layers['attention_activation'](memory_projection)

    # Retrieve weighted memory values (simulates attractor retrieval)
    # Input: [batch, seq, memory_size] -> Output: [batch, seq, embedding_dim] (values_dense projects back)
    memory_values = layers['values_dense'](memory_attention)

    # Normalize the retrieved values
    # Input: [batch, seq, embed_dim] -> Output: [batch, seq, embed_dim]
    normalized_memory = layers['norm_layer'](memory_values)

    # Calculate the final modulation signal
    # Input: [batch, seq, embed_dim] -> Output: [batch, seq, embed_dim]
    memory_modulation = layers['modulation_dense'](normalized_memory)

    # Return the modulation and proxy values
    return {
        'modulation': memory_modulation,
        'attractors': normalized_memory, # Proxy
        'similarity': memory_attention  # Proxy
    }

# Example Layer Instantiation (Illustrative)
# embedding_dim = 256
# memory_size = 128
# memory_idx = 1
# layers_dict = {
#     'projection_dense': tf.keras.layers.Dense(memory_size, activation='linear'),
#     'attention_activation': tf.keras.layers.Activation('softmax'),
#     'values_dense': tf.keras.layers.Dense(embedding_dim),
#     'norm_layer': tf.keras.layers.LayerNormalization(epsilon=1e-6),
#     'modulation_dense': tf.keras.layers.Dense(embedding_dim, activation='tanh')
# }
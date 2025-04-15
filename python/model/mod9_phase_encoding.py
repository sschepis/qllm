import tensorflow as tf
# Import using absolute path from rkm package root
from python.layers.trig_activations import SineActivation, CosineActivation

def mod9_phase_encoding(inputs, layers):
    """
    Implements Mod 9 Harmonic Phase Encoding using pre-instantiated Keras layers.
    Approximates G_φ = exp(i·2π·φ(w)/9) using sin/cos components.

    Args:
        inputs (tf.Tensor): Input token tensor, expected shape [batch, seq_len].
                            Assumed to contain token indices.
        layers (dict): A dictionary containing pre-instantiated Keras layer instances.
                       Expected keys:
                       'token_indices_embedding': tf.keras.layers.Embedding
                       'mod9_projection_layer': tf.keras.layers.Dense
                       'sine_dense_layer': tf.keras.layers.Dense
                       'sine_activation_layer': SineActivation (custom layer)
                       'cosine_dense_layer': tf.keras.layers.Dense
                       'cosine_activation_layer': CosineActivation (custom layer)
                       'concat_layer': tf.keras.layers.Concatenate

    Returns:
        tf.Tensor: The phase encoding tensor, shape [batch, seq_len, 2] (cos, sin components).
    """

    # Apply embedding layer (as per original JS, though potentially redundant if inputs are already indices)
    # Ensure the embedding layer exists in the provided dictionary
    if 'token_indices_embedding' not in layers:
        raise ValueError("Missing 'token_indices_embedding' in layers dictionary")
    token_indices_embedded = layers['token_indices_embedding'](inputs) # Embedding output shape: [batch, seq_len, embedding_dim]

    # Project to 9 units for mod-9 representation (approximation)
    if 'mod9_projection_layer' not in layers:
        raise ValueError("Missing 'mod9_projection_layer' in layers dictionary")
    # Input to Dense should be appropriate, embedding output is usually suitable
    mod9_projection = layers['mod9_projection_layer'](token_indices_embedded) # Output shape: [batch, seq_len, 9]

    # Generate sine component
    if 'sine_dense_layer' not in layers or 'sine_activation_layer' not in layers:
        raise ValueError("Missing sine layers ('sine_dense_layer' or 'sine_activation_layer') in layers dictionary")
    sine_dense = layers['sine_dense_layer'](mod9_projection) # Output shape: [batch, seq_len, 1] (assuming dense units=1)
    sine_component = layers['sine_activation_layer'](sine_dense) # Output shape: [batch, seq_len, 1]

    # Generate cosine component
    if 'cosine_dense_layer' not in layers or 'cosine_activation_layer' not in layers:
        raise ValueError("Missing cosine layers ('cosine_dense_layer' or 'cosine_activation_layer') in layers dictionary")
    cosine_dense = layers['cosine_dense_layer'](mod9_projection) # Output shape: [batch, seq_len, 1] (assuming dense units=1)
    cosine_component = layers['cosine_activation_layer'](cosine_dense) # Output shape: [batch, seq_len, 1]

    # Concatenate real (cos) and imaginary (sin) parts
    if 'concat_layer' not in layers:
         raise ValueError("Missing 'concat_layer' in layers dictionary")
    # Ensure the concat layer is configured with axis=-1
    phase_encoding = layers['concat_layer']([cosine_component, sine_component]) # Output shape: [batch, seq_len, 2]

    return phase_encoding

# Example of how layers might be instantiated (outside this function)
# layers_dict = {
#     'token_indices_embedding': tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
#     'mod9_projection_layer': tf.keras.layers.Dense(units=9, activation='softmax'), # Softmax might approximate distribution
#     'sine_dense_layer': tf.keras.layers.Dense(units=1),
#     'sine_activation_layer': SineActivation(),
#     'cosine_dense_layer': tf.keras.layers.Dense(units=1),
#     'cosine_activation_layer': CosineActivation(),
#     'concat_layer': tf.keras.layers.Concatenate(axis=-1)
# }
# Assuming inputs_tensor is defined
# encoding = mod9_phase_encoding(inputs_tensor, layers_dict)
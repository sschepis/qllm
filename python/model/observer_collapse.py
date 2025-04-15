import tensorflow as tf
# Import using absolute path from rkm package root
from python.layers.scale_layer import ScaleLayer

def observer_collapse(config, hidden_states, observer_state, layers):
    """
    Implements Observer-Conditioned Collapse using pre-instantiated Keras layers.
    Formalism: ℓ = softmax((1 - γ)Wx + γ·⟨x, o⟩) where γ = σ(V_o o)
    Uses projection of alignment term to vocab size.

    Args:
        config (dict): Configuration object (potentially containing sequence_length, though unused here).
        hidden_states (tf.Tensor): Hidden states tensor, shape [batch, seq_len, embedding_dim].
        observer_state (tf.Tensor): Observer state tensor, shape [batch, embedding_dim].
        layers (dict): A dictionary containing pre-instantiated Keras layer instances.
                       Expected keys: 'gamma_logits_dense', 'gamma_activation',
                       'standard_projection_dense', 'observer_tile_repeat',
                       'xo_elementwise_multiply', 'xo_sum_dense',
                       'alignment_projection_dense', 'negate_gamma_scale',
                       'one_minus_gamma_activation', 'gamma_reshape',
                       'one_minus_gamma_reshape', 'scale_standard_multiply',
                       'scale_observer_multiply', 'combine_projections_add'.

    Returns:
        dict: A dictionary containing:
              'logits': Combined logits tensor, shape [batch, seq_len, vocab_size].
              'gamma': Gamma tensor (observer influence factor), shape [batch, 1].
              'observer_alignment': Calculated observer alignment term <x, o>, shape [batch, seq_len, 1].
                                     (Added this return value as it's often needed for loss)
    """
    # sequence_length = config.get('sequence_length') # Not directly used in this refactored version

    # --- Check for required layers ---
    required_layers = [
        'gamma_logits_dense', 'gamma_activation', 'standard_projection_dense',
        'observer_tile_repeat', 'xo_elementwise_multiply', 'xo_sum_dense',
        'alignment_projection_dense', 'negate_gamma_scale', 'one_minus_gamma_activation',
        'gamma_reshape', 'one_minus_gamma_reshape', 'scale_standard_multiply',
        'scale_observer_multiply', 'combine_projections_add'
    ]
    for key in required_layers:
        if key not in layers:
            raise ValueError(f"Missing required layer '{key}' in layers dictionary")

    # --- Calculate gamma (observer influence factor) ---
    # gamma = σ(V_o o)
    gamma_logits = layers['gamma_logits_dense'](observer_state) # Shape: [batch, 1]
    gamma = layers['gamma_activation'](gamma_logits)           # Shape: [batch, 1]

    # --- Standard projection Wx ---
    standard_projection = layers['standard_projection_dense'](hidden_states) # Shape: [batch, seq, vocab]

    # --- Calculate observer alignment term ⟨x, o⟩ ---
    # Tile observer state 'o' [batch, embeddingDim] -> [batch, seq_len, embeddingDim]
    observer_tiled = layers['observer_tile_repeat'](observer_state)
    # Element-wise product x * o_tiled
    elementwise_product = layers['xo_elementwise_multiply']([hidden_states, observer_tiled]) # Shape: [batch, seq, embed]
    # Sum across embedding dim -> ⟨x, o⟩ for each position
    # Using Dense(1) to simulate sum reduction as per JS code
    observer_alignment = layers['xo_sum_dense'](elementwise_product) # Shape: [batch, seq, 1]

    # --- Project alignment term to vocab size ---
    projected_alignment = layers['alignment_projection_dense'](observer_alignment) # Shape: [batch, seq, vocab]
    observer_influence = projected_alignment # Renamed for clarity

    # --- Calculate 1 - gamma ---
    # Using 1 - sigmoid(x) = sigmoid(-x)
    negative_gamma_logits = layers['negate_gamma_scale'](gamma_logits) # ScaleLayer with factor -1
    one_minus_gamma = layers['one_minus_gamma_activation'](negative_gamma_logits) # Shape: [batch, 1]

    # --- Combine terms: (1 - γ)Wx + γ·(Projected ⟨x, o⟩) ---
    # Reshape gamma and 1-gamma for broadcasting: [batch, 1] -> [batch, 1, 1]
    gamma_broadcast = layers['gamma_reshape'](gamma)
    one_minus_gamma_broadcast = layers['one_minus_gamma_reshape'](one_minus_gamma)

    # Scale projections
    scaled_standard = layers['scale_standard_multiply']([standard_projection, one_minus_gamma_broadcast])
    scaled_observer = layers['scale_observer_multiply']([observer_influence, gamma_broadcast])

    # Combine the scaled terms
    combined_logits = layers['combine_projections_add']([scaled_standard, scaled_observer]) # Shape: [batch, seq, vocab]

    return {
        'logits': combined_logits,
        'gamma': gamma,
        'observer_alignment': observer_alignment # Return this as it's needed for the custom loss
    }

# Example Layer Instantiation (Illustrative)
# embedding_dim = 256
# vocab_size = 10000
# sequence_length = 128
# layers_dict = {
#     'gamma_logits_dense': tf.keras.layers.Dense(1, name='gamma_logits'),
#     'gamma_activation': tf.keras.layers.Activation('sigmoid', name='gamma_activation'),
#     'standard_projection_dense': tf.keras.layers.Dense(vocab_size, name='standard_projection'),
#     'observer_tile_repeat': tf.keras.layers.RepeatVector(sequence_length, name='observer_tile'),
#     'xo_elementwise_multiply': tf.keras.layers.Multiply(name='xo_multiply'),
#     'xo_sum_dense': tf.keras.layers.Dense(1, name='xo_sum_projection'), # Simulates sum
#     'alignment_projection_dense': tf.keras.layers.Dense(vocab_size, name='alignment_projection'),
#     'negate_gamma_scale': ScaleLayer(scale_factor=-1.0, name='negate_gamma'),
#     'one_minus_gamma_activation': tf.keras.layers.Activation('sigmoid', name='one_minus_gamma'),
#     'gamma_reshape': tf.keras.layers.Reshape((1, 1), name='gamma_reshape'),
#     'one_minus_gamma_reshape': tf.keras.layers.Reshape((1, 1), name='one_minus_gamma_reshape'),
#     'scale_standard_multiply': tf.keras.layers.Multiply(name='scale_standard'),
#     'scale_observer_multiply': tf.keras.layers.Multiply(name='scale_observer'),
#     'combine_projections_add': tf.keras.layers.Add(name='combine_logits')
# }
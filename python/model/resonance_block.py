import tensorflow as tf
# Import using absolute path from rkm package root
from python.model.resonance_attention import resonance_attention

def resonance_block(config, inputs, mask, block_index, layers, training=False):
    """
    Implements a Resonance Block using pre-instantiated Keras layers.
    Simplified structure: x_out = N(x_attn_norm + FFN(x_attn_norm))
    where x_attn_norm = N(x_in + Attn(x_in))

    Args:
        config (dict): Configuration object (passed to attention, might contain embedding_dim etc.).
        inputs (tf.Tensor): Input hidden states, shape [batch, seq_len, embedding_dim].
        mask (tf.Tensor | None): Attention mask (passed to attention).
        block_index (int): Index of the block (used for selecting layers).
        layers (dict): A dictionary containing pre-instantiated Keras layer instances for this block.
                       Expected keys: 'attn_layers' (dict for attention), 'attn_residual_add',
                       'attn_norm', 'ffn1_dense', 'ffn2_dense', 'ffn_norm', 'ffn_residual_add'.
        training (bool): Whether in training mode (passed to attention).

    Returns:
        tf.Tensor: Output hidden states, shape [batch, seq_len, embedding_dim].
    """
    # --- Check for required layers ---
    required_layers = [
        'attn_layers', 'attn_residual_add', 'attn_norm',
        'ffn1_dense', 'ffn2_dense', 'ffn_norm', 'ffn_residual_add'
    ]
    for key in required_layers:
        if key not in layers:
            raise ValueError(f"Missing required layer '{key}' in layers dictionary for block {block_index}")
    if not isinstance(layers['attn_layers'], dict):
        raise ValueError(f"'attn_layers' must be a dictionary for block {block_index}")

    # --- First Sublayer: Multi-Head Resonance Attention + Residual + Norm ---
    # Pass the pre-instantiated attention layers object to resonance_attention
    attn_results = resonance_attention(
        config=config,
        inputs=inputs,
        mask=mask,
        block_index=block_index,
        layers=layers['attn_layers'], # Pass the nested dictionary
        training=training
    )
    attention_output = attn_results['output'] # Extract the output tensor

    # Residual connection: inputs + attention_output
    attention_residual = layers['attn_residual_add']([inputs, attention_output])

    # Normalization after residual connection
    normalized_attention = layers['attn_norm'](attention_residual)

    # --- Second Sublayer: Feed-Forward Network + Residual + Norm ---
    # FFN typically consists of two dense layers with an activation in between
    ffn1 = layers['ffn1_dense'](normalized_attention) # Assumes activation is built into ffn1_dense
    ffn2 = layers['ffn2_dense'](ffn1)                 # Typically no activation on the second FFN layer before norm/residual

    # Normalization *after* FFN (as per the JS code structure provided)
    # Note: Standard Transformer often applies residual *before* the final norm.
    # This implementation follows the JS code's structure: FFN -> Norm -> Residual Add
    normalized_ffn = layers['ffn_norm'](ffn2)

    # Residual connection for FFN: connects back to the output of the *first* sublayer's norm
    final_output = layers['ffn_residual_add']([normalized_attention, normalized_ffn])

    return final_output

# Example Layer Instantiation (Illustrative)
# embedding_dim = 256
# ffn_dim = embedding_dim * 4 # Example FFN intermediate dimension
# dropout_rate = 0.1 # Example dropout
# block_idx = 0
# layers_dict = {
#     'attn_layers': { # Nested dict for attention layers (see resonance_attention example)
#         # ... attention layers ...
#     },
#     'attn_residual_add': tf.keras.layers.Add(),
#     'attn_norm': tf.keras.layers.LayerNormalization(epsilon=1e-6),
#     'ffn1_dense': tf.keras.layers.Dense(ffn_dim, activation='relu'), # Or 'gelu'
#     'ffn2_dense': tf.keras.layers.Dense(embedding_dim),
#     # Optional Dropout can be added here or within FFN layers
#     'ffn_norm': tf.keras.layers.LayerNormalization(epsilon=1e-6),
#     'ffn_residual_add': tf.keras.layers.Add()
# }
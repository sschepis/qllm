import tensorflow as tf
import math
# Import using absolute path from rkm package root
from python.layers.scale_layer import ScaleLayer

def resonance_attention(config, inputs, mask, block_index, layers, training=False):
    """
    Multi-Head Resonance Attention mechanism using pre-instantiated Keras layers.
    Implements Resonance Attention Dynamics (Section 4 of formalism) using standard tf.matmul.
    Entropy calculation is approximated as per the original implementation.

    Args:
        config (dict): Configuration object containing 'embedding_dim', 'num_heads',
                       'sequence_length', 'beta' (though beta is expected in layer).
        inputs (tf.Tensor): Input hidden states, shape [batch, seq_len, embedding_dim].
        mask (tf.Tensor | None): Attention mask, shape [batch, 1, 1, seq_len] or similar broadcastable.
        block_index (int): Index of the block (potentially used for layer naming/selection).
        layers (dict): A dictionary containing pre-instantiated Keras layer instances.
                       Expected keys: 'query_projection', 'key_projection', 'value_projection',
                       'q_reshape', 'q_permute', 'k_reshape', 'k_permute', 'v_reshape', 'v_permute',
                       'scale_scores_layer', 'beta_sharpening_layer', 'softmax_layer',
                       'entropy_square_multiply', 'entropy_pool',
                       'output_permute', 'output_reshape', 'output_dense'.
        training (bool): Whether in training mode (currently unused in this function).

    Returns:
        dict: A dictionary containing:
              'output': The attention output tensor, shape [batch, seq_len, embedding_dim].
              'entropy': The entropy proxy tensor, shape [batch, num_heads].
              'attention_weights': The attention weights tensor, shape [batch, num_heads, seq_len, seq_len].
    """
    num_heads = config.get('num_heads')
    embedding_dim = config.get('embedding_dim')
    sequence_length = config.get('sequence_length') # May not be strictly needed if shapes are dynamic

    if not num_heads or not embedding_dim:
        raise ValueError("Missing 'num_heads' or 'embedding_dim' in config")

    head_dim = embedding_dim // num_heads
    if head_dim * num_heads != embedding_dim:
        raise ValueError("embedding_dim must be divisible by num_heads")

    # --- Check for required layers ---
    required_layers = [
        'query_projection', 'key_projection', 'value_projection',
        'q_reshape', 'q_permute', 'k_reshape', 'k_permute', 'v_reshape', 'v_permute',
        'scale_scores_layer', 'beta_sharpening_layer', 'softmax_layer',
        'entropy_square_multiply', 'entropy_pool',
        'output_permute', 'output_reshape', 'output_dense'
    ]
    for key in required_layers:
        if key not in layers:
            raise ValueError(f"Missing required layer '{key}' in layers dictionary")

    # --- Project Q, K, V ---
    query = layers['query_projection'](inputs) # Shape: [batch, seq, embed]
    key = layers['key_projection'](inputs)     # Shape: [batch, seq, embed]
    value = layers['value_projection'](inputs)   # Shape: [batch, seq, embed]

    # --- Reshape and Permute for Multi-Head ---
    # Reshape: [batch, seq, embed] -> [batch, seq, heads, head_dim]
    # Permute: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    def reshape_and_permute(tensor, reshape_layer, permute_layer):
        reshaped = reshape_layer(tensor)
        return permute_layer(reshaped)

    q = reshape_and_permute(query, layers['q_reshape'], layers['q_permute'])
    k = reshape_and_permute(key, layers['k_reshape'], layers['k_permute'])
    v = reshape_and_permute(value, layers['v_reshape'], layers['v_permute'])

    # --- Calculate Scaled Dot-Product Attention Scores ---
    # MatMul: q * k^T -> [batch, heads, seq, seq]
    # Use tf.matmul for standard attention calculation
    scores = tf.matmul(q, k, transpose_b=True)

    # Scale scores by 1 / sqrt(head_dim)
    # Assumes scale_scores_layer is initialized with scale_factor = 1.0 / math.sqrt(head_dim)
    scaled_scores = layers['scale_scores_layer'](scores)

    # Apply beta sharpening
    # Assumes beta_sharpening_layer is a ScaleLayer initialized with the desired beta factor
    sharpened_scores = layers['beta_sharpening_layer'](scaled_scores)

    # Apply mask if provided
    if mask is not None:
        # Ensure mask is broadcastable to scores shape [batch, heads, seq, seq]
        # Common mask shape: [batch, 1, 1, seq_len]
        # Add a large negative number where mask is zero (or False)
        sharpened_scores += (tf.cast(mask, dtype=sharpened_scores.dtype) * -1e9)

    # Apply softmax
    attention_weights = layers['softmax_layer'](sharpened_scores) # Shape: [batch, heads, seq, seq]

    # --- Calculate Entropy Proxy ---
    # H_proxy = mean(α^2) over seq, seq dimensions
    squared_weights = layers['entropy_square_multiply']([attention_weights, attention_weights])
    # Pool across the last two dimensions (seq, seq) -> [batch, heads]
    entropy = layers['entropy_pool'](squared_weights)

    # --- Calculate Weighted Values ---
    # MatMul: attention_weights * v -> [batch, heads, seq, head_dim]
    weighted_values = tf.matmul(attention_weights, v)

    # --- Reshape and Project Output ---
    # Permute: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
    permuted_output = layers['output_permute'](weighted_values)
    # Reshape: [batch, seq, heads, head_dim] -> [batch, seq, embedding_dim]
    reshaped_output = layers['output_reshape'](permuted_output)
    # Final dense projection
    output = layers['output_dense'](reshaped_output) # Shape: [batch, seq, embedding_dim]

    return {
        'output': output,
        'entropy': entropy,
        'attention_weights': attention_weights
    }

# Example Layer Instantiation (Illustrative - specific layer configs depend on ResonantKnowledgeModel)
# batch_size = 4
# sequence_length = 128
# embedding_dim = 256
# num_heads = 8
# head_dim = embedding_dim // num_heads
# beta_value = 1.5 # Example beta
#
# layers_dict = {
#     'query_projection': tf.keras.layers.Dense(embedding_dim),
#     'key_projection': tf.keras.layers.Dense(embedding_dim),
#     'value_projection': tf.keras.layers.Dense(embedding_dim),
#     'q_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim)),
#     'q_permute': tf.keras.layers.Permute((2, 1, 3)), # batch, heads, seq, head_dim
#     'k_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim)),
#     'k_permute': tf.keras.layers.Permute((2, 1, 3)), # batch, heads, seq, head_dim
#     'v_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim)),
#     'v_permute': tf.keras.layers.Permute((2, 1, 3)), # batch, heads, seq, head_dim
#     'scale_scores_layer': ScaleLayer(scale_factor=1.0 / math.sqrt(float(head_dim))),
#     'beta_sharpening_layer': ScaleLayer(scale_factor=beta_value),
#     'softmax_layer': tf.keras.layers.Softmax(axis=-1),
#     'entropy_square_multiply': tf.keras.layers.Multiply(),
#     'entropy_pool': tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first'), # Pool over last 2 dims
#     'output_permute': tf.keras.layers.Permute((2, 1, 3)), # batch, seq, heads, head_dim
#     'output_reshape': tf.keras.layers.Reshape((sequence_length, embedding_dim)),
#     'output_dense': tf.keras.layers.Dense(embedding_dim)
# }
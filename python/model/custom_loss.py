import tensorflow as tf

def custom_loss(config):
    """
    Implements the Resonant Knowledge Learning Objective (approximated).
    L = L_CE + λ₁S_p + λ₂(mod9 Phase Dispersion) + λ₃(1 - ⟨x, o⟩)
    Note: The λ₄E(σ) term is handled separately in the custom training loop.

    Args:
        config (dict): Configuration object containing lambda hyperparameters
                       (lambda1, lambda2, lambda3).

    Returns:
        function: A loss function that takes (y_true, y_pred) and returns a scalar loss tensor.
    """
    # Loss hyperparameters from config
    lambda1 = config.get('lambda1', 0.1)  # S_p: Symbolic prime entropy weight (approximated)
    lambda2 = config.get('lambda2', 0.1)  # Mod9 Phase Dispersion weight (approximated)
    lambda3 = config.get('lambda3', 0.2)  # Observer alignment weight (1 - <x,o>) (approximated)
    # lambda4 = config.get('lambda4', 0.1) # Monad penalty handled in training loop

    # Define the actual loss function closure
    @tf.function # Optional: Decorate with tf.function for potential performance improvement
    def loss_fn(y_true, y_pred):
        """
        Calculates the custom loss.

        Args:
            y_true: The true target values. Expected shape compatible with logits for cross-entropy.
            y_pred: A list or tuple of model outputs:
                    [logits, gamma, hidden_states, observer_state, observer_alignment].

        Returns:
            tf.Tensor: A scalar tensor representing the total calculated loss.
        """
        # Ensure the order matches the model's output definition
        # Note: Renamed hiddenStates -> hidden_states etc. for Python conventions
        logits, gamma, hidden_states, observer_state, observer_alignment = y_pred

        # Ensure y_true has the correct dtype, often float32 for cross-entropy targets if they are one-hot encoded
        # If y_true contains class indices, SparseCategoricalCrossentropy might be needed instead.
        # Assuming y_true is appropriately formatted (e.g., one-hot encoded float32)
        # y_true = tf.cast(y_true, dtype=logits.dtype) # Cast if necessary

        # 1. Primary loss: L_CE (Categorical Cross-Entropy)
        # Use tf.keras.losses for standard implementations
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        ce_loss_per_sample = cce(y_true, logits)
        ce_loss = tf.reduce_mean(ce_loss_per_sample) # Average over the batch

        # 2. Symbolic Prime Entropy (S_p) Approximation
        # Using L2 norm on logits as a proxy. Encourages smoother distributions.
        logits_l2 = tf.reduce_mean(tf.reduce_sum(tf.square(logits), axis=-1))
        # Use tf.constant for scalar hyperparameters
        entropy_penalty = tf.multiply(logits_l2, tf.constant(lambda1 * 0.01, dtype=logits.dtype)) # Scaled S_p approximation

        # 3. Mod9 Phase Dispersion Approximation
        # Using L2 norm on hidden states as a proxy. Penalizes large activations.
        hidden_l2 = tf.reduce_mean(tf.reduce_sum(tf.square(hidden_states), axis=-1))
        dispersion_penalty = tf.multiply(hidden_l2, tf.constant(lambda2 * 0.001, dtype=hidden_states.dtype)) # Scaled dispersion approximation

        # 4. Observer Alignment Penalty: λ₃·mean(1 - ⟨x, o⟩)
        # Use the observer_alignment tensor passed from the model outputs
        # observer_alignment expected shape [batch, seq, 1] or similar
        alignment_mean = tf.reduce_mean(observer_alignment) # Mean over all dimensions
        one_minus_alignment_mean = tf.subtract(tf.constant(1.0, dtype=alignment_mean.dtype), alignment_mean)
        alignment_penalty = tf.multiply(one_minus_alignment_mean, tf.constant(lambda3, dtype=alignment_mean.dtype))

        # 5. Monad Symbolic Entropy Penalty: λ₄E(σ) - REMOVED
        # This term is calculated and added *outside* this loss function in the training loop.

        # Combine losses
        total_loss = ce_loss
        total_loss = tf.add(total_loss, entropy_penalty)
        total_loss = tf.add(total_loss, dispersion_penalty)
        total_loss = tf.add(total_loss, alignment_penalty)

        return total_loss

    return loss_fn

# Example usage (optional, for testing):
# if __name__ == '__main__':
#     config = {'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.2}
#     loss_func = custom_loss(config)
#
#     # Dummy data
#     batch_size = 4
#     seq_len = 10
#     vocab_size = 100
#     hidden_dim = 32
#
#     y_true_dummy = tf.one_hot(tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32), depth=vocab_size)
#     logits_dummy = tf.random.normal((batch_size, seq_len, vocab_size))
#     gamma_dummy = tf.random.normal((batch_size, seq_len, 1)) # Example shape
#     hidden_states_dummy = tf.random.normal((batch_size, seq_len, hidden_dim))
#     observer_state_dummy = tf.random.normal((batch_size, hidden_dim)) # Example shape
#     observer_alignment_dummy = tf.random.uniform((batch_size, seq_len, 1), minval=0.0, maxval=1.0)
#
#     y_pred_dummy = [logits_dummy, gamma_dummy, hidden_states_dummy, observer_state_dummy, observer_alignment_dummy]
#
#     loss_value = loss_func(y_true_dummy, y_pred_dummy)
#     print(f"Calculated loss: {loss_value.numpy()}")
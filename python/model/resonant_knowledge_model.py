import tensorflow as tf
import math
import numpy as np # For position generation

# Import custom layers and component functions
from ..layers.trig_activations import SineActivation, CosineActivation
from ..layers.scale_layer import ScaleLayer
from ..utils.math_utils import digital_root # Assuming this exists in Python utils
from .prime_hilbert_embedding import prime_hilbert_embedding
from .mod9_phase_encoding import mod9_phase_encoding
from .resonance_attention import resonance_attention
from .resonance_block import resonance_block
from .resonant_memory import resonant_memory
from .observer_collapse import observer_collapse
from .custom_loss import custom_loss # Import the custom loss function

class ResonantKnowledgeModel(tf.keras.Model):
    """
    Resonant Knowledge Model Implementation (TensorFlow/Keras).
    A neural language model with quantum-inspired processing based on formalism from papers/learn.md.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = {
            'vocab_size': 30000,
            'embedding_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'primes': [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
            'sequence_length': 512,
            'beta': 1.0, # Default beta for attention sharpening
            'memory_size': 128, # Default memory size
            'ffn_dim_multiplier': 4, # Standard transformer FFN multiplier
            'dropout_rate': 0.1, # Example dropout rate
            # Loss hyperparameters (can be passed in config or used separately)
            'lambda1': 0.1,
            'lambda2': 0.1,
            'lambda3': 0.2,
            'lambda4': 0.1,
            **config # Override defaults with provided config
        }

        # Extract config values for convenience
        vocab_size = self.config['vocab_size']
        embedding_dim = self.config['embedding_dim']
        num_layers = self.config['num_layers']
        num_heads = self.config['num_heads']
        primes = self.config['primes']
        sequence_length = self.config['sequence_length']
        beta = self.config['beta']
        memory_size = self.config['memory_size']
        ffn_dim = embedding_dim * self.config['ffn_dim_multiplier']
        dropout_rate = self.config['dropout_rate'] # Added dropout

        if embedding_dim % num_heads != 0:
             raise ValueError("embedding_dim must be divisible by num_heads")
        head_dim = embedding_dim // num_heads

        # --- Layer Dictionary ---
        self.layers_dict = {}

        # --- Embedding Layers ---
        self.layers_dict['embedding_layer'] = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, name='base_embedding'
        )
        self.layers_dict['pos_reshape_layer'] = tf.keras.layers.Reshape(
            (sequence_length, 1), name='positions_reshape'
        )
        # Prime Hilbert Embedding Sub-layers
        self.layers_dict['prime_projection_layers'] = {}
        self.layers_dict['prime_scale_layers'] = {}
        self.layers_dict['prime_sine_layers'] = {}
        self.layers_dict['prime_multiply_layers'] = {}
        for prime in primes:
            p_key = int(prime)
            self.layers_dict['prime_projection_layers'][p_key] = tf.keras.layers.Dense(
                p_key, name=f'prime_projection_{p_key}'
            )
            scale_factor = (2 * math.pi) / p_key
            self.layers_dict['prime_scale_layers'][p_key] = ScaleLayer(
                scale_factor=scale_factor, name=f'angle_scale_{p_key}'
            )
            self.layers_dict['prime_sine_layers'][p_key] = SineActivation(
                name=f'sin_angle_{p_key}'
            )
            self.layers_dict['prime_multiply_layers'][p_key] = tf.keras.layers.Multiply(
                name=f'pos_encoding_{p_key}'
            )
        self.layers_dict['prime_concat_layer'] = tf.keras.layers.Concatenate(
            axis=-1, name='prime_embedding_concat'
        )
        # Project combined prime embedding back to embedding_dim
        self.layers_dict['prime_final_projection'] = tf.keras.layers.Dense(
            embedding_dim, name='project_prime_embedding'
        )

        # Mod9 Phase Encoding Sub-layers
        # Note: Using a non-trainable embedding for indices might be unusual.
        # Consider if this layer is truly needed or if indices can be used directly.
        self.layers_dict['mod9_token_indices_embedding'] = tf.keras.layers.Embedding(
             vocab_size, 1, trainable=False, name='token_indices_embedding'
        )
        self.layers_dict['mod9_projection_layer'] = tf.keras.layers.Dense(
            9, activation='softmax', name='mod9_projection'
        )
        self.layers_dict['mod9_sine_dense_layer'] = tf.keras.layers.Dense(
            1, name='sine_dense'
        )
        self.layers_dict['mod9_sine_activation_layer'] = SineActivation(
            name='sine_component'
        )
        self.layers_dict['mod9_cosine_dense_layer'] = tf.keras.layers.Dense(
            1, name='cosine_dense'
        )
        self.layers_dict['mod9_cosine_activation_layer'] = CosineActivation(
            name='cosine_component'
        )
        self.layers_dict['mod9_concat_layer'] = tf.keras.layers.Concatenate(
            axis=-1, name='phase_encoding'
        )
        # Project combined phase encoding back to embedding_dim
        self.layers_dict['mod9_final_projection'] = tf.keras.layers.Dense(
            embedding_dim, name='project_phase_encoding'
        )

        # Combine Embeddings
        self.layers_dict['combine_embeddings_add'] = tf.keras.layers.Add(
            name='combine_embeddings'
        )
        self.layers_dict['embedding_norm'] = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='embedding_norm'
        )
        self.layers_dict['embedding_dropout'] = tf.keras.layers.Dropout(dropout_rate) # Added dropout

        # --- Resonance Blocks ---
        self.layers_dict['resonance_blocks'] = []
        for i in range(num_layers):
            name_prefix = f'block_{i}'
            attn_prefix = f'{name_prefix}_attn'
            block_layers = {
                'attn_layers': { # Layers needed by Python resonance_attention
                    'query_projection': tf.keras.layers.Dense(embedding_dim, name=f'{attn_prefix}_query_proj'),
                    'key_projection': tf.keras.layers.Dense(embedding_dim, name=f'{attn_prefix}_key_proj'),
                    'value_projection': tf.keras.layers.Dense(embedding_dim, name=f'{attn_prefix}_value_proj'),
                    'q_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim), name=f'{attn_prefix}_q_reshape'),
                    'k_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim), name=f'{attn_prefix}_k_reshape'),
                    'v_reshape': tf.keras.layers.Reshape((sequence_length, num_heads, head_dim), name=f'{attn_prefix}_v_reshape'),
                    'q_permute': tf.keras.layers.Permute((2, 1, 3), name=f'{attn_prefix}_q_permute'), # batch, heads, seq, head_dim
                    'k_permute': tf.keras.layers.Permute((2, 1, 3), name=f'{attn_prefix}_k_permute'),
                    'v_permute': tf.keras.layers.Permute((2, 1, 3), name=f'{attn_prefix}_v_permute'),
                    'scale_scores_layer': ScaleLayer(scale_factor=1.0 / math.sqrt(float(head_dim)), name=f'{attn_prefix}_scale_scores'),
                    'beta_sharpening_layer': ScaleLayer(scale_factor=beta, name=f'{attn_prefix}_beta_sharpen'),
                    'softmax_layer': tf.keras.layers.Softmax(axis=-1, name=f'{attn_prefix}_softmax'),
                    'entropy_square_multiply': tf.keras.layers.Multiply(name=f'{attn_prefix}_entropy_square'),
                    'entropy_pool': tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=f'{attn_prefix}_entropy_pool'), # Pool over last 2 dims
                    'output_permute': tf.keras.layers.Permute((2, 1, 3), name=f'{attn_prefix}_out_permute'), # batch, seq, heads, head_dim
                    'output_reshape': tf.keras.layers.Reshape((sequence_length, embedding_dim), name=f'{attn_prefix}_out_reshape'),
                    'output_dense': tf.keras.layers.Dense(embedding_dim, name=f'{attn_prefix}_out_dense'),
                    'attn_dropout': tf.keras.layers.Dropout(dropout_rate) # Added dropout
                },
                'attn_residual_add': tf.keras.layers.Add(name=f'{name_prefix}_attn_add'),
                'attn_norm': tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_attn_norm'),
                'ffn1_dense': tf.keras.layers.Dense(ffn_dim, activation='gelu', name=f'{name_prefix}_ffn1'), # Using gelu
                'ffn_dropout': tf.keras.layers.Dropout(dropout_rate), # Added dropout
                'ffn2_dense': tf.keras.layers.Dense(embedding_dim, name=f'{name_prefix}_ffn2'),
                'ffn_norm': tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ffn_norm'),
                'ffn_residual_add': tf.keras.layers.Add(name=f'{name_prefix}_ffn_add')
            }
            self.layers_dict['resonance_blocks'].append(block_layers)

        # --- Resonant Memory Layers ---
        self.layers_dict['resonant_memory'] = []
        for i in range(num_layers):
             # Apply memory every 2 layers, starting after layer 1 (index 1)
            if i % 2 == 1:
                name_prefix = f'memory_{i}'
                memory_layers = {
                    'projection_dense': tf.keras.layers.Dense(memory_size, activation='linear', name=f'{name_prefix}_proj'),
                    'attention_activation': tf.keras.layers.Activation('softmax', name=f'{name_prefix}_attn'),
                    'values_dense': tf.keras.layers.Dense(embedding_dim, name=f'{name_prefix}_values'),
                    'norm_layer': tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_norm'),
                    'modulation_dense': tf.keras.layers.Dense(embedding_dim, activation='tanh', name=f'{name_prefix}_mod'),
                    'memory_add_layer': tf.keras.layers.Add(name=f'{name_prefix}_add'),
                    'memory_norm_layer': tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_final_norm')
                }
                self.layers_dict['resonant_memory'].append(memory_layers)
            else:
                self.layers_dict['resonant_memory'].append(None) # Placeholder

        # --- Observer State Layers ---
        self.layers_dict['observer_pool'] = tf.keras.layers.GlobalAveragePooling1D(name='observer_pool')
        self.layers_dict['observer_projection'] = tf.keras.layers.Dense(
            embedding_dim, activation='tanh', name='observer_projection'
        )

        # --- Observer Collapse Layers ---
        # Layers specifically for calculating the final output logits
        self.layers_dict['collapse_gamma_logits_dense'] = tf.keras.layers.Dense(1, name='collapse_gamma_logits')
        self.layers_dict['collapse_gamma_activation'] = tf.keras.layers.Activation('sigmoid', name='collapse_gamma')
        self.layers_dict['collapse_standard_projection_dense'] = tf.keras.layers.Dense(vocab_size, name='collapse_standard_proj')
        self.layers_dict['collapse_observer_tile_repeat'] = tf.keras.layers.RepeatVector(sequence_length, name='collapse_observer_tile')
        self.layers_dict['collapse_xo_elementwise_multiply'] = tf.keras.layers.Multiply(name='collapse_xo_multiply')
        self.layers_dict['collapse_xo_sum_dense'] = tf.keras.layers.Dense(1, use_bias=False, name='collapse_xo_sum') # Simulate sum
        self.layers_dict['collapse_alignment_projection_dense'] = tf.keras.layers.Dense(vocab_size, name='collapse_alignment_proj')
        self.layers_dict['collapse_negate_gamma_scale'] = ScaleLayer(scale_factor=-1.0, name='collapse_negate_gamma')
        self.layers_dict['collapse_one_minus_gamma_activation'] = tf.keras.layers.Activation('sigmoid', name='collapse_one_minus_gamma')
        self.layers_dict['collapse_gamma_reshape'] = tf.keras.layers.Reshape((1, 1), name='collapse_gamma_reshape')
        self.layers_dict['collapse_one_minus_gamma_reshape'] = tf.keras.layers.Reshape((1, 1), name='collapse_one_minus_gamma_reshape')
        self.layers_dict['collapse_scale_standard_multiply'] = tf.keras.layers.Multiply(name='collapse_scale_standard')
        self.layers_dict['collapse_scale_observer_multiply'] = tf.keras.layers.Multiply(name='collapse_scale_observer')
        self.layers_dict['collapse_combine_projections_add'] = tf.keras.layers.Add(name='collapse_combine_logits')

        # Layers for calculating observer_alignment output needed for loss function
        # Re-instantiate these specifically for the loss calculation path if needed,
        # or reuse if shapes/behavior are identical. Reusing for now.
        self.layers_dict['loss_observer_tile_repeat'] = self.layers_dict['collapse_observer_tile_repeat']
        self.layers_dict['loss_xo_elementwise_multiply'] = self.layers_dict['collapse_xo_elementwise_multiply']
        self.layers_dict['loss_xo_sum_dense'] = self.layers_dict['collapse_xo_sum_dense']

        # Instantiate the loss function
        self.loss_calculator = custom_loss(self.config)


    def call(self, inputs, training=False):
        """
        Forward pass of the Resonant Knowledge Model. Calculates loss internally.

        Args:
            inputs: A dictionary containing:
                    - 'input_tokens': tf.Tensor, shape [batch, seq_len].
                    - 'positions_input': tf.Tensor, shape [batch, seq_len].
                    - 'target_one_hot': (Optional) tf.Tensor, shape [batch, seq_len, vocab_size]. Targets for loss calculation.
            training (bool): Indicates if the model is in training mode (for dropout).

        Returns:
            dict: A dictionary containing the primary output 'logits' and other
                  tensors needed for metric calculation:
                  - 'logits': Output logits [batch, seq_len, vocab_size]
                  - 'gamma': Observer gate values [batch, seq_len, 1]
                  - 'final_hidden_states': Hidden states before collapse [batch, seq_len, embed_dim]
                  - 'observer_state': Pooled observer state [batch, embed_dim]
                  - 'observer_alignment': Alignment term <x, o> [batch, seq_len, 1]
                  - 'projected_prime': Projected prime Hilbert embedding [batch, seq_len, embed_dim]
                  - 'projected_phase': Projected mod9 phase embedding [batch, seq_len, embed_dim]
        """
        # Unpack inputs from the dictionary
        input_tokens = inputs['input_tokens']
        positions_input = inputs['positions_input']
        # Get target_one_hot if it exists, otherwise None
        target_one_hot = inputs.get('target_one_hot', None)

        # --- Embedding ---
        prime_emb_layers = {
            'embedding_layer': self.layers_dict['embedding_layer'],
            'reshape_layer': self.layers_dict['pos_reshape_layer'],
            'projection_layers': self.layers_dict['prime_projection_layers'],
            'scale_layers': self.layers_dict['prime_scale_layers'],
            'sine_layers': self.layers_dict['prime_sine_layers'],
            'multiply_layers': self.layers_dict['prime_multiply_layers'],
            'concat_layer': self.layers_dict['prime_concat_layer']
        }
        prime_emb = prime_hilbert_embedding(self.config, input_tokens, positions_input, prime_emb_layers)

        mod9_enc_layers = {
            'token_indices_embedding': self.layers_dict['mod9_token_indices_embedding'],
            'mod9_projection_layer': self.layers_dict['mod9_projection_layer'],
            'sine_dense_layer': self.layers_dict['mod9_sine_dense_layer'],
            'sine_activation_layer': self.layers_dict['mod9_sine_activation_layer'],
            'cosine_dense_layer': self.layers_dict['mod9_cosine_dense_layer'],
            'cosine_activation_layer': self.layers_dict['mod9_cosine_activation_layer'],
            'concat_layer': self.layers_dict['mod9_concat_layer']
        }
        phase_enc = mod9_phase_encoding(input_tokens, mod9_enc_layers) # config not needed

        projected_prime = self.layers_dict['prime_final_projection'](prime_emb)
        projected_phase = self.layers_dict['mod9_final_projection'](phase_enc)

        embedding_combined = self.layers_dict['combine_embeddings_add']([projected_prime, projected_phase])
        hidden_states = self.layers_dict['embedding_norm'](embedding_combined)
        hidden_states = self.layers_dict['embedding_dropout'](hidden_states, training=training)


        # --- Resonance Blocks ---
        attention_mask = None # Placeholder for potential future mask implementation
        all_hidden_states = [hidden_states] # Keep track for potential analysis

        for i in range(self.config['num_layers']):
            block_layers = self.layers_dict['resonance_blocks'][i]
            hidden_states = resonance_block(
                config=self.config,
                inputs=hidden_states,
                mask=attention_mask,
                block_index=i,
                layers=block_layers,
                training=training
            )
            all_hidden_states.append(hidden_states) # Store state after block

            # --- Resonant Memory ---
            if i % 2 == 1:
                memory_layers = self.layers_dict['resonant_memory'][i]
                if memory_layers: # Check if memory layers exist for this index
                    memory_output = resonant_memory(
                        config=self.config,
                        inputs=hidden_states,
                        memory_index=i, # Pass index for potential layer selection inside
                        layers=memory_layers,
                        training=training
                    )
                    # Combine memory output (modulation) with hidden state
                    hidden_states = memory_layers['memory_add_layer']([hidden_states, memory_output['modulation']])
                    hidden_states = memory_layers['memory_norm_layer'](hidden_states)
                    all_hidden_states.append(hidden_states) # Store state after memory

        final_hidden_states = hidden_states # Use the output of the last layer/memory step

        # --- Observer State & Collapse ---
        pooled_states = self.layers_dict['observer_pool'](final_hidden_states)
        observer_state = self.layers_dict['observer_projection'](pooled_states)

        collapse_layers = {
            'gamma_logits_dense': self.layers_dict['collapse_gamma_logits_dense'],
            'gamma_activation': self.layers_dict['collapse_gamma_activation'],
            'standard_projection_dense': self.layers_dict['collapse_standard_projection_dense'],
            'observer_tile_repeat': self.layers_dict['collapse_observer_tile_repeat'],
            'xo_elementwise_multiply': self.layers_dict['collapse_xo_elementwise_multiply'],
            'xo_sum_dense': self.layers_dict['collapse_xo_sum_dense'],
            'alignment_projection_dense': self.layers_dict['collapse_alignment_projection_dense'],
            'negate_gamma_scale': self.layers_dict['collapse_negate_gamma_scale'],
            'one_minus_gamma_activation': self.layers_dict['collapse_one_minus_gamma_activation'],
            'gamma_reshape': self.layers_dict['collapse_gamma_reshape'],
            'one_minus_gamma_reshape': self.layers_dict['collapse_one_minus_gamma_reshape'],
            'scale_standard_multiply': self.layers_dict['collapse_scale_standard_multiply'],
            'scale_observer_multiply': self.layers_dict['collapse_scale_observer_multiply'],
            'combine_projections_add': self.layers_dict['collapse_combine_projections_add']
        }
        collapse_output = observer_collapse(
            config=self.config,
            hidden_states=final_hidden_states,
            observer_state=observer_state,
            layers=collapse_layers
        )
        logits = collapse_output['logits']
        gamma = collapse_output['gamma']
        # observer_alignment_from_collapse = collapse_output['observer_alignment'] # This is calculated inside collapse

        # Recompute observer_alignment for the loss function output, using the final hidden states
        # Use the layers designated for the loss calculation path
        observer_tiled_loss = self.layers_dict['loss_observer_tile_repeat'](observer_state)
        elementwise_product_loss = self.layers_dict['loss_xo_elementwise_multiply']([final_hidden_states, observer_tiled_loss])
        observer_alignment_loss = self.layers_dict['loss_xo_sum_dense'](elementwise_product_loss)


        # --- Loss Calculation (only if targets are provided) ---
        if target_one_hot is not None:
            # Prepare y_pred list for the loss function
            # Note: observer_alignment_loss is the <x, o> term needed by the loss
            y_pred_for_loss = [logits, gamma, final_hidden_states, observer_state, observer_alignment_loss]
            # Calculate the total loss using the instantiated loss function
            total_loss = self.loss_calculator(y_true=target_one_hot, y_pred=y_pred_for_loss)
            # Add the calculated total loss to the model's list of losses
            self.add_loss(total_loss) # Keras will track this as 'loss'

        # --- Prepare outputs for metrics ---
        # Return a dictionary containing logits and other tensors needed for external metrics
        outputs = {
            'logits': logits,
            'gamma': gamma,
            'final_hidden_states': final_hidden_states,
            'observer_state': observer_state,
            'observer_alignment': observer_alignment_loss, # Renaming for clarity in metrics
            'projected_prime': projected_prime,
            'projected_phase': projected_phase
        }
        return outputs

    # Optional: Add get_config for serialization
    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        # Custom layers might need registration or custom handling here
        # For now, assume standard Keras layers and basic custom layers work
        # Pop custom layer configs if necessary before passing to cls
        return cls(config)

# Example Usage (Illustrative)
# config = { 'vocab_size': 10000, 'sequence_length': 64, 'embedding_dim': 128, 'num_layers': 2, 'num_heads': 4 }
# model = ResonantKnowledgeModel(config)
#
# batch_size = 4
# input_tok = tf.random.uniform((batch_size, config['sequence_length']), maxval=config['vocab_size'], dtype=tf.int32)
# input_pos = tf.range(config['sequence_length'])
# input_pos = tf.expand_dims(input_pos, 0)
# input_pos = tf.tile(input_pos, [batch_size, 1])
#
# outputs = model((input_tok, input_pos), training=True)
# logits_output = outputs[0]
# print("Logits shape:", logits_output.shape) # Should be [batch, seq_len, vocab_size]

# To compile with custom loss:
# loss_fn = custom_loss(model.config) # Assuming custom_loss takes the config dict
# model.compile(optimizer='adam', loss=loss_fn)
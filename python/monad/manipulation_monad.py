import tensorflow as tf
import math
import numpy as np
import logging # Use logging instead of console

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManipulationMonad:
    """
    Implementation of the Manipulation Monad Symbolic Processor (Python version).

    Tracks symbolic state (sigma), entropy, resonance, and parity derived
    from model tensors and transformation history. Detects collapse conditions.
    """
    def __init__(self, config=None):
        if config is None:
            config = {}

        # --- Default Configurations ---
        default_config = {
            'resonance_threshold': 0.7,
            'entropy_min': 0.2,
            'entropy_max': 0.3,
            'primes': [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
            'attractors': [ # Canonical Attractor Set A
                "110110.110110", # 108-like
                "100100.100100", # 144-like
                "110110110.110", # 432-like
                "111000.000111", # mirror states
                "101010.010101", # binary harmonics
            ],
            'distance_entropy_weight': 0.1,
            'operation_costs': {
                'rotateBits': 0.05, 'foldStructure': 0.08, 'primeProbe': 0.12,
                'xorReflect': 0.07, 'shiftRadix': 0.04, 'noop': 0.00,
                'identity': 0.00, 'customOpX': 0.10, 'default': 0.01
            },
            'sigma_source_tensor': 'hidden_states', # Default source tensor name
            'radix_position': 'middle' # Default radix position
        }

        self.config = {**default_config, **config}
        # Ensure primes are correctly sourced
        self.config['primes'] = config.get('primes', default_config['primes'])

        self.symbolic_state = None # sigma (binary string with radix)
        self.transformation_history = [] # T (list of {'op': name, 'params': ...})
        self.symbolic_entropy = 0.0 # E(sigma)
        self.resonance = 0.0 # R(sigma)
        self.parity = None # Parity(sigma) ('even' or 'odd')
        self.collapse_info = {'collapsed': False, 'collapse_value': None}

    def update(self, model_data, transform_op=None):
        """
        Updates the Monad's internal state based on the current model state.

        Args:
            model_data (dict): Dictionary containing tensors from the model
                               (e.g., {'hidden_states': tf.Tensor, ...}).
            transform_op (dict, optional): Info about the last transformation applied
                                           {'op': name, 'params': ...}. Defaults to None.
        """
        if transform_op and transform_op.get('op'):
            self.transformation_history.append(transform_op)

        source_tensor_name = self.config['sigma_source_tensor']
        source_tensor = model_data.get(source_tensor_name)

        if source_tensor is None:
            logger.warning(f"ManipulationMonad: Source tensor '{source_tensor_name}' not found in model_data.")
            self._reset_state_on_error()
            return

        # --- Derive sigma (potentially involves synchronous tensor access) ---
        sigma = self._derive_sigma(source_tensor)
        self.symbolic_state = sigma

        if sigma is not None: # Check if sigma derivation was successful
            self.symbolic_entropy = self._calculate_symbolic_entropy(sigma)
            self.resonance = self._calculate_resonance(sigma)
            self.parity = self._calculate_parity(sigma)
            self.check_for_collapse()
        else:
            self._reset_state_on_error()

    def _reset_state_on_error(self):
        """Resets internal state variables typically on error."""
        self.symbolic_state = None
        self.symbolic_entropy = 0.0
        self.resonance = 0.0
        self.parity = None
        self.collapse_info = {'collapsed': False, 'collapse_value': None}

    def check_for_collapse(self):
        """Checks if the current symbolic state meets the collapse conditions."""
        if self.symbolic_state is None:
            self.collapse_info = {'collapsed': False, 'collapse_value': None}
            return

        meets_resonance = self.resonance >= self.config['resonance_threshold']
        meets_entropy = (self.config['entropy_min'] <= self.symbolic_entropy <= self.config['entropy_max'])
        meets_parity = self.parity == 'even'

        if meets_resonance and meets_entropy and meets_parity:
            collapse_value = self._calculate_collapse_value(self.symbolic_state)
            self.collapse_info = {
                'collapsed': True,
                'collapse_value': collapse_value # The target attractor string sigma'
            }
        else:
            self.collapse_info = {'collapsed': False, 'collapse_value': None}

    def get_symbolic_entropy(self):
        """Returns the calculated symbolic entropy E(σ)."""
        return self.symbolic_entropy

    def get_collapse_value(self):
        """Returns the target attractor string sigma' if a collapse occurred, else None."""
        return self.collapse_info['collapse_value']

    def is_collapsed(self):
        """Returns True if the state is collapsed, False otherwise."""
        return self.collapse_info['collapsed']

    # --- Private Calculation Methods ---

    def _derive_sigma(self, tensor):
        """
        Derives the symbolic state string sigma from a model tensor.
        Warning: Uses .numpy(), which can impact performance if called frequently within a graph.
        """
        try:
            if not isinstance(tensor, tf.Tensor):
                 logger.error("ManipulationMonad._derive_sigma: Input is not a TensorFlow tensor.")
                 return None
            # Check if tensor is empty or has zero size
            if tf.size(tensor) == 0:
                 logger.warning("ManipulationMonad._derive_sigma: Input tensor is empty.")
                 return "" # Return empty string for empty tensor

            # Handle potential batch dimension - process first batch item for sigma
            tensor_to_process = tensor
            if len(tensor.shape) > 1 and tensor.shape[0] > 1:
                # logger.info("ManipulationMonad._derive_sigma: Input tensor has batch > 1. Using first item.")
                tensor_to_process = tensor[0:1, ...] # Take first item, keep dims

            # Flatten and get numpy array - THIS IS SYNCHRONOUS
            flat_values = tf.reshape(tensor_to_process, [-1]).numpy()

            if flat_values.size == 0:
                return ""

            min_val = np.min(flat_values)
            max_val = np.max(flat_values)
            val_range = max_val - min_val

            # Binarize based on normalized value or midpoint (0.5)
            if val_range == 0:
                # Handle constant tensor case
                binary_bits = ['1' if v >= 0.5 else '0' for v in flat_values]
            else:
                binary_bits = ['1' if (v - min_val) / val_range >= 0.5 else '0' for v in flat_values]

            # Place radix point based on config
            radix_pos_config = self.config.get('radix_position', 'middle')
            if radix_pos_config == 'middle':
                mid_point = len(binary_bits) // 2
            else: # Default to middle if config is invalid
                 mid_point = len(binary_bits) // 2

            s_l = "".join(binary_bits[:mid_point])
            s_r = "".join(binary_bits[mid_point:])
            return f"{s_l}.{s_r}"

        except Exception as e:
            logger.error(f"ManipulationMonad._derive_sigma Error: {e}", exc_info=True)
            return None

    def _calculate_shannon_entropy(self, binary_string):
        """Calculates Shannon entropy H(s) for a binary string."""
        if not binary_string:
            return 0.0
        length = len(binary_string)
        if length == 0:
            return 0.0

        ones = binary_string.count('1')
        p1 = ones / length
        p0 = 1.0 - p1

        entropy = 0.0
        if p1 > 0:
            entropy -= p1 * math.log2(p1)
        if p0 > 0:
            entropy -= p0 * math.log2(p0)

        return entropy if not math.isnan(entropy) else 0.0

    def _calculate_symbolic_entropy(self, sigma):
        """Calculates total symbolic entropy E(sigma)."""
        parts = sigma.split('.')
        s_l = parts[0] if len(parts) > 0 else ""
        s_r = parts[1] if len(parts) > 1 else ""

        h_l = self._calculate_shannon_entropy(s_l)
        h_r = self._calculate_shannon_entropy(s_r)

        # Simple average of entropies
        base_entropy = (h_l + h_r) / 2.0

        operation_cost_sum = sum(
            self.config['operation_costs'].get(t.get('op', 'default'), self.config['operation_costs']['default'])
            for t in self.transformation_history
        )

        return base_entropy + operation_cost_sum

    def _are_complementary(self, str1, str2):
        """Checks if two binary strings are bitwise complements."""
        if not str1 or not str2 or len(str1) != len(str2):
            return False
        for char1, char2 in zip(str1, str2):
            if not ((char1 == '0' and char2 == '1') or (char1 == '1' and char2 == '0')):
                return False
        return True

    def _calculate_phase_match(self, sigma, p):
        """Calculates phase match score phi_p(sigma)."""
        binary_string = sigma.replace('.', '')
        n = len(binary_string)
        if p <= 0 or n < 2 * p:
            return 0.0

        total_score = 0.0
        window_count = 0

        for i in range(n - 2 * p + 1):
            pattern1 = binary_string[i : i + p]
            pattern2 = binary_string[i + p : i + 2 * p]
            if pattern1 == pattern2:
                total_score += 1.0
            elif self._are_complementary(pattern1, pattern2):
                total_score += 0.5
            window_count += 1

        return total_score / window_count if window_count > 0 else 0.0

    def _calculate_resonance(self, sigma):
        """Calculates Resonance coefficient R(sigma)."""
        resonance_sum = 0.0
        primes = self.config.get('primes', [])

        for p in primes:
            if p > 0: # Ensure prime is positive
                phi_p = self._calculate_phase_match(sigma, p)
                resonance_sum += (1.0 / p) * phi_p
        return resonance_sum

    def _calculate_parity(self, sigma):
        """Calculates Parity(sigma)."""
        binary_string = sigma.replace('.', '')
        ones_count = binary_string.count('1')
        return 'even' if ones_count % 2 == 0 else 'odd'

    def _hamming_distance(self, s1, s2):
        """Calculates Hamming distance between two binary strings (padded)."""
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)
        # Pad shorter string with '0' at the end
        p1 = s1.ljust(max_len, '0')
        p2 = s2.ljust(max_len, '0')

        hamming = sum(c1 != c2 for c1, c2 in zip(p1, p2))
        return hamming

    def _symbolic_distance(self, sigma1, sigma2):
        """Calculates entropy-weighted symbolic distance D(sigma, sigma')."""
        b1 = sigma1.replace('.', '')
        b2 = sigma2.replace('.', '')

        hamming = self._hamming_distance(b1, b2)

        # Calculate entropy penalty using Shannon entropy on the full binary string
        e1 = self._calculate_shannon_entropy(b1)
        e2 = self._calculate_shannon_entropy(b2)
        entropy_penalty = abs(e1 - e2)

        # Apply weighting
        distance = hamming + self.config['distance_entropy_weight'] * entropy_penalty
        return distance

    def _calculate_collapse_value(self, sigma):
        """Calculates the collapse value (nearest attractor string)."""
        attractors = self.config.get('attractors', [])
        if not attractors:
            logger.warning("ManipulationMonad: No attractors defined in config.")
            return None

        nearest_attractor = None
        min_distance = float('inf')

        for attractor in attractors:
            distance = self._symbolic_distance(sigma, attractor)
            if distance < min_distance:
                min_distance = distance
                nearest_attractor = attractor

        return nearest_attractor

    # Placeholder for transformation logic if needed later
    # def _update_transformation(self, sigma):
    #     return {'type': "placeholder_transform"}
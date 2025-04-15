# Manipulation Monad Symbolic Engine Functions (Python version)
import tensorflow as tf
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
OPERATION_COSTS = {
    'rotateBits': 0.05,
    'foldStructure': 0.08,
    'primeProbe': 0.12,
    'xorReflect': 0.07,
    'shiftRadix': 0.04,
    'noop': 0.0,
    'identity': 0.0,
    'customOpX': 0.10,
    'default': 0.01
}

ATTRACTORS = [
    "110110.110110", # 108-like
    "100100.100100", # 144-like
    "110110110.110", # 432-like
    "111000.000111", # mirror states
    "101010.010101", # binary harmonics
]

# --- Functions ---

def derive_sigma(tensor, radix_position='middle'):
    """
    Derives the symbolic state string sigma from a model tensor.
    Warning: Uses .numpy(), which can impact performance if called frequently within a graph.

    Args:
        tensor (tf.Tensor): Input tensor, assumed to be [batch?, ..., features].
                            Uses first batch item if batched.
        radix_position (str): Where to place the radix ('middle', etc.). Defaults to 'middle'.

    Returns:
        str | None: Sigma string "s_L.s_R" or None on error.
    """
    try:
        if not isinstance(tensor, tf.Tensor):
            logger.error("symbolic_engine.derive_sigma: Input is not a TensorFlow tensor.")
            return None
        if tf.size(tensor) == 0:
            logger.warning("symbolic_engine.derive_sigma: Input tensor is empty.")
            return "." # Return empty parts

        # Handle potential batch dimension
        tensor_to_process = tensor
        if len(tensor.shape) > 1 and tensor.shape[0] > 1:
            tensor_to_process = tensor[0:1, ...] # Take first item

        # Flatten and get numpy array - SYNCHRONOUS
        flat_values = tf.reshape(tensor_to_process, [-1]).numpy()

        if flat_values.size == 0:
            return "."

        min_val = np.min(flat_values)
        max_val = np.max(flat_values)
        val_range = max_val - min_val

        # Binarize
        if val_range == 0:
            binary_bits = ['1' if v >= 0.5 else '0' for v in flat_values]
        else:
            binary_bits = ['1' if (v - min_val) / val_range >= 0.5 else '0' for v in flat_values]

        # Place radix point
        if radix_position == 'middle':
            mid_point = len(binary_bits) // 2
        else: # Default to middle
            mid_point = len(binary_bits) // 2

        s_l = "".join(binary_bits[:mid_point])
        s_r = "".join(binary_bits[mid_point:])
        return f"{s_l}.{s_r}"

    except Exception as e:
        logger.error(f"symbolic_engine.derive_sigma Error: {e}", exc_info=True)
        return None

def calculate_shannon_entropy(binary_string):
    """Calculates Shannon entropy H(s) for a binary string."""
    if not binary_string: return 0.0
    length = len(binary_string)
    if length == 0: return 0.0

    ones = binary_string.count('1')
    p1 = ones / length
    p0 = 1.0 - p1

    entropy = 0.0
    if p1 > 0: entropy -= p1 * math.log2(p1)
    if p0 > 0: entropy -= p0 * math.log2(p0)

    return entropy if not math.isnan(entropy) else 0.0

def compute_entropy(sigma, transformation_history=None):
    """
    Calculates symbolic entropy E(sigma) = H(sigma) + operation costs.

    Args:
        sigma (str): The symbolic state string "s_L.s_R".
        transformation_history (list, optional): List of dicts {'op': name, 'params': ...}. Defaults to [].

    Returns:
        float: The symbolic entropy E(sigma).
    """
    if transformation_history is None:
        transformation_history = []
    if sigma is None: return 0.0 # Handle null sigma

    parts = sigma.split('.')
    s_l = parts[0] if len(parts) > 0 else ""
    s_r = parts[1] if len(parts) > 1 else ""

    # Calculate Shannon entropy (average of left/right parts)
    h_l = calculate_shannon_entropy(s_l)
    h_r = calculate_shannon_entropy(s_r)
    base_entropy = (h_l + h_r) / 2.0

    # Add costs of transformations
    operation_cost_sum = sum(
        OPERATION_COSTS.get(t.get('op', 'default'), OPERATION_COSTS['default'])
        for t in transformation_history
    )

    return base_entropy + operation_cost_sum

def are_complementary(str1, str2):
    """Checks if two binary strings are bitwise complements."""
    if not str1 or not str2 or len(str1) != len(str2):
        return False
    return all(c1 != c2 for c1, c2 in zip(str1, str2))

def calculate_phase_match(sigma, p):
    """Calculates phase match score phi_p(sigma)."""
    if sigma is None: return 0.0
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
        elif are_complementary(pattern1, pattern2):
            total_score += 0.5
        window_count += 1

    return total_score / window_count if window_count > 0 else 0.0

def calculate_resonance(sigma, primes):
    """
    Calculates Resonance coefficient R(sigma).

    Args:
        sigma (str): The symbolic state string "s_L.s_R".
        primes (list): Array of primes to check resonance against.

    Returns:
        float: The resonance coefficient R(sigma).
    """
    if sigma is None: return 0.0
    resonance_sum = 0.0
    for p in primes:
        if p > 0:
            phi_p = calculate_phase_match(sigma, p)
            resonance_sum += (1.0 / p) * phi_p
    return resonance_sum

def compute_parity(sigma):
    """Calculates Parity(sigma)."""
    if sigma is None: return None
    ones = sigma.replace('.', '').count('1')
    return "even" if ones % 2 == 0 else "odd"

def hamming_distance(s1, s2):
    """Calculates Hamming distance between two binary strings (padded)."""
    if s1 is None or s2 is None: return float('inf') # Or handle differently
    b1 = s1.replace('.', '')
    b2 = s2.replace('.', '')
    len1, len2 = len(b1), len(b2)
    max_len = max(len1, len2)
    p1 = b1.ljust(max_len, '0')
    p2 = b2.ljust(max_len, '0')

    hamming = sum(c1 != c2 for c1, c2 in zip(p1, p2))
    return hamming

def symbolic_distance(sigma1, sigma2, entropy_weight=0.1):
    """
    Calculates entropy-weighted symbolic distance D(sigma1, sigma2).

    Args:
        sigma1 (str): First sigma string.
        sigma2 (str): Second sigma string.
        entropy_weight (float, optional): Weight for the entropy penalty. Defaults to 0.1.

    Returns:
        float: The symbolic distance.
    """
    if sigma1 is None or sigma2 is None: return float('inf') # Or handle differently
    ham_dist = hamming_distance(sigma1, sigma2)

    # Calculate Shannon entropy on the full binary string representation
    e1 = calculate_shannon_entropy(sigma1.replace('.', ''))
    e2 = calculate_shannon_entropy(sigma2.replace('.', ''))
    entropy_penalty = abs(e1 - e2)

    return ham_dist + entropy_weight * entropy_penalty

def get_collapse_value(sigma, attractor_list, entropy_weight):
    """
    Finds the nearest attractor string to sigma based on symbolic distance.

    Args:
        sigma (str): The current sigma string.
        attractor_list (list): The list of attractor strings.
        entropy_weight (float): Weight for entropy penalty in distance.

    Returns:
        str | None: The nearest attractor string or None if list is empty or sigma is None.
    """
    if sigma is None or not attractor_list:
        return None

    min_d = float('inf')
    best_attractor = None
    for attractor in attractor_list:
        d = symbolic_distance(sigma, attractor, entropy_weight)
        if d < min_d:
            min_d = d
            best_attractor = attractor
    return best_attractor

def is_collapsed(sigma, entropy, resonance, parity, config):
    """
    Checks if a sigma state meets the collapse conditions.

    Args:
        sigma (str): The sigma string.
        entropy (float): Pre-calculated E(sigma).
        resonance (float): Pre-calculated R(sigma).
        parity (str): Pre-calculated Parity(sigma).
        config (dict): Configuration object with thresholds
                       ('entropy_min', 'entropy_max', 'resonance_threshold').

    Returns:
        bool: True if collapsed, False otherwise.
    """
    if sigma is None: return False
    meets_resonance = resonance >= config.get('resonance_threshold', 0.7)
    meets_entropy = (config.get('entropy_min', 0.2) <= entropy <= config.get('entropy_max', 0.3))
    meets_parity = parity == 'even'
    return meets_resonance and meets_entropy and meets_parity

# Export dict (less common in Python, users import specific functions)
# __all__ = [
#     'derive_sigma', 'compute_entropy', 'calculate_shannon_entropy',
#     'compute_parity', 'calculate_resonance', 'calculate_phase_match',
#     'are_complementary', 'symbolic_distance', 'hamming_distance',
#     'get_collapse_value', 'is_collapsed', 'OPERATION_COSTS', 'ATTRACTORS'
# ]
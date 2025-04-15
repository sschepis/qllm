// Manipulation Monad Symbolic Engine Functions
// Based on provided formalism and definitions.

const tf = require('@tensorflow/tfjs-node'); // Keep for deriveSigma

const operationCosts = {
  rotateBits: 0.05,
  foldStructure: 0.08,
  primeProbe: 0.12,
  xorReflect: 0.07,
  shiftRadix: 0.04,
  noop: 0.0,
  identity: 0.0, // Added alias for clarity
  default: 0.01
};

const attractors = [
  "110110.110110", // 108-like
  "100100.100100", // 144-like
  "110110110.110", // 432-like
  "111000.000111", // mirror states
  "101010.010101", // binary harmonics
];

/**
 * Derives the symbolic state string sigma from a model tensor (e.g., hiddenStates).
 * @param {tf.Tensor} tensor - Input tensor, assumed to be [batch?, ..., features]. Uses first batch item if batched.
 * @returns {string | null} Sigma string "s_L.s_R" or null on error.
 */
function deriveSigma(tensor) {
  try {
    if (!tensor || typeof tensor.shape === 'undefined' || tensor.isDisposed) {
      console.error("symbolicEngine.deriveSigma: Invalid or disposed tensor input.");
      return null;
    }

    // Handle potential batch dimension - process first batch item for sigma
    let tensorToProcess = tensor;
    if (tensor.shape.length > 1 && tensor.shape[0] > 1) {
      tensorToProcess = tensor.slice(0, 1); // Take first item
    }

    const flatTensor = tensorToProcess.flatten();
    const values = flatTensor.dataSync(); // Synchronous!
    tf.dispose(flatTensor);
    if (tensorToProcess !== tensor) tf.dispose(tensorToProcess);

    if (values.length === 0) return "."; // Return empty parts

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    let binaryBits = [];
    if (range === 0) {
      binaryBits = values.map(v => (v >= 0.5 ? '1' : '0'));
    } else {
      binaryBits = values.map(v => ((v - min) / range >= 0.5 ? '1' : '0'));
    }

    // Place radix point at midpoint
    const midPoint = Math.floor(binaryBits.length / 2);
    const s_L = binaryBits.slice(0, midPoint).join('');
    const s_R = binaryBits.slice(midPoint).join('');
    return `${s_L}.${s_R}`;

  } catch (error) {
    console.error("symbolicEngine.deriveSigma Error:", error);
    if (tensor && !tensor.isDisposed) tf.dispose(tensor);
    return null;
  }
}

/** Calculates Shannon entropy H(s) for a binary string. */
function calculateShannonEntropy(binaryString) {
    if (!binaryString || binaryString.length === 0) return 0;
    const len = binaryString.length;
    let ones = 0;
    for (let i = 0; i < len; i++) {
        if (binaryString[i] === '1') ones++;
    }
    const p1 = ones / len;
    const p0 = 1 - p1;

    let entropy = 0;
    if (p1 > 0) entropy -= p1 * Math.log2(p1);
    if (p0 > 0) entropy -= p0 * Math.log2(p0);
    return isNaN(entropy) ? 0 : entropy;
}


/**
 * Calculates symbolic entropy E(sigma) = H(sigma) + operation costs.
 * @param {string} sigma - The symbolic state string "s_L.s_R".
 * @param {Array<object>} transformationHistory - Array of {op: name, params}.
 * @returns {number} The symbolic entropy E(sigma).
 */
function computeEntropy(sigma, transformationHistory = []) {
  const parts = sigma.split('.');
  const s_L = parts[0] || "";
  const s_R = parts[1] || "";

  // Calculate Shannon entropy (average of left/right parts)
  const h_L = calculateShannonEntropy(s_L);
  const h_R = calculateShannonEntropy(s_R);
  const baseEntropy = (h_L + h_R) / 2; // Using average as per previous implementation

  // Add costs of transformations
  let operationCostSum = 0;
  for (const transform of transformationHistory) {
    operationCostSum += (operationCosts[transform.op] || operationCosts.default);
  }

  return baseEntropy + operationCostSum;
}

/** Checks if two binary strings are bitwise complements. */
function areComplementary(str1, str2) {
  if (!str1 || !str2 || str1.length !== str2.length) return false;
  return str1.split('').every((v, i) => v !== str2[i]);
}

/** Calculates phase match score phi_p(sigma). */
function calculatePhaseMatch(sigma, p) {
    const binaryString = sigma.replace('.', '');
    if (p <= 0 || binaryString.length < 2 * p) return 0;

    let totalScore = 0;
    let windowCount = 0;

    for (let i = 0; i <= binaryString.length - 2 * p; i++) {
        const pattern1 = binaryString.substring(i, i + p);
        const pattern2 = binaryString.substring(i + p, i + 2 * p);
        if (pattern1 === pattern2) {
            totalScore += 1.0;
        } else if (areComplementary(pattern1, pattern2)) {
            totalScore += 0.5;
        }
        windowCount++;
    }
    return windowCount > 0 ? totalScore / windowCount : 0;
}

/**
 * Calculates Resonance coefficient R(sigma).
 * @param {string} sigma - The symbolic state string "s_L.s_R".
 * @param {Array<number>} primes - Array of primes to check resonance against.
 * @returns {number} The resonance coefficient R(sigma).
 */
function calculateResonance(sigma, primes) {
  let resonanceSum = 0;
  for (const p of primes) {
    if (p > 0) {
      const phi_p = calculatePhaseMatch(sigma, p);
      resonanceSum += (1 / p) * phi_p;
    }
  }
  return resonanceSum;
}

/** Calculates Parity(sigma). */
function computeParity(sigma) {
  const ones = (sigma.match(/1/g) || []).length;
  return ones % 2 === 0 ? "even" : "odd";
}

/** Calculates Hamming distance between two binary strings (padded). */
function hammingDistance(s1, s2) {
    const b1 = s1.replace('.', '');
    const b2 = s2.replace('.', '');
    const maxLen = Math.max(b1.length, b2.length);
    const p1 = b1.padEnd(maxLen, '0');
    const p2 = b2.padEnd(maxLen, '0');

    let hamming = 0;
    for (let i = 0; i < maxLen; i++) {
        if (p1[i] !== p2[i]) hamming++;
    }
    return hamming;
}

/**
 * Calculates entropy-weighted symbolic distance D(sigma1, sigma2).
 * @param {string} sigma1 - First sigma string.
 * @param {string} sigma2 - Second sigma string.
 * @param {number} [entropyWeight=0.1] - Weight for the entropy penalty.
 * @returns {number} The symbolic distance.
 */
function symbolicDistance(sigma1, sigma2, entropyWeight = 0.1) {
  const hamming = hammingDistance(sigma1, sigma2);

  // Calculate Shannon entropy on the full binary string representation
  const e1 = calculateShannonEntropy(sigma1.replace('.', ''));
  const e2 = calculateShannonEntropy(sigma2.replace('.', ''));
  const entropyPenalty = Math.abs(e1 - e2);

  return hamming + entropyWeight * entropyPenalty;
}

/**
 * Finds the nearest attractor string to sigma based on symbolic distance.
 * @param {string} sigma - The current sigma string.
 * @param {Array<string>} attractorList - The list of attractor strings.
 * @param {number} entropyWeight - Weight for entropy penalty in distance.
 * @returns {string | null} The nearest attractor string or null if list is empty.
 */
function getCollapseValue(sigma, attractorList, entropyWeight) {
  if (!attractorList || attractorList.length === 0) return null;

  let minD = Infinity;
  let bestAttractor = null;
  for (let attractor of attractorList) {
    const d = symbolicDistance(sigma, attractor, entropyWeight);
    if (d < minD) {
      minD = d;
      bestAttractor = attractor;
    }
  }
  return bestAttractor;
}

/**
 * Checks if a sigma state meets the collapse conditions.
 * @param {string} sigma - The sigma string.
 * @param {number} entropy - Pre-calculated E(sigma).
 * @param {number} resonance - Pre-calculated R(sigma).
 * @param {string} parity - Pre-calculated Parity(sigma).
 * @param {object} config - Configuration object with thresholds (entropyMin, entropyMax, resonanceThreshold).
 * @returns {boolean} True if collapsed, false otherwise.
 */
function isCollapsed(sigma, entropy, resonance, parity, config) {
  const meetsResonance = resonance >= config.resonanceThreshold;
  const meetsEntropy = entropy >= config.entropyMin && entropy <= config.entropyMax;
  const meetsParity = parity === 'even';
  return (sigma !== null && meetsResonance && meetsEntropy && meetsParity);
}

module.exports = {
  deriveSigma,
  computeEntropy,
  calculateShannonEntropy, // Export helper if needed elsewhere
  computeParity,
  calculateResonance,
  calculatePhaseMatch, // Export helper if needed
  areComplementary, // Export helper
  symbolicDistance,
  hammingDistance, // Export helper
  getCollapseValue,
  isCollapsed,
  operationCosts, // Export constants
  attractors      // Export constants
};
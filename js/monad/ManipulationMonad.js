const tf = require('@tensorflow/tfjs-node'); // Keep for potential future tensor ops

/**
 * Implementation of the Manipulation Monad Symbolic Processor
 * based on the formalism provided (rkm/src/paper.md and clarifications).
 *
 * Tracks symbolic state (sigma), entropy, resonance, and parity derived
 * from model tensors and transformation history. Detects collapse conditions.
 */
class ManipulationMonad {
  constructor(config = {}) {
    // --- Confirmed Defaults & Configurations ---
    const defaultConfig = {
      resonanceThreshold: 0.7,
      entropyMin: 0.2,
      entropyMax: 0.3,
      primes: [11, 13, 17, 19, 23, 29, 31, 37, 41, 43], // Default if not in config
      attractors: [ // Canonical Attractor Set A
        "110110.110110", // 108-like
        "100100.100100", // 144-like
        "110110110.110", // 432-like
        "111000.000111", // mirror states
        "101010.010101", // binary harmonics
      ],
      distanceEntropyWeight: 0.1, // Weight for entropy penalty in distance D
      operationCosts: { // Default costs cost(t_i)
        'rotateBits': 0.05,
        'foldStructure': 0.08,
        'primeProbe': 0.12,
        'xorReflect': 0.07, // Added from table
        'shiftRadix': 0.04, // Added from table
        'noop': 0.00,       // Added from table
        'identity': 0.00,   // Added from table
        'customOpX': 0.10,  // Added from table
        'default': 0.01
      },
      sigmaSourceTensor: 'hiddenStates', // Confirmed: Use hiddenStates
      radixPosition: 'middle' // Confirmed: Place radix at midpoint
    };

    this.config = { ...defaultConfig, ...config };
    // Ensure primes are taken from config if provided, otherwise use default
    this.config.primes = config.primes || defaultConfig.primes;

    this.symbolicState = null; // sigma (binary string with radix)
    this.transformationHistory = []; // T (list of {op: name, params})
    this.symbolicEntropy = 0.0; // E(sigma)
    this.resonance = 0.0; // R(sigma)
    this.parity = null; // Parity(sigma) ('even' or 'odd')
    this.collapseInfo = { collapsed: false, collapseValue: null }; // collapseValue is the target attractor string sigma'
  }

  /**
   * Updates the Monad's internal state based on the current model state.
   * @param {object} modelData - Object containing tensors from the model (e.g., {hiddenStates: Tensor, ...}).
   * @param {object} [transformOp={}] - Optional info about the last transformation applied {op: name, params}.
   */
  update(modelData, transformOp = {}) {
    if (transformOp.op) {
        this.transformationHistory.push(transformOp);
    }

    const sourceTensor = modelData[this.config.sigmaSourceTensor];
    if (!sourceTensor) {
        console.warn(`ManipulationMonad: Source tensor '${this.config.sigmaSourceTensor}' not found in modelData.`);
        this._resetStateOnError();
        return;
    }

    // --- Note: Synchronous tensor data retrieval ---
    const sigma = this._deriveSigma(sourceTensor);
    this.symbolicState = sigma;

    if (sigma !== null) { // Check if sigma derivation was successful
        this.symbolicEntropy = this._calculateSymbolicEntropy(sigma);
        this.resonance = this._calculateResonance(sigma);
        this.parity = this._calculateParity(sigma);
        this.checkForCollapse();
    } else {
        this._resetStateOnError();
    }
  }

  _resetStateOnError() {
    this.symbolicState = null;
    this.symbolicEntropy = 0.0;
    this.resonance = 0.0;
    this.parity = null;
    this.collapseInfo = { collapsed: false, collapseValue: null };
  }

  /** Checks if the current symbolic state meets the collapse conditions. */
  checkForCollapse() {
    if (this.symbolicState === null) {
        this.collapseInfo = { collapsed: false, collapseValue: null };
        return;
    }

    const meetsResonance = this.resonance >= this.config.resonanceThreshold;
    const meetsEntropy = this.symbolicEntropy >= this.config.entropyMin && this.symbolicEntropy <= this.config.entropyMax;
    const meetsParity = this.parity === 'even';

    if (meetsResonance && meetsEntropy && meetsParity) {
      const collapseValue = this._calculateCollapseValue(this.symbolicState);
      this.collapseInfo = {
        collapsed: true,
        collapseValue: collapseValue // The target attractor string sigma'
      };
    } else {
      this.collapseInfo = { collapsed: false, collapseValue: null };
    }
  }

  /** Returns the calculated symbolic entropy E(σ). */
  getSymbolicEntropy() {
    return this.symbolicEntropy; // Return dynamically calculated value
  }

  /** Returns the collapse value (target attractor string sigma') if a collapse occurred. */
  getCollapseValue() {
    // Note: Formalism o_new = o + γ * X(σ)_collapse implies X is numerical.
    // Calculation yields nearest attractor *string*. Conversion needed if used numerically.
    return this.collapseInfo.collapseValue;
  }

  // --- Private Calculation Methods ---

  /** Derives the symbolic state string sigma from a model tensor. */
  _deriveSigma(tensor) {
    try {
      if (!tensor || typeof tensor.shape === 'undefined' || tensor.isDisposed) {
          console.error("ManipulationMonad._deriveSigma: Invalid or disposed tensor input.");
          return null;
      }

      // Handle potential batch dimension - process first batch item for sigma
      let tensorToProcess = tensor;
      if (tensor.shape.length > 1 && tensor.shape[0] > 1) {
          // console.warn("ManipulationMonad._deriveSigma: Input tensor has batch > 1. Using first item.");
          tensorToProcess = tensor.slice(0, 1); // Take first item
      }

      const flatTensor = tensorToProcess.flatten();
      const values = flatTensor.dataSync(); // Synchronous!
      tf.dispose(flatTensor);
      if (tensorToProcess !== tensor) tf.dispose(tensorToProcess); // Dispose slice if created

      if (values.length === 0) return "";

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
      console.error("ManipulationMonad._deriveSigma Error:", error);
      if (tensor && !tensor.isDisposed) tf.dispose(tensor); // Try to dispose original tensor on error
      return null;
    }
  }

  /** Calculates Shannon entropy H(s) for a binary string. */
  _calculateShannonEntropy(binaryString) {
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
    // Handle NaN case if p=0 or p=1 (log2(0) is -Infinity)
    return isNaN(entropy) ? 0 : entropy;
  }

  /** Calculates total symbolic entropy E(sigma). */
  _calculateSymbolicEntropy(sigma) {
    const parts = sigma.split('.');
    const s_L = parts[0] || "";
    const s_R = parts[1] || "";

    const h_L = this._calculateShannonEntropy(s_L);
    const h_R = this._calculateShannonEntropy(s_R);

    const totalLength = s_L.length + s_R.length;
    // Use simple average of entropies if weighted average is complex
    // const baseEntropy = totalLength > 0 ? (h_L * s_L.length + h_R * s_R.length) / totalLength : 0;
    const baseEntropy = (h_L + h_R) / 2; // Simpler average

    let operationCostSum = 0;
    for (const transform of this.transformationHistory) {
      operationCostSum += (this.config.operationCosts[transform.op] || this.config.operationCosts.default);
    }

    return baseEntropy + operationCostSum;
  }

  /** Checks if two binary strings are bitwise complements. */
  _areComplementary(str1, str2) {
    if (!str1 || !str2 || str1.length !== str2.length) return false;
    for (let i = 0; i < str1.length; i++) {
      if (!((str1[i] === '0' && str2[i] === '1') || (str1[i] === '1' && str2[i] === '0'))) {
        return false;
      }
    }
    return true;
  }

  /** Calculates phase match score phi_p(sigma). */
  _calculatePhaseMatch(sigma, p) {
    const binaryString = sigma.replace('.', '');
    if (p <= 0 || binaryString.length < 2 * p) return 0; // Ensure p is positive

    let totalScore = 0;
    let windowCount = 0;

    for (let i = 0; i <= binaryString.length - 2 * p; i++) {
      const pattern1 = binaryString.substring(i, i + p);
      const pattern2 = binaryString.substring(i + p, i + 2 * p);
      if (pattern1 === pattern2) {
        totalScore += 1.0;
      } else if (this._areComplementary(pattern1, pattern2)) {
        totalScore += 0.5;
      }
      windowCount++;
    }

    return windowCount > 0 ? totalScore / windowCount : 0;
  }

  /** Calculates Resonance coefficient R(sigma). */
  _calculateResonance(sigma) {
    let resonanceSum = 0;
    const primes = this.config.primes;

    for (const p of primes) {
      if (p > 0) { // Ensure prime is positive
        const phi_p = this._calculatePhaseMatch(sigma, p);
        resonanceSum += (1 / p) * phi_p;
      }
    }
    return resonanceSum;
  }

  /** Calculates Parity(sigma). */
  _calculateParity(sigma) {
    const binaryString = sigma.replace('.', '');
    let onesCount = 0;
    for (let i = 0; i < binaryString.length; i++) {
      if (binaryString[i] === '1') onesCount++;
    }
    return (onesCount % 2 === 0) ? 'even' : 'odd';
  }

  /** Calculates Hamming distance between two binary strings (padded). */
  _hammingDistance(s1, s2) {
    const maxLen = Math.max(s1.length, s2.length);
    const p1 = s1.padEnd(maxLen, '0');
    const p2 = s2.padEnd(maxLen, '0');

    let hamming = 0;
    for (let i = 0; i < maxLen; i++) {
      if (p1[i] !== p2[i]) hamming++;
    }
    return hamming;
  }

  /** Calculates entropy-weighted symbolic distance D(sigma, sigma'). */
  _symbolicDistance(sigma1, sigma2) {
      const b1 = sigma1.replace('.', '');
      const b2 = sigma2.replace('.', '');

      const hamming = this._hammingDistance(b1, b2);

      // Calculate entropy penalty
      // Note: Using _calculateShannonEntropy on the *full* sigma string here
      const e1 = this._calculateShannonEntropy(b1);
      const e2 = this._calculateShannonEntropy(b2);
      const entropyPenalty = Math.abs(e1 - e2);

      // Apply weighting (using a simple addition as per formula structure)
      // D = Hamming + EntropyPenalty (Weighting factor seems missing in formula, using 1.0)
      // Let's assume the formula meant Hamming + weight * EntropyPenalty
      return hamming + this.config.distanceEntropyWeight * entropyPenalty;
  }


  /** Calculates the collapse value (nearest attractor string). */
  _calculateCollapseValue(sigma) {
    const attractors = this.config.attractors;
    if (!attractors || attractors.length === 0) {
        console.warn("ManipulationMonad: No attractors defined in config.");
        return null;
    }

    let nearestAttractor = null;
    let minDistance = Infinity;

    for (const attractor of attractors) {
      const distance = this._symbolicDistance(sigma, attractor);

      if (distance < minDistance) {
        minDistance = distance;
        nearestAttractor = attractor;
      }
    }
    return nearestAttractor;
  }

  // _updateTransformation remains a placeholder
  _updateTransformation(sigma) {
    return { type: "placeholder_transform" };
  }
}

module.exports = ManipulationMonad;
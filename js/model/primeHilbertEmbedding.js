const tf = require('@tensorflow/tfjs-node');
const { SineActivation } = require('../layers/TrigActivations'); // Import SineActivation
const ScaleLayer = require('../layers/ScaleLayer'); // Import ScaleLayer

/**
 * Prime Hilbert Embedding implementation (Extracted from ResonantKnowledgeModel)
 * Implements Prime Hilbert Embedding (Section 2 of formalism)
 * Projects token embeddings into prime-dimensional subspaces with position-dependent phase shifts.
 *
 * @param {Object} config - Configuration object (sequenceLength, primes)
 * @param {Tensor} inputs - Input token tensor [batch, seq_len]
 * @param {Tensor} positions - Position tensor [batch, seq_len]
 * @param {Object} layers - Object containing pre-instantiated layer instances. Expected keys:
 *                          embeddingLayer, reshapeLayer, projectionLayers (object),
 *                          scaleLayers (object), sineLayers (object),
 *                          multiplyLayers (object), concatLayer.
 * @returns {Tensor} - The combined prime embedding [batch, seq_len, sum_of_primes]
 */
function primeHilbertEmbedding(config, inputs, positions, layers) {
  const { primes, sequenceLength } = config; // Only need primes and seqLength from config now

  // Use pre-instantiated embedding layer
  const baseEmbedding = layers.embeddingLayer.apply(inputs);

  // Reshape positions to ensure it's a 3D tensor with shape [batch, seq_len, 1]
  // This assumes 'positions' input is [batch, seq_len]
  // Use pre-instantiated reshape layer
  // Note: Ensure the reshape layer in the main class is configured with targetShape: [sequenceLength, 1]
  const reshapedPositions = layers.reshapeLayer.apply(positions);

  // Track the embedding parts that we'll concatenate later
  const embeddingParts = [];

  // Project into prime subspaces
  // Removed unused 'offset' variable
  for (let i = 0; i < primes.length; i++) {
    const prime = primes[i];

    // Project base embedding to prime dimension
    // Use pre-instantiated projection layer for this prime
    const projectionLayer = layers.projectionLayers[prime];
    if (!projectionLayer) throw new Error(`Missing projection layer for prime ${prime}`);
    const projection = projectionLayer.apply(baseEmbedding);

    // Formalism: E(w, n) = ⊕ (P_pi(e_w) * sin(2πn/pi))
    // No additional phase offset in the formalism (as implemented in the class method).

    // Use layers API exclusively with the symbolic positions tensor
    const scaleFactor = (2 * Math.PI) / prime;
    // Use pre-instantiated scale layer for this prime
    const scaleLayer = layers.scaleLayers[prime];
    if (!scaleLayer) throw new Error(`Missing scale layer for prime ${prime}`);
    // Set scale factor dynamically if needed, or assume it was set during construction
    // scaleLayer.scaleFactor = scaleFactor; // This won't work if layer is already built
    // Assuming scale factor was set correctly during layer construction in the main class
    const angle = scaleLayer.apply(reshapedPositions);

    // Apply sine activation directly to the scaled angle (2πn/pi) using custom layer
    // Use pre-instantiated sine activation layer for this prime
    const sineLayer = layers.sineLayers[prime];
    if (!sineLayer) throw new Error(`Missing sine layer for prime ${prime}`);
    const sinAngle = sineLayer.apply(angle);

    // Multiply symbolic projection and symbolic sinAngle
    // Use pre-instantiated multiply layer for this prime
    const multiplyLayer = layers.multiplyLayers[prime];
    if (!multiplyLayer) throw new Error(`Missing multiply layer for prime ${prime}`);
    const posEncoding = multiplyLayer.apply([projection, sinAngle]);

    // Store this part of the embedding
    embeddingParts.push(posEncoding);
  }

  // Concatenate all embedding parts along the last dimension
  // Use pre-instantiated concatenate layer
  const embedding = layers.concatLayer.apply(embeddingParts);

  // Note: The output dimension will be the sum of primes, not necessarily embeddingDim.
  // The createModel function handles projecting this back to embeddingDim if needed.
  return embedding;
}

module.exports = primeHilbertEmbedding;
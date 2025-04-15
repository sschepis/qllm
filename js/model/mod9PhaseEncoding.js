const tf = require('@tensorflow/tfjs-node');
const { SineActivation, CosineActivation } = require('../layers/TrigActivations'); // Import custom activations

/**
 * Implements Mod 9 Harmonic Phase Encoding (Section 3 of formalism) - Extracted from ResonantKnowledgeModel
 * Approximates G_φ = exp(i·2π·φ(w)/9) using sin/cos components due to symbolic limitations.
 * Uses dense projections and custom activations.
 *
 * @param {Object} config - Configuration object (unused after refactor).
 * @param {Tensor} inputs - Input token tensor [batch, seq_len].
 * @param {Object} layers - Object containing pre-instantiated layer instances. Expected keys:
 *                          tokenIndicesEmbedding, mod9ProjectionLayer, sineDenseLayer,
 *                          sineActivationLayer, cosineDenseLayer, cosineActivationLayer, concatLayer.
 * @returns {Tensor} - The phase encoding [batch, seq_len, 2] (cos, sin components).
 */
function mod9PhaseEncoding(config, inputs, layers) { // Added layers parameter
  // Config no longer needed directly here

  // Using embedding to get token indices (symbolic) - This seems unnecessary if 'inputs' are already indices.
  // The original class method applied this to 'inputs', assuming 'inputs' were token indices.
  // Let's keep the embedding layer for now, but note it might be redundant if inputs are already indices.
  // Use pre-instantiated embedding layer
  const tokenIndices = layers.tokenIndicesEmbedding.apply(inputs);

  // We can't use tf.mod directly in symbolic mode, so we'll approximate
  // by using a custom approach with layers (as done in the class method)

  // Project to 9 units for mod-9 representation (approximation)
  // Use pre-instantiated dense layer
  const mod9Projection = layers.mod9ProjectionLayer.apply(tokenIndices);

  // Generate sine component using our custom layer
  // Dense layer before SineActivation allows learning the mapping from the softmax distribution to the sine value
  // Use pre-instantiated dense layer
  const sineDense = layers.sineDenseLayer.apply(mod9Projection);

  // Use pre-instantiated activation layer
  const sineComponent = layers.sineActivationLayer.apply(sineDense);

  // Generate cosine component using our custom layer
  // Use pre-instantiated dense layer
  const cosineDense = layers.cosineDenseLayer.apply(mod9Projection);

  // Use pre-instantiated activation layer
  const cosineComponent = layers.cosineActivationLayer.apply(cosineDense);

  // Concatenate real (cos) and imaginary (sin) parts
  // Use pre-instantiated concatenate layer
  const phaseEncoding = layers.concatLayer.apply([cosineComponent, sineComponent]);

  return phaseEncoding;
}

module.exports = mod9PhaseEncoding;
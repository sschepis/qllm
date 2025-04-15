const tf = require('@tensorflow/tfjs-node');
const ScaleLayer = require('../layers/ScaleLayer'); // Import ScaleLayer

/**
 * Implements Observer-Conditioned Collapse (Section 5 of formalism) - Extracted from ResonantKnowledgeModel
 * Formalism: ℓ = softmax((1 - γ)Wx + γ·⟨x, o⟩) where γ = σ(V_o o)
 * Uses projection of alignment term to vocab size.
 *
 * @param {Object} config - Configuration object (sequenceLength).
 * @param {Tensor} hiddenStates - Hidden states tensor [batch, seq_len, embeddingDim].
 * @param {Tensor} observerState - Observer state tensor [batch, embeddingDim].
 * @param {Object} layers - Object containing pre-instantiated layer instances.
 * @returns {Object} - Object containing { logits: [batch, seq_len, vocabSize], gamma: [batch, 1] }.
 */
function observerCollapse(config, hiddenStates, observerState, layers) { // Added layers parameter
  const { sequenceLength } = config; // Only need sequenceLength from config

  // Calculate gamma (observer influence factor)
  // Calculate gamma = σ(V_o o)
  // First, get the pre-activation logits for gamma
  // Use pre-instantiated layers
  const gammaLogits = layers.gammaLogitsDense.apply(observerState);

  // Apply sigmoid to get gamma
  const gamma = layers.gammaActivation.apply(gammaLogits);

  // Standard projection Wx
  const standardProjection = layers.standardProjectionDense.apply(hiddenStates);

  // Calculate observer alignment term ⟨x, o⟩
  // Tile the 2D observer state 'o' [batch, embeddingDim] across the sequence length
  // Note: repeatVector needs 'n' configured during layer creation
  const observerTiled = layers.observerTileRepeat.apply(observerState);
  // Calculate dot product ⟨x, o⟩ element-wise then sum. Use multiply + sum.
  const elementwiseProduct = layers.xoElementwiseMultiply.apply([hiddenStates, observerTiled]);
  // Sum across the embedding dimension to get the dot product for each position
  // Simulate sum reduction using a dense layer with 1 unit
  const observerAlignment = layers.xoSumDense.apply(elementwiseProduct);

  // Formalism: (1 - γ)Wx + γ·⟨x, o⟩
  // Option 1 from class method: Project alignment to vocab size, then scale by gamma.
  const projectedAlignment = layers.alignmentProjectionDense.apply(observerAlignment);
  const observerInfluence = projectedAlignment; // Renamed for clarity

  // Calculate 1 - gamma using 1 - sigmoid(x) = sigmoid(-x)
  // Negate the gamma logits
  // Note: ScaleLayer needs scaleFactor configured during creation
  const negativeGammaLogits = layers.negateGammaScale.apply(gammaLogits);
  // Apply sigmoid to get 1 - gamma
  const oneMinusGamma = layers.oneMinusGammaActivation.apply(negativeGammaLogits);

  // Combine terms: (1 - γ)Wx + γ·(Projected ⟨x, o⟩)
  // Need to broadcast gamma [batch, 1] and (1-gamma) [batch, 1] to match logits [batch, seq, vocab]
  // Reshape gamma and 1-gamma to [batch, 1, 1] for broadcasting
  const gammaBroadcast = layers.gammaReshape.apply(gamma);
  const oneMinusGammaBroadcast = layers.oneMinusGammaReshape.apply(oneMinusGamma);

  // Scale projections
  const scaledStandard = layers.scaleStandardMultiply.apply([standardProjection, oneMinusGammaBroadcast]);
  // Scale the *projected* alignment by gamma
  const scaledObserver = layers.scaleObserverMultiply.apply([observerInfluence, gammaBroadcast]);

  // Combine the scaled terms
  const combinedLogits = layers.combineProjectionsAdd.apply([scaledStandard, scaledObserver]);

  // Return the combined logits before softmax, assuming loss handles it.
  const modelProjection = combinedLogits;

  return {
    logits: modelProjection, // Shape: [batch, seq, vocab]
    gamma: gamma          // Shape: [batch, 1]
    // Note: observerAlignment [batch, seq, 1] is calculated internally but not returned here.
    // The createModel function re-calculates it if needed for the loss function output.
  };
}

module.exports = observerCollapse;
const tf = require('@tensorflow/tfjs-node');
// Import the CORRECT, externalized resonanceAttention function
const resonanceAttention = require('./resonanceAttention');

/**
 * Implements Resonance Block (Section 4 of formalism) - Extracted from ResonantKnowledgeModel
 * Simplified for symbolic mode: x_out = N(x_in + FFN(N(x_in + Attn(x_in))))
 * Does not implement iterative refinement x⁽ᵗ⁺¹⁾ = N(x⁽ᵗ⁾ + Attn(x⁽ᵗ⁾))
 *
 * @param {Object} config - Configuration object (embeddingDim).
 * @param {Tensor} inputs - Input hidden states [batch, seq_len, embeddingDim].
 * @param {Tensor|null} mask - Attention mask (passed to attention).
 * @param {number} blockIndex - Index of the block (used for selecting layers).
 * @param {Object} layers - Object containing pre-instantiated layer instances for this block.
 *                          Expected keys: attnLayers (object), attnResidualAdd, attnNorm,
 *                          ffn1Dense, ffn2Dense, ffnNorm, ffnResidualAdd.
 * @param {boolean} training - Whether in training mode (passed to attention).
 * @returns {Tensor} - Output hidden states [batch, seq_len, embeddingDim].
 */
function resonanceBlock(config, inputs, mask, blockIndex, layers, training = false) { // Added layers parameter
  // Name prefixing is handled during layer construction
  // const namePrefix = `block_${blockIndex}`;
  // EmbeddingDim might still be needed if not implicitly handled by layer shapes
  const { embeddingDim } = config;

  // First sublayer: Multi-Head Resonance Attention + Residual + Norm
  // Pass blockIndex and training status to the imported resonanceAttention
  // Pass the pre-instantiated attention layers object to resonanceAttention
  const { output: attentionOutput } = resonanceAttention(config, inputs, mask, blockIndex, layers.attnLayers, training);
  const attentionResidual = layers.attnResidualAdd.apply([inputs, attentionOutput]);
  const normalizedAttention = layers.attnNorm.apply(attentionResidual);

  // Second sublayer: Feed-Forward Network + Residual + Norm
  // Use pre-instantiated FFN layers
  const ffn1 = layers.ffn1Dense.apply(normalizedAttention);

  const ffn2 = layers.ffn2Dense.apply(ffn1);

  // Layer normalization *after* FFN (different from original standalone file)
  const normalizedFfn = layers.ffnNorm.apply(ffn2);

  // Residual connection for FFN (connects back to the output of the first sublayer's norm)
  const finalOutput = layers.ffnResidualAdd.apply([normalizedAttention, normalizedFfn]);

  return finalOutput;
}

module.exports = resonanceBlock;
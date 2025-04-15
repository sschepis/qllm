const tf = require('@tensorflow/tfjs-node');

/**
 * Implements Resonant Knowledge Learning Objective (Section 9 of rkm/src/paper.md) - Extracted from ResonantKnowledgeModel
 * L = L_CE + λ₁S_p + λ₂(mod9 Phase Dispersion) + λ₃(1 - ⟨x, o⟩)
 * Note: The λ₄E(σ) term is handled separately in the custom training loop.
 * Approximated for symbolic execution.
 *
 * @param {Object} config - Configuration object containing lambda hyperparameters (lambda1, lambda2, lambda3)
 * @returns {Function} - Loss function (targets, predictions) => Tensor
 */
function customLoss(config) { // Removed monad parameter
  // Loss hyperparameters from config
  const lambda1 = config.lambda1 || 0.1; // S_p: Symbolic prime entropy weight (approximated)
  const lambda2 = config.lambda2 || 0.1; // Mod9 Phase Dispersion weight (approximated)
  const lambda3 = config.lambda3 || 0.2; // Observer alignment weight (1 - <x,o>) (approximated)
  // const lambda4 = config.lambda4 || 0.1; // Monad penalty handled in training loop

  // Return the actual loss function closure
  return (targets, predictions) => {
    // predictions = [logits, gamma, hiddenStates, observerState, observerAlignment] from createModel outputs
    // Ensure the order matches the model's output definition
    const [logits, gamma, hiddenStates, observerState, observerAlignment] = predictions;

    // 1. Primary loss: L_CE (Categorical Cross-Entropy)
    const ceLoss = tf.losses.softmaxCrossEntropy(targets, logits);

    // 2. Symbolic Prime Entropy (S_p) Approximation
    // Using L2 norm on logits as a proxy. Encourages smoother distributions.
    const logitsL2 = tf.mean(tf.sum(tf.square(logits), -1));
    const entropyPenalty = tf.mul(logitsL2, tf.scalar(lambda1 * 0.01)); // Scaled S_p approximation

    // 3. Mod9 Phase Dispersion Approximation
    // Using L2 norm on hidden states as a proxy. Penalizes large activations.
    const hiddenL2 = tf.mean(tf.sum(tf.square(hiddenStates), -1));
    const dispersionPenalty = tf.mul(hiddenL2, tf.scalar(lambda2 * 0.001)); // Scaled dispersion approximation

    // 4. Observer Alignment Penalty: λ₃·mean(1 - ⟨x, o⟩)
    // Use the observerAlignment tensor passed from the model outputs
    // observerAlignment has shape [batch, seq, 1]
    const alignmentMean = tf.mean(observerAlignment); // Mean over batch, sequence, and final dim (which is 1)
    const oneMinusAlignmentMean = tf.sub(tf.scalar(1.0), alignmentMean);
    const alignmentPenalty = tf.mul(oneMinusAlignmentMean, tf.scalar(lambda3));

    // 5. Monad Symbolic Entropy Penalty: λ₄E(σ) - REMOVED
    // This term is now calculated and added *outside* this loss function in the training loop,
    // as E(σ) depends on Monad state updated *after* the forward pass but *before* gradient calculation.
    // const monadEntropy = monad.getSymbolicEntropy(); // No longer accessible here
    // const monadPenalty = tf.mul(tf.scalar(monadEntropy), tf.scalar(lambda4));
    // Combine losses using tf.add for symbolic compatibility
    let totalLoss = ceLoss;
    totalLoss = tf.add(totalLoss, entropyPenalty);
    totalLoss = tf.add(totalLoss, dispersionPenalty);
    totalLoss = tf.add(totalLoss, alignmentPenalty);
    // totalLoss = tf.add(totalLoss, monadPenalty); // Monad penalty removed

    return totalLoss;
  };
}

module.exports = customLoss;
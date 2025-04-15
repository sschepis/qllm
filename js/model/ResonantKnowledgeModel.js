// Resonant Knowledge Model - Implementing formalism from papers/learn.md
const tf = require('@tensorflow/tfjs-node');
const { SineActivation, CosineActivation } = require('../layers/TrigActivations');
const ScaleLayer = require('../layers/ScaleLayer');
const { digitalRoot } = require('../utils/mathUtils');
// const ManipulationMonad = require('../monad/ManipulationMonad'); // No longer needed here
const resonanceAttention = require('./resonanceAttention'); // Import the external function
const primeHilbertEmbedding = require('./primeHilbertEmbedding'); // Import the external function
const mod9PhaseEncoding = require('./mod9PhaseEncoding'); // Import the external function
const resonanceBlock = require('./resonanceBlock'); // Import the external function
const observerCollapse = require('./observerCollapse'); // Import the external function
const resonantMemory = require('./resonantMemory'); // Import the external function
const customLoss = require('./customLoss'); // Import the external function

/**
 * Resonant Knowledge Model Implementation
 * A neural language model with quantum-inspired processing
 */

// Resonant Knowledge Model Implementation in TensorFlow.js
class ResonantKnowledgeModel { // Implements formalism from papers/learn.md
  constructor(config) {
    this.config = {
      vocabSize: 30000,
      embeddingDim: 512,
      numLayers: 6,
      numHeads: 8,
      primes: [11, 13, 17, 19, 23, 29, 31, 37, 41, 43], // Selected prime dimensions
      batchSize: 32, // Note: batchSize in config might not be used in eager mode
      sequenceLength: 512,
      beta: 1.0, // Default beta for attention sharpening
      memorySize: 128, // Default memory size
      ...config
    };

    // Loss hyperparameters
    this.lambda1 = this.config.lambda1 || 0.1;
    this.lambda2 = this.config.lambda2 || 0.1;
    this.lambda3 = this.config.lambda3 || 0.2;
    this.lambda4 = this.config.lambda4 || 0.1;

    // --- Instantiate ALL layers here ---
    this.layers = {};
    const { vocabSize, embeddingDim, sequenceLength, primes, numLayers, numHeads, beta, memorySize } = this.config;
    const headDim = Math.floor(embeddingDim / numHeads);

    // Embedding Layers (Prime Hilbert + Mod9)
    this.layers.embeddingLayer = tf.layers.embedding({ inputDim: vocabSize, outputDim: embeddingDim, name: 'base_embedding' });
    this.layers.posReshapeLayer = tf.layers.reshape({ targetShape: [sequenceLength, 1], name: 'positions_reshape' });
    this.layers.prime_projectionLayers = {};
    this.layers.prime_scaleLayers = {};
    this.layers.prime_sineLayers = {};
    this.layers.prime_multiplyLayers = {};
    primes.forEach(prime => {
        const pStr = `prime_${prime}`;
        this.layers.prime_projectionLayers[prime] = tf.layers.dense({ units: prime, name: `prime_projection_${prime}` });
        const scaleFactor = (2 * Math.PI) / prime;
        this.layers.prime_scaleLayers[prime] = new ScaleLayer({ scaleFactor: scaleFactor, name: `angle_scale_${prime}` });
        this.layers.prime_sineLayers[prime] = new SineActivation({ name: `sin_angle_${prime}` });
        this.layers.prime_multiplyLayers[prime] = tf.layers.multiply({ name: `pos_encoding_${prime}` });
    });
    this.layers.prime_concatLayer = tf.layers.concatenate({ axis: -1, name: 'prime_embedding_concat' });
    this.layers.prime_finalProjection = tf.layers.dense({ units: embeddingDim, name: 'project_prime_embedding' }); // Project sum of primes to embeddingDim

    this.layers.mod9_tokenIndicesEmbedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: 1, trainable: false, name: 'token_indices_embedding' });
    this.layers.mod9_projectionLayer = tf.layers.dense({ units: 9, activation: 'softmax', name: 'mod9_projection' });
    this.layers.mod9_sineDenseLayer = tf.layers.dense({ units: 1, name: 'sine_dense' });
    this.layers.mod9_sineActivationLayer = new SineActivation({ name: 'sine_component' });
    this.layers.mod9_cosineDenseLayer = tf.layers.dense({ units: 1, name: 'cosine_dense' });
    this.layers.mod9_cosineActivationLayer = new CosineActivation({ name: 'cosine_component' });
    this.layers.mod9_concatLayer = tf.layers.concatenate({ axis: -1, name: 'phase_encoding' });
    this.layers.mod9_finalProjection = tf.layers.dense({ units: embeddingDim, name: 'project_phase_encoding' }); // Project phase (2 dims) to embeddingDim

    this.layers.combineEmbeddingsAdd = tf.layers.add({ name: 'combine_embeddings' });
    this.layers.embeddingNorm = tf.layers.layerNormalization({ name: 'embedding_norm' });

    // Resonance Block Layers (per layer)
    this.layers.resonanceBlocks = [];
    for (let i = 0; i < numLayers; i++) {
        const namePrefix = `block_${i}`;
        const attnPrefix = `${namePrefix}_attn`;
        const blockLayers = {
            // Attention Sub-layers
            attnLayers: {
                queryProjection: tf.layers.dense({ units: embeddingDim, name: `${attnPrefix}_query_projection` }),
                keyProjection: tf.layers.dense({ units: embeddingDim, name: `${attnPrefix}_key_projection` }),
                valueProjection: tf.layers.dense({ units: embeddingDim, name: `${attnPrefix}_value_projection` }),
                qReshape: tf.layers.reshape({ targetShape: [sequenceLength, numHeads, headDim], name: `${attnPrefix}_query_reshape` }),
                kReshape: tf.layers.reshape({ targetShape: [sequenceLength, numHeads, headDim], name: `${attnPrefix}_key_reshape` }),
                vReshape: tf.layers.reshape({ targetShape: [sequenceLength, numHeads, headDim], name: `${attnPrefix}_value_reshape` }),
                qPermute: tf.layers.permute({ dims: [2, 1, 3], name: `${attnPrefix}_query_permute` }),
                kPermute: tf.layers.permute({ dims: [2, 1, 3], name: `${attnPrefix}_key_permute` }),
                vPermute: tf.layers.permute({ dims: [2, 1, 3], name: `${attnPrefix}_value_permute` }),
                qReshape3D: tf.layers.reshape({ targetShape: [-1, sequenceLength, headDim], name: `${attnPrefix}_q_reshape_3d` }),
                qkDenseSim: tf.layers.dense({ units: sequenceLength, useBias: false, name: `${attnPrefix}_qk_dense_sim` }),
                scoresReshape4D: tf.layers.reshape({ targetShape: [numHeads, sequenceLength, sequenceLength], name: `${attnPrefix}_scores_reshape_4d` }),
                scaleScoresLayer: new ScaleLayer({ scaleFactor: 1.0 / Math.sqrt(headDim), name: `${attnPrefix}_scale_scores` }),
                betaSharpeningLayer: new ScaleLayer({ scaleFactor: beta, name: `${attnPrefix}_beta_sharpening` }),
                softmaxLayer: tf.layers.softmax({ axis: -1, name: `${attnPrefix}_attn_weights` }),
                entropySquareMultiply: tf.layers.multiply({ name: `${attnPrefix}_entropy_square` }),
                entropyPool: tf.layers.globalAveragePooling2d({ dataFormat: 'channelsFirst', name: `${attnPrefix}_entropy_pool` }),
                vReshape3D: tf.layers.reshape({ targetShape: [-1, sequenceLength, headDim], name: `${attnPrefix}_v_reshape_3d` }),
                weightsReshape3D: tf.layers.reshape({ targetShape: [-1, sequenceLength, sequenceLength], name: `${attnPrefix}_weights_reshape_3d` }),
                weightsVDenseSim: tf.layers.dense({ units: headDim, useBias: false, name: `${attnPrefix}_weights_v_dense_sim` }),
                weightedValsReshape4D: tf.layers.reshape({ targetShape: [numHeads, sequenceLength, headDim], name: `${attnPrefix}_weighted_vals_4d` }),
                outputPermute: tf.layers.permute({ dims: [2, 1, 3], name: `${attnPrefix}_output_permute` }),
                outputReshape: tf.layers.reshape({ targetShape: [sequenceLength, embeddingDim], name: `${attnPrefix}_output_reshape` }),
                outputDense: tf.layers.dense({ units: embeddingDim, name: `${attnPrefix}_output_dense` })
            },
            // Block Layers
            attnResidualAdd: tf.layers.add({ name: `${namePrefix}_attn_residual` }),
            attnNorm: tf.layers.layerNormalization({ name: `${namePrefix}_attn_norm` }),
            ffn1Dense: tf.layers.dense({ units: embeddingDim * 4, activation: 'gelu', name: `${namePrefix}_ffn1` }),
            ffn2Dense: tf.layers.dense({ units: embeddingDim, name: `${namePrefix}_ffn2` }),
            ffnNorm: tf.layers.layerNormalization({ name: `${namePrefix}_ffn_norm` }),
            ffnResidualAdd: tf.layers.add({ name: `${namePrefix}_ffn_residual` })
        };
        this.layers.resonanceBlocks.push(blockLayers);
    }

    // Resonant Memory Layers (per memory step, e.g., every 2 layers)
    this.layers.resonantMemory = [];
    for (let i = 0; i < numLayers; i++) {
        if (i % 2 === 1) {
            const namePrefix = `memory_${i}`;
            const memoryLayers = {
                projectionDense: tf.layers.dense({ units: memorySize, activation: 'linear', name: `${namePrefix}_projection` }),
                attentionActivation: tf.layers.activation({ activation: 'softmax', name: `${namePrefix}_attention` }),
                valuesDense: tf.layers.dense({ units: embeddingDim, name: `${namePrefix}_values`, kernelInitializer: 'randomNormal' }),
                normLayer: tf.layers.layerNormalization({ name: `${namePrefix}_norm` }),
                modulationDense: tf.layers.dense({ units: embeddingDim, activation: 'tanh', name: `${namePrefix}_modulation` }),
                memoryAddLayer: tf.layers.add({ name: `memory_add_layer_${i}` }), // Add layer for combining memory
                memoryNormLayer: tf.layers.layerNormalization({ name: `memory_norm_layer_${i}` }) // Norm after adding memory
            };
            this.layers.resonantMemory.push(memoryLayers);
        } else {
             this.layers.resonantMemory.push(null); // Placeholder for non-memory layers
        }
    }

    // Observer Collapse Layers
    this.layers.observerPool = tf.layers.globalAveragePooling1d({ name: 'global_avg_pool' });
    this.layers.observerProjection = tf.layers.dense({ units: embeddingDim, activation: 'tanh', name: 'observer_state_projection' });
    this.layers.observerTileRepeat = tf.layers.repeatVector({ n: sequenceLength, name: 'observer_tile_for_output' }); // Note: n needs config
    this.layers.xoElementwiseMultiply = tf.layers.multiply({ name: 'xo_elementwise_for_output' });
    this.layers.xoSumDense = tf.layers.dense({ units: 1, useBias: false, name: 'xo_sum_for_output' });

    this.layers.collapse_gammaLogitsDense = tf.layers.dense({ units: 1, activation: 'linear', name: 'observer_gamma_logits' });
    this.layers.collapse_gammaActivation = tf.layers.activation({ activation: 'sigmoid', name: 'observer_gamma' });
    this.layers.collapse_standardProjectionDense = tf.layers.dense({ units: vocabSize, name: 'output_projection' });
    this.layers.collapse_alignmentProjectionDense = tf.layers.dense({ units: vocabSize, name: 'alignment_vocab_projection' });
    this.layers.collapse_negateGammaScale = new ScaleLayer({ scaleFactor: -1.0, name: 'negate_gamma_logits' });
    this.layers.collapse_oneMinusGammaActivation = tf.layers.activation({ activation: 'sigmoid', name: 'one_minus_gamma' });
    this.layers.collapse_gammaReshape = tf.layers.reshape({ targetShape: [1, 1], name: 'gamma_broadcast_reshape' });
    this.layers.collapse_oneMinusGammaReshape = tf.layers.reshape({ targetShape: [1, 1], name: 'one_minus_gamma_broadcast_reshape' });
    this.layers.collapse_scaleStandardMultiply = tf.layers.multiply({ name: 'scale_standard_proj' });
    this.layers.collapse_scaleObserverMultiply = tf.layers.multiply({ name: 'scale_observer_influence' });
    this.layers.collapse_combineProjectionsAdd = tf.layers.add({ name: 'combine_projections' });

    // Variables will be collected externally after first call.
  }

  // _collectTrainableVariables removed

  // buildModel removed
  
  // digitalRoot is now imported from utils
  
  // primeHilbertEmbedding method removed, logic moved to ./primeHilbertEmbedding.js
  // (Original method code was here)
  
  // mod9PhaseEncoding method removed, logic moved to ./mod9PhaseEncoding.js
  
  // resonanceAttention method removed, logic moved to ./resonanceAttention.js
  
  // resonanceBlock method removed, logic moved to ./resonanceBlock.js
  
  // observerCollapse method removed, logic moved to ./observerCollapse.js
  
  // resonantMemory method removed, logic moved to ./resonantMemory.js
  
  // _buildSymbolicModel removed

  /**
   * Executes the forward pass of the model eagerly using pre-instantiated layers.
   * @param {tf.Tensor} inputTokens - Input token tensor [batch, seq_len].
   * @param {tf.Tensor} positionsInput - Position tensor [batch, seq_len].
   * @param {boolean} [training=false] - Indicates if the model is in training mode.
   * @returns {Array<tf.Tensor>} - Array of output tensors: [logits, gamma, hiddenStates, observerState, observerAlignment].
   */
  call(inputTokens, positionsInput, training = false) {
    // Use pre-instantiated layers stored in this.layers
    return tf.tidy(() => {
      // --- Embedding ---
      const primeEmb = primeHilbertEmbedding(this.config, inputTokens, positionsInput, {
          embeddingLayer: this.layers.embeddingLayer,
          reshapeLayer: this.layers.posReshapeLayer,
          projectionLayers: this.layers.prime_projectionLayers,
          scaleLayers: this.layers.prime_scaleLayers,
          sineLayers: this.layers.prime_sineLayers,
          multiplyLayers: this.layers.prime_multiplyLayers,
          concatLayer: this.layers.prime_concatLayer
      });
      const phaseEnc = mod9PhaseEncoding(this.config, inputTokens, {
          tokenIndicesEmbedding: this.layers.mod9_tokenIndicesEmbedding,
          mod9ProjectionLayer: this.layers.mod9_projectionLayer,
          sineDenseLayer: this.layers.mod9_sineDenseLayer,
          sineActivationLayer: this.layers.mod9_sineActivationLayer,
          cosineDenseLayer: this.layers.mod9_cosineDenseLayer,
          cosineActivationLayer: this.layers.mod9_cosineActivationLayer,
          concatLayer: this.layers.mod9_concatLayer
      });

      const projectedPhase = this.layers.mod9_finalProjection.apply(phaseEnc);
      const projectedPrime = this.layers.prime_finalProjection.apply(primeEmb);
      const embeddingCombined = this.layers.combineEmbeddingsAdd.apply([projectedPrime, projectedPhase]);
      let hiddenStates = this.layers.embeddingNorm.apply(embeddingCombined);

      // --- Resonance Blocks ---
      const attentionMask = null; // Placeholder
      for (let i = 0; i < this.config.numLayers; i++) {
          const blockLayers = this.layers.resonanceBlocks[i];
          hiddenStates = resonanceBlock(this.config, hiddenStates, attentionMask, i, blockLayers, training);

          // --- Resonant Memory ---
          if (i % 2 === 1) {
              const memoryLayers = this.layers.resonantMemory[i];
              if (memoryLayers) { // Check if memory layers exist for this index
                  const { modulation } = resonantMemory(this.config, hiddenStates, i, memoryLayers);
                  hiddenStates = memoryLayers.memoryAddLayer.apply([hiddenStates, modulation]);
                  hiddenStates = memoryLayers.memoryNormLayer.apply(hiddenStates);
              }
          }
      }

      // --- Observer State & Collapse ---
      const pooledStates = this.layers.observerPool.apply(hiddenStates);
      const observerState = this.layers.observerProjection.apply(pooledStates);

      const { logits, gamma } = observerCollapse(this.config, hiddenStates, observerState, {
          gammaLogitsDense: this.layers.collapse_gammaLogitsDense,
          gammaActivation: this.layers.collapse_gammaActivation,
          standardProjectionDense: this.layers.collapse_standardProjectionDense,
          observerTileRepeat: this.layers.observerTileRepeat, // Use the output-specific layer instance
          xoElementwiseMultiply: this.layers.xoElementwiseMultiply, // Use the output-specific layer instance
          xoSumDense: this.layers.xoSumDense, // Use the output-specific layer instance
          alignmentProjectionDense: this.layers.collapse_alignmentProjectionDense,
          negateGammaScale: this.layers.collapse_negateGammaScale,
          oneMinusGammaActivation: this.layers.collapse_oneMinusGammaActivation,
          gammaReshape: this.layers.collapse_gammaReshape,
          oneMinusGammaReshape: this.layers.collapse_oneMinusGammaReshape,
          scaleStandardMultiply: this.layers.collapse_scaleStandardMultiply,
          scaleObserverMultiply: this.layers.collapse_scaleObserverMultiply,
          combineProjectionsAdd: this.layers.collapse_combineProjectionsAdd
      });

      // Recompute observerAlignment using output-specific layers
      const observerTiled_out = this.layers.observerTileRepeat.apply(observerState);
      const elementwiseProduct_out = this.layers.xoElementwiseMultiply.apply([hiddenStates, observerTiled_out]);
      const observerAlignment = this.layers.xoSumDense.apply(elementwiseProduct_out);

      return [logits, gamma, hiddenStates, observerState, observerAlignment];
    });
  }
  
  // customLoss method removed, logic moved to ./customLoss.js
  
  // compileModel method removed (using custom training loop)

  // fit method removed (using custom training loop)
  // Method for prediction - used during inference after model is compiled
  /**
   * Predicts outputs for given inputs using the eager `call` method.
   * Assumes inputs is an object like {inputTokens: Tensor, positionsInput: Tensor}.
   * @param {object} inputs - Object containing input tensors {inputTokens, positionsInput}.
   * @returns {tf.Tensor} - The primary output tensor (logits).
   */
  /**
   * Predicts outputs for given inputs using the eager `call` method.
   * Assumes inputs is an object like {inputTokens: Tensor, positionsInput: Tensor}.
   * @param {object} inputs - Object containing input tensors {inputTokens, positionsInput}.
   * @returns {tf.Tensor} - The primary output tensor (logits).
   */
   predict(inputs) {
    // Use tf.noGrad for inference
    return tf.noGrad(() => {
        // Ensure inputs are tensors, create if not
        const inputTokensTensor = tf.isTensor(inputs.inputTokens) ? inputs.inputTokens : tf.tensor(inputs.inputTokens);
        const positionsInputTensor = tf.isTensor(inputs.positionsInput) ? inputs.positionsInput : tf.tensor(inputs.positionsInput);

        // Run the call method
        const outputs = this.call(inputTokensTensor, positionsInputTensor, false); // training = false
        const logits = outputs[0];
        tf.keep(logits); // Keep the output tensor

        // Dispose input tensors *if they were created here*
        if (!tf.isTensor(inputs.inputTokens)) tf.dispose(inputTokensTensor);
        if (!tf.isTensor(inputs.positionsInput)) tf.dispose(positionsInputTensor);

        return logits;
    });
  }

  /** Returns the collected list of trainable variables */
  getTrainableVariables() {
      if (!this.trainableVariables) {
          // This should ideally not happen if constructor ran correctly
          console.error("Trainable variables requested but not collected. Check constructor logic.");
          return [];
      }
      return this.trainableVariables;
  }
}

module.exports = ResonantKnowledgeModel;
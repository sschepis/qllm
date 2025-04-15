const tf = require('@tensorflow/tfjs-node');
const ResonantKnowledgeModel = require('./model/ResonantKnowledgeModel');
const ScaleLayer = require('./layers/ScaleLayer');
const { createDataset } = require('./utils/datasetUtils');
const { digitalRoot } = require('./utils/mathUtils');

// Export main components
module.exports = {
  ResonantKnowledgeModel,
  ScaleLayer,
  createDataset,
  digitalRoot,
  runModel
};

/**
 * Usage example function to create and test a model
 * @returns {Promise<void>}
 */
async function runModel() {
  // Sample data would be provided here
  const model = new ResonantKnowledgeModel({
    vocabSize: 10000,
    embeddingDim: 256,
    numLayers: 4,
    sequenceLength: 128,
    batchSize: 16
  });
  
  console.log('Model created successfully');
  
  // Print model summary
  model.model.summary();
  
  // Sample input for prediction
  const sampleInput = tf.ones([1, 128], 'int32');
  const prediction = model.predict(sampleInput);
  console.log('Model prediction completed');
}

// Execute if this file is run directly
if (require.main === module) {
  runModel().catch(console.error);
}
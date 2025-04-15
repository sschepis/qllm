const tf = require('@tensorflow/tfjs-node');

/**
 * Create dataset helper function
 * Converts array of input/target pairs into a TensorFlow dataset
 * 
 * @param {Array} data - Array of [input, target] pairs
 * @param {number} batchSize - Batch size for the dataset
 * @returns {tf.data.Dataset} - TensorFlow dataset
 */
function createDataset(data, batchSize = 32) {
  return tf.data.generator(function* () {
    for (const [input, target] of data) {
      yield {
        xs: tf.tensor(input, [1, input.length], 'int32'),
        ys: tf.tensor(target, [1, target.length], 'int32')
      };
    }
  }).batch(batchSize);
}

module.exports = {
  createDataset
};
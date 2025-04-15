const tf = require('@tensorflow/tfjs-node');

// Custom Layer for simple scaling (multiplication by a scalar)
class ScaleLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.scaleFactor = config.scaleFactor;
  }

  // input is a SymbolicTensor
  call(input) {
    // Use tf.mul for broadcasting the scalar factor
    // Ensure input is treated as a single tensor if it's passed as an array
    const inputTensor = Array.isArray(input) ? input[0] : input;
    return tf.mul(inputTensor, tf.scalar(this.scaleFactor));
  }

  // Define computeOutputShape based on input shape
  computeOutputShape(inputShape) {
    // Output shape is the same as input shape
    return inputShape;
  }

  // Define className for serialization/deserialization
  static get className() {
    return 'ScaleLayer';
  }
}

// Register the custom layer
tf.serialization.registerClass(ScaleLayer);

module.exports = ScaleLayer;
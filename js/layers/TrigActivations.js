const tf = require('@tensorflow/tfjs-node');

// Custom Sine Activation Layer
class SineActivation extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  call(inputs) {
    return tf.sin(inputs);
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  static get className() {
    return 'SineActivation';
  }
}

// Custom Cosine Activation Layer
class CosineActivation extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  call(inputs) {
    return tf.cos(inputs);
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  static get className() {
    return 'CosineActivation';
  }
}

// Register these custom layers
tf.serialization.registerClass(SineActivation);
tf.serialization.registerClass(CosineActivation);

module.exports = {
  SineActivation,
  CosineActivation
};
import tensorflow as tf

# Custom Sine Activation Layer
class SineActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """Initializes the SineActivation layer."""
        super(SineActivation, self).__init__(**kwargs)

    def call(self, inputs):
        """Applies the sine activation function."""
        return tf.math.sin(inputs)

    def compute_output_shape(self, input_shape):
        """Returns the output shape, which is the same as the input shape."""
        return input_shape

    def get_config(self):
        """Returns the layer's configuration for serialization."""
        config = super(SineActivation, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer instance from its config."""
        return cls(**config)

# Custom Cosine Activation Layer
class CosineActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """Initializes the CosineActivation layer."""
        super(CosineActivation, self).__init__(**kwargs)

    def call(self, inputs):
        """Applies the cosine activation function."""
        return tf.math.cos(inputs)

    def compute_output_shape(self, input_shape):
        """Returns the output shape, which is the same as the input shape."""
        return input_shape

    def get_config(self):
        """Returns the layer's configuration for serialization."""
        config = super(CosineActivation, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer instance from its config."""
        return cls(**config)

# Keras typically handles registration automatically when saving/loading models
# containing these custom layers, especially if they inherit directly from Layer
# and implement get_config/from_config.

# To make them easily importable:
__all__ = ['SineActivation', 'CosineActivation']
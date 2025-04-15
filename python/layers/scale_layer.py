import tensorflow as tf

# Custom Layer for simple scaling (multiplication by a scalar)
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        """
        Initializes the ScaleLayer.

        Args:
            scale_factor (float): The factor to scale the input by.
            **kwargs: Additional keyword arguments passed to the base Layer class.
        """
        super(ScaleLayer, self).__init__(**kwargs)
        # Ensure scale_factor is stored as a float or compatible numeric type
        self.scale_factor = tf.cast(scale_factor, self.dtype) # Use layer's dtype

    def call(self, inputs):
        """
        Performs the scaling operation.

        Args:
            inputs: The input tensor(s).

        Returns:
            The scaled tensor.
        """
        # tf.multiply handles broadcasting the scalar factor
        return tf.multiply(inputs, self.scale_factor)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: The shape of the input tensor.

        Returns:
            The shape of the output tensor (same as input shape).
        """
        return input_shape

    def get_config(self):
        """
        Returns the layer's configuration for serialization.
        """
        config = super(ScaleLayer, self).get_config()
        # Convert TensorFlow tensor back to a Python float for serialization
        config.update({
            'scale_factor': float(tf.keras.backend.get_value(self.scale_factor))
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer instance from its config.
        """
        return cls(**config)

# Registration is typically handled automatically by Keras when saving/loading models
# if the layer is defined correctly and used within a Keras model.
# Explicit registration like tf.serialization.registerClass is less common for standard Keras layers.
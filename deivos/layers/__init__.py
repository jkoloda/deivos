"""Implementation of custom layers not available in Keras."""

import tensorflow as tf
from tensorflow.keras.layers import Concatenate


class ZeroPaddingChannels(tf.keras.layers.Layer):
    """Zero-pad channels (last tensor dimension)."""

    def __init__(self, channels, **kwargs):
        """Construct zero-padding layer object.

        Parameters
        ----------
        channels : int
            Final depth of resulting tensor.

        """
        self.channels = channels
        super(ZeroPaddingChannels, self).__init__(**kwargs)

    def call(self, x):
        """Apply zero-padding on channels (last dimension).

        Parameters
        ----------
        x : tensor
            Batch of samples that are to be padded.

        Returns
        -------
        padded : tensor
            Padded batch of samples.

        """
        assert self.channels >= x.shape[-1]
        padding_shape = (x.shape).as_list()
        padding_shape[-1] = self.channels - x.shape[-1]
        padding = tf.zeros(shape=padding_shape)

        padded = Concatenate()([x, padding])
        return padded

    def compute_output_shape(self, input_shape):
        """Compute output shape for shape inference."""
        return (input_shape[:-1] + (self.channels,))

    def build(self, input_shape):
        """Build function necessary for correct compilation."""
        super(ZeroPaddingChannels, self).build(input_shape)

    def get_config(self):
        """Get configuration."""
        config = {'depth': self.channels}
        base_config = super(ZeroPaddingChannelsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

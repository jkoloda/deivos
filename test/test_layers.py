"""Tester for custom layers."""

import unittest
import numpy as np
import tensorflow as tf

from deivos.layers import ZeroPaddingChannels


class TestLayers(unittest.TestCase):
    """Tester for custom layers."""

    def test_zero_padding_channel(self):
        """Test ZeroPaddingChannels layer."""
        inputs = tf.random.normal((64, 70, 70, 20))

        for channels in [20, 30, 40, 50, 100]:
            outputs = ZeroPaddingChannels(channels=channels)(inputs)
            self.assertTrue(outputs.shape == \
                            tf.TensorShape((64, 70, 70, channels)))

            self.assertTrue(np.array_equal(inputs, outputs[:, :, :, :20]))
            self.assertTrue(np.all(inputs[:, :, :, 20:] == 0))

        with self.assertRaises(AssertionError):
            outputs = ZeroPaddingChannels(channels=10)(inputs)


if __name__ == '__main__':
    unittest.main()

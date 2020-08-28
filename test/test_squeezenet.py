"""Tester for squeezenet module."""

import unittest
import numpy as np
import tensorflow as tf
from deivos.architectures.squeezenet import (
    default_preprocessor,
    expand,
    fire_module,
    get_model_v10,
    get_model_v11,
    squeeze,
)


class TestLayers(unittest.TestCase):
    """Tester for SqueezeNet architecture."""

    # pylint: disable=too-many-instance-attributes

    def setUp(self):
        self.batch_size = 16
        self.rows = 24
        self.cols = 32
        self.channels = 20
        self.input_shape = (self.batch_size, self.rows,
                            self.cols, self.channels)
        self.default_input_shape = (self.batch_size, 227, 227, 3)
        self.inputs = tf.random.normal(shape=self.input_shape)
        self.default_inputs = tf.random.normal(shape=self.default_input_shape)

    def test_squeeze(self):
        """Test squeeze module."""
        for _ in range(0, 100):
            filters = np.random.randint(low=1, high=2*self.channels)
            output_shape = (self.batch_size, self.rows, self.cols, filters)
            # Check shape after squeezing
            if filters < self.channels:
                outputs = squeeze(self.inputs, name='', filters=filters)
                self.assertTrue(outputs.shape == tf.TensorShape(output_shape))
            else:
                # Squeeze module cannot expand input
                with self.assertRaises(AssertionError):
                    outputs = squeeze(self.inputs, name='', filters=filters)

    def test_expand(self):
        """Test expand module."""
        for _ in range(0, 100):
            filters = np.random.randint(low=1, high=2*self.channels)
            output_shape = (self.batch_size, self.rows, self.cols, 2*filters)
            # Check shape after expanding
            if filters > self.channels:
                outputs = expand(self.inputs, name='', filters=filters)
                self.assertTrue(outputs.shape == tf.TensorShape(output_shape))
            else:
                # Expand module cannot squeeze input
                with self.assertRaises(AssertionError):
                    outputs = expand(self.inputs, name='', filters=filters)

    def test_fire_module(self):
        """Test fire module."""
        # Only test for one number of filters
        # Filter variety is tested by expand and squeeze tests
        filters_in = 10
        for squeeze_expand_ratio in [2, 3, 4]:
            # Expand squeezed dimension for both 1x1 and 3x3 filters
            filters_out = squeeze_expand_ratio * 2 * filters_in
            # No bypass
            output_shape = (self.batch_size, self.rows, self.cols, filters_out)
            outputs = fire_module(self.inputs, name='',
                                  squeeze_filters=filters_in, bypass=False,
                                  squeeze_expand_ratio=squeeze_expand_ratio)
            self.assertTrue(outputs.shape == tf.TensorShape(output_shape))
            # Complex bypass
            outputs = fire_module(self.inputs, name='',
                                  squeeze_filters=filters_in, bypass=True,
                                  squeeze_expand_ratio=squeeze_expand_ratio)
            self.assertTrue(outputs.shape == tf.TensorShape(output_shape))

        # Simple bypass
        for squeeze_expand_ratio in [2, 4, 5]:
            filters_in = self.channels//squeeze_expand_ratio
            # Expand squeezed dimension for both 1x1 and 3x3 filters
            filters_out = squeeze_expand_ratio * 2 * filters_in
            output_shape = (self.batch_size, self.rows, self.cols, filters_out)
            outputs = fire_module(self.inputs, name='',
                                  squeeze_filters=filters_in, bypass=True,
                                  squeeze_expand_ratio=squeeze_expand_ratio)
            self.assertTrue(outputs.shape == tf.TensorShape(output_shape))

    def test_default_preprocessor(self):
        """Test deafult preprocessor."""
        # Version 1.0, default input
        outputs = default_preprocessor(self.default_inputs, version='1.0')
        self.assertTrue(outputs.shape == (self.batch_size, 55, 55, 96))
        # Not default input
        with self.assertRaises(AssertionError):
            outputs = default_preprocessor(self.inputs, '1.0')

        # Version 1.1, default input
        outputs = default_preprocessor(self.default_inputs, version='1.1')
        self.assertTrue(outputs.shape == (self.batch_size, 56, 56, 64))
        # Not default input
        with self.assertRaises(AssertionError):
            outputs = default_preprocessor(self.inputs, version='1.1')

    def test_get_model_v10(self):
        """Test SqueezeNet v1.0."""
        layers = {'fire2_concat': (None, 55, 55, 128),
                  'fire3_concat': (None, 55, 55, 128),
                  'fire4_concat': (None, 55, 55, 256),
                  'fire5_concat': (None, 27, 27, 256),
                  'fire6_concat': (None, 27, 27, 384),
                  'fire7_concat': (None, 27, 27, 384),
                  'fire8_concat': (None, 27, 27, 512),
                  'fire9_concat': (None, 13, 13, 512)}

        for num_classes in [10, 100, 100]:
            for bypass_type in [None, 'simple', 'complex']:
                model = get_model_v10(num_classes=num_classes,
                                      bypass_type=bypass_type)
                for name, shape in layers.items():
                    layer = model.get_layer(name)
                    self.assertTrue(layer.output_shape == shape)
                self.assertTrue(model.output_shape == (None, num_classes))
                del model

    def test_get_model_v11(self):
        """Test SqueezeNet v1.1."""
        layers = {'fire2_concat': (None, 56, 56, 128),
                  'fire3_concat': (None, 56, 56, 128),
                  'fire4_concat': (None, 28, 28, 256),
                  'fire5_concat': (None, 28, 28, 256),
                  'fire6_concat': (None, 14, 14, 384),
                  'fire7_concat': (None, 14, 14, 384),
                  'fire8_concat': (None, 14, 14, 512),
                  'fire9_concat': (None, 14, 14, 512)}

        for num_classes in [10, 100, 100]:
            for bypass_type in [None, 'simple', 'complex']:
                model = get_model_v11(num_classes=num_classes,
                                      bypass_type=bypass_type)
                for name, shape in layers.items():
                    layer = model.get_layer(name)
                    self.assertTrue(layer.output_shape == shape)
                self.assertTrue(model.output_shape == (None, num_classes))
                del model

    def test_squeezenet(self):
        # TODO: Check that corresponding get models have been called
        pass


if __name__ == '__main__':
    unittest.main()

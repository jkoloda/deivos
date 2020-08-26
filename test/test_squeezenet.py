import unittest
import numpy as np
import tensorflow as tf
from deivos.architectures.squeezenet import (
    default_preprocessor,
    expand,
    fire_module,
    get_model_v10,
    squeeze,
)


class TestLayers(unittest.TestCase):
    """Tester for SqueezeNet architecture."""

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
            print(filters_out, self.channels)

    def test_default_preprocessor(self):
        """Test deafult preprocessor."""
        # Default input
        outputs = default_preprocessor(self.default_inputs)
        self.assertTrue(outputs.shape == (self.batch_size, 55, 55, 96))
        # Not default input
        with self.assertRaises(AssertionError):
            outputs = default_preprocessor(self.inputs)

    def test_get_model_v10(self):
        model = get_model_v10(num_classes=10)
        model.summary()


    #
    #
    # def test_squeezenet(self):
    #     model = SqueezeNet.get_model(num_classes=100)
    #     self.assertTrue(model.layers[-1].output_shape == (None, 100))
    #     model.compile(optimizer='nadam', loss='categorical_crossentropy')
    #     del model
    #
    #     model = SqueezeNet.get_model(num_classes=100)
    #     self.assertTrue(model.layers[-1].output_shape == (None, 100))
    #     model.compile(optimizer='nadam', loss='categorical_crossentropy')
    #     del model
    #
    #     input = Input(shape=(110, 110, 3))
    #     output = Conv2D(filters=96,
    #                     kernel_size=(3, 3),
    #                     strides=(2, 2),
    #                     activation='relu',
    #                     padding='same',
    #                     )(input)
    #     prep = Model(input, output)
    #     model = SqueezeNet.get_model(num_classes=2, preprocessing=prep)
    #     self.assertTrue(model.layers[-1].output_shape == (None, 2))
    #     del model


if __name__ == '__main__':
    unittest.main()

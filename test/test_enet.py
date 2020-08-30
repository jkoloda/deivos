"""Tester for Enet module."""

import unittest
import numpy as np

import tensorflow as tf
from deivos.architectures.enet import (
    # bottleneck,
    bottleneck_core,
    # bottleneck_merge,
    # ENet,
    initial_block,
    knead,
    # stage1,
    # stage2,
    # stage3,
    # stage4,
    # stage5,
)
# from keras.models import Model
# from keras.layers import Input


class TestEnet(unittest.TestCase):
    """Tester for Enet architecture."""

    def test_initial_block(self):
        """Test initial block."""
        batch_size = 20
        rows = 512
        cols = 512
        num_channels = 3
        input_shape = (batch_size, rows, cols, num_channels)
        output_shape = (batch_size, rows//2, cols//2, 16)
        inputs = tf.random.normal(shape=input_shape)
        outputs = initial_block(inputs)
        self.assertTrue(outputs.shape == tf.TensorShape(output_shape))

    def test_knead(self):
        """Test knead module. """
        batch_size = 16
        rows = 24
        cols = 32
        num_channels = 48
        input_shape = (batch_size, rows, cols, num_channels)
        inputs = tf.random.normal(shape=input_shape)

        # Check compression, no change, and expansion
        for output_depth in [num_channels - 8, num_channels, num_channels + 8]:
            output_shape = (batch_size, rows, cols, output_depth)
            output_shape_down = (batch_size, rows//2, cols//2, output_depth)

            # Check comrpession without downsampling
            outputs = knead(inputs, output_depth, name='c', downsampling=False)
            self.assertTrue(outputs.shape == tf.TensorShape(output_shape))

            # Check comrpession with downsampling
            outputs = knead(inputs, output_depth, name='c', downsampling=True)
            self.assertTrue(outputs.shape == tf.TensorShape(output_shape_down))

#
#     def test_bottleneck_merge(self):
#         right_shape = (16, 32, 32, 20)
#         left_shapes = [(16, 32, 32, 20), (16, 32, 32, 15), (16, 32, 32, 30)]
#
#         right = K.random_normal(right_shape)
#
#         for left_shape in left_shapes:
#             left = K.random_normal(left_shape)
#             merge = bottleneck_merge(left, right)
#             self.assertTrue(K.int_shape(merge) == K.int_shape(right))
#
    def test_bottleneck_core(self):
        """Test bottleneck core module."""
        batch_size = 16
        rows = 24
        cols = 32
        num_channels = 48
        input_shape = (batch_size, rows, cols, num_channels)
        inputs = tf.random.normal(shape=input_shape)

        for dilation in [1, 2, 5]:
            outputs = bottleneck_core(inputs, name='_', dilation=dilation)
            self.assertTrue(outputs.shape == tf.TensorShape(input_shape))

        for asymmetric in [1, 2, 5]:
            outputs = bottleneck_core(inputs, name='_', asymmetric=asymmetric)
            self.assertTrue(outputs.shape == tf.TensorShape(input_shape))

        outputs = bottleneck_core(inputs, name='_', upsampling=True)
        self.assertTrue(outputs.shape == tf.TensorShape((batch_size, rows*2,
                                                        cols*2, num_channels)))
#
#
#     def test_bottleneck(self):
#         """ Test bottleneck module. """
#         batch_size = 16
#         rows = 24
#         cols = 32
#         num_channels = 40
#         output_depth = 48
#         projection_ratio = 4
#         input_shape = (batch_size, rows, cols, num_channels)
#         output_shape = (batch_size, rows, cols, output_depth)
#         output_shape_down = (batch_size, rows//2, cols//2, output_depth)
#         output_shape_up = (batch_size, rows*2, cols*2, output_depth)
#         index_shape = (batch_size, rows//2, cols//2, num_channels)
#
#         inputs = K.random_normal(input_shape)
#
#         # Test dilation & downsampling
#         for dilation in [1, 2, 3, 4]:
#             for ds in [True, False]:
#                 out, idx = bottleneck(inputs, output_depth=output_depth,
#                                       dropout=0.1, downsampling=ds, name='b',
#                                       dilation=dilation, asymmetric=0,
#                                       projection_ratio=projection_ratio,
#                                       argmax_pooling=True)
#                 if ds is True:
#                     self.assertTrue(K.int_shape(out) == output_shape_down)
#                     self.assertTrue(K.int_shape(idx) == (index_shape))
#                 else:
#                     self.assertTrue(K.int_shape(out) == output_shape)
#
#         # Test asymmetric & downsampling
#         for asymmetric in [1, 2, 5]:
#             for ds in [True, False]:
#                 out, idx = bottleneck(inputs, output_depth=output_depth,
#                                       dropout=0.1, downsampling=ds, name='b',
#                                       dilation=1, asymmetric=asymmetric,
#                                       projection_ratio=projection_ratio,
#                                       argmax_pooling=True)
#                 if ds is True:
#                     self.assertTrue(K.int_shape(out) == output_shape_down)
#                     self.assertTrue(K.int_shape(idx) == (index_shape))
#                 else:
#                     self.assertTrue(K.int_shape(out) == output_shape)
#
#         # Test upsampling
#         out, _ = bottleneck(inputs, output_depth=output_depth, dropout=0.1,
#                             upsampling=True, name='b',
#                             projection_ratio=projection_ratio)
#         self.assertTrue(K.int_shape(out) == output_shape_up)
#
#     def test_stage1(self):
#         input_shape = (20, 256, 256, 16)
#         inputs = K.random_normal(input_shape)
#
#         x, indices = stage1(inputs, argmax_pooling=False)
#         self.assertTrue(indices is None)
#         self.assertTrue(K.int_shape(x) == (20, 128, 128, 64))
#
#         x, indices = stage1(inputs, argmax_pooling=True)
#         self.assertTrue(K.int_shape(indices) == (20, 128, 128, 16))
#         self.assertTrue(K.int_shape(x) == (20, 128, 128, 64))
#
#     def test_stage2(self):
#         input_shape = (20, 128, 128, 64)
#         inputs = K.random_normal(input_shape)
#
#         x, indices = stage2(inputs, argmax_pooling=False)
#         self.assertTrue(indices is None)
#         self.assertTrue(K.int_shape(x) == (20, 64, 64, 128))
#
#         x, indices = stage2(inputs, argmax_pooling=True)
#         self.assertTrue(K.int_shape(indices) == (20, 64, 64, 64))
#         self.assertTrue(K.int_shape(x) == (20, 64, 64, 128))
#
#     def test_stage3(self):
#         input_shape = (20, 64, 64, 128)
#         inputs = K.random_normal(input_shape)
#         x = stage3(inputs)
#         self.assertTrue(K.int_shape(x) == (20, 64, 64, 128))
#
#     def test_stage4(self):
#         input_shape = (20, 64, 64, 128)
#         output_shape = (20, 128, 128, 64)
#
#         inputs = K.random_normal(input_shape)
#         unpool_indices = K.ones((20, 64, 64, 64), dtype='int64')
#
#         x = stage4(inputs)
#         self.assertTrue(K.int_shape(x) == output_shape)
#         x = stage4(inputs, unpool_indices=unpool_indices)
#         self.assertTrue(K.int_shape(x) == output_shape)
#
#         inputs = Input((64, 64, 128))
#         unpool_indices = Input((64, 64, 64), dtype='int64')
#
#         out = stage4(inputs, unpool_indices=unpool_indices)
#         model = Model([inputs, unpool_indices], out)
#         mask = ['max_unpooling' in l.name for l in model.layers]
#         self.assertTrue(np.any(mask))
#
#         out = stage4(inputs)
#         model = Model(inputs, out)
#         mask = ['max_unpooling' in l.name for l in model.layers]
#         self.assertTrue(not np.all(mask))
#
#     def test_stage5(self):
#         input_shape = (20, 128, 128, 64)
#         inputs = K.random_normal(input_shape)
#         unpool_indices = K.ones((20, 128, 128, 16), dtype='int64')
#
#         x = stage5(inputs)
#         self.assertTrue(K.int_shape(x) == (20, 256, 256, 16))
#         x = stage5(inputs, unpool_indices=unpool_indices)
#         self.assertTrue(K.int_shape(x) == (20, 256, 256, 16))
#
#         inputs = Input((128, 128, 64))
#         unpool_indices = Input((128, 128, 16), dtype='int64')
#
#         out = stage5(inputs, unpool_indices=unpool_indices)
#         model = Model([inputs, unpool_indices], out)
#         mask = ['max_unpooling' in l.name for l in model.layers]
#         self.assertTrue(np.any(mask))
#
#         out = stage5(inputs)
#         model = Model(inputs, out)
#         mask = ['max_unpooling' in l.name for l in model.layers]
#         self.assertTrue(not np.all(mask))
#
#
#     def assert_layer_shape(self, model, layer_name, shape):
#         for layer in model.layers:
#             if layer.name == layer_name:
#                 self.assertTrue(layer.output_shape == shape)
#                 return
#         self.assertTrue(False)
#
#
#     def test_enet(self):
#         input_shape = (512, 512, 3)
#         num_classes = 4
#         model = ENet.get_model(input_shape, num_classes, reshape='False')
#
#         # Test stage 1
#         stage1 = ['bottleneck1.' + str(i) + '_prelu_final' for i in range(0, 5)]
#         shape1 = (None, 128, 128, 64)
#         for layer in stage1:
#             self.assert_layer_shape(model, layer, shape1)
#
#         # Test stage 2
#         stage2 = ['bottleneck2.' + str(i) + '_prelu_final' for i in range(0, 9)]
#         shape2 = (None, 64, 64, 128)
#         for layer in stage2:
#             self.assert_layer_shape(model, layer, shape2)
#
#         # Test stage 3
#         stage3 = ['bottleneck3.' + str(i) + '_prelu_final' for i in range(1, 9)]
#         shape3 = (None, 64, 64, 128)
#         for layer in stage3:
#             self.assert_layer_shape(model, layer, shape3)
#
#         # Test stage 4
#         stage4 = ['bottleneck4.' + str(i) + '_prelu_final' for i in range(0, 3)]
#         shape4 = (None, 128, 128, 64)
#         for layer in stage4:
#             self.assert_layer_shape(model, layer, shape4)
#
#         # Test stage 5
#         stage5 = ['bottleneck5.' + str(i) + '_prelu_final' for i in range(0, 2)]
#         shape5 = (None, 256, 256, 16)
#         for layer in stage5:
#             self.assert_layer_shape(model, layer, shape5)
#
#         # Test output
#         self.assertTrue(model.layers[-1].output_shape == \
#             (None, 512, 512, num_classes))
#
#         del model
#
#         model = ENet.get_model(input_shape, num_classes, reshape=True)
#         self.assertTrue(model.layers[-1].output_shape == \
#             (None, 512*512, num_classes))
#         del model
#
#
if __name__ == '__main__':
    unittest.main()

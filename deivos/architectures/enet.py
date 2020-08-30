"""Implementation of  Efficient Neural Network (ENet) as described in [1].

References
----------
[1] A.Paszke, A. Chaurasia, S. Kim, E. Culurciello, "ENet: A Deep Neural
Network Architecture for Real-Time Semantic Segmentation", arXiv, Jun. 2016.


NOTE: Current implementation uses simple MaxPooling and Deconv layers instead
of proper unpooling (unavailable in Keras). We still need to build custom
Pooling/Unpooling layers.

"""

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPool2D,
    PReLU,
    Reshape,
    SpatialDropout2D,
    UpSampling2D,
)
# from keras import backend as K
# from keras.models import Model


def initial_block(x, num_filters=13):
    """Create initial block according to [1].

    Parameters
    ----------
    x : tensor
        Input to initial block module.

    num_filters : int
        NUmber of filters for first Conv2D layer, defaults to 13
        (see Fig. 2(a) [1]).

    Returns
    -------
    tensor
        Output tensor with halved row and columns and num_filters + 3 channels.

    """
    left = Conv2D(filters=num_filters,
                  kernel_size=(3, 3),
                  strides=(2, 2),
                  padding='same',
                  )(x)
    right = MaxPool2D()(x)
    return Concatenate()([left, right])


def knead(x, output_depth, name, downsampling=False):
    """Knead tensor by 1x1 convolution (no bias) followed by BN and PReLU.

    Depending on input parameter, knead can mean expansion or compression.
    Knead module corresponds to "1x1" box of right branch in Fig. 2(b)[1].

    Parameters
    ----------
    x : tensor
        Input to knead module.

    output_depth : int
        Tensor depth after compression/expansion.

    name : str
        Module name (e.g. compression10).

    downsampling : bool
        Indicates whether to apply 2x downsampling to input tensor.

    Returns
    -------
    x : tensor
        Knead module.

    """
    if downsampling is False:
        x = Conv2D(filters=output_depth,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same',
                   use_bias=False,
                   name=name+'_conv',
                   )(x)
    else:
        x = Conv2D(filters=output_depth,
                   kernel_size=(2, 2),
                   strides=(2, 2),
                   padding='same',
                   use_bias=False,
                   name=name+'_conv',
                   )(x)
    x = BatchNormalization(name=name+'_bn')(x)
    # ReLU with additional parameter per feature map (see also Keras docu)
    x = PReLU(shared_axes=[1, 2], name=name+'_prelu')(x)
    return x


def bottleneck_core(x, name, dilation=1, asymmetric=0, upsampling=False):
    """Core bottleneck convolution followed by BN and PReLU.

    It corresponds to "conv" box of right branch in Fig. 2(b)[1].

    Parameters
    ----------
    x : tensor
        Input to core module.

    name : str
        Module name (e.g. compression10).

    dilation : int
        Indicates whether to apply dilated convolution instead of normal one,
        switched off by default (i.e. equal to 1).

    asymmetric : int
        Indicates whether to apply asymmetric convolution, i.e.,
        (1 x asymmetric) filter followed by (asymmetric x 1) filter. Requires
        dilation equal to 1. Switched off by default (i.e. equal to 0).

    upsampling : bool
        Indicates whether to upsample (2x) feature map during convolution.
        Overrides asymmetric filetring and dilation settings.

    Returns
    -------
    x : tensor
        Bottleneck core module.

    """
    num_filters = x.shape[-1]
    if upsampling is True:
        x = Conv2DTranspose(filters=num_filters,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            name=name+'_deconv',
                            )(x)
    else:
        if asymmetric == 0:
            x = Conv2D(filters=num_filters,
                       kernel_size=(3, 3),
                       dilation_rate=(dilation, dilation),
                       strides=(1, 1),
                       padding='same',
                       name=name+'_conv',
                       )(x)
        else:
            assert dilation == 1
            x = Conv2D(filters=num_filters,
                       kernel_size=(1, asymmetric),
                       strides=(1, 1),
                       padding='same',
                       name=name+'_conv_hor',
                       )(x)
            x = Conv2D(filters=num_filters,
                       kernel_size=(asymmetric, 1),
                       strides=(1, 1),
                       padding='same',
                       name=name+'_conv_ver',
                       )(x)

    x = BatchNormalization(name=name+'_bn')(x)
    # ReLU with additional parameter per feature map (see also Keras docu)
    x = PReLU(shared_axes=[1, 2], name=name+'_prelu')(x)
    return x
#
#
# def bottleneck_merge(left, right):
#     """ Merge (by adding) two tensors as specified in bottleneck layer [1].
#
#     Parameters
#     ----------
#     left : tensorflow.python.framework.ops.Tensor
#         Left input tensor (residual branch). This tensor is eventually
#         processed to match dimensions of right tensor. According to [1],
#         expanding number of channels is done by zero padding, reducing number
#         of channels is done by 1x1 convolution.
#
#     right : tensorflow.python.framework.ops.Tensor
#         Right input tensor whose dimensions determine output dimensions.
#
#     Returns
#     -------
#     merge : tensorflow.python.framework.ops.Tensor
#         Merged tensor.
#
#     """
#     output_depth = K.int_shape(right)[-1]
#     if K.int_shape(left)[-1] == output_depth:
#         merge = Add()([left, right])
#
#     elif K.int_shape(left)[-1] < output_depth:
#         # Zero padding, in [1] used for downsampling only
#         left = ZeroPaddingChannelsLayer(output_depth)(left)
#         merge = Add()([left, right])
#
#     else:
#         # Squeezing, in [1] used for upsampling only
#         left = Conv2D(filters=output_depth,
#                       kernel_size=(1, 1),
#                       strides=(1, 1),
#                       padding='same',
#                       use_bias=False,
#                       )(left)
#         merge = Add()([left, right])
#     return merge
#
#
# def bottleneck(x, output_depth, name, dropout=None, downsampling=False,
#                upsampling=False, asymmetric=0, dilation=1, projection_ratio=4,
#                argmax_pooling=False, unpool_indices=None):
#     """ Create bottleneck module as described in [1], Fig. 2b.
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to bottleneck module.
#
#     output_depth : int
#         Tensor depth after passing through bottleneck.
#
#     name : str
#         Module name (e.g. bottleneck10).
#
#     dropout : float
#         Dropout rate for spatial dropout, indicates probability of *dropping*
#         a feature map.
#
#     downsampling : bool
#         Indicates whether to apply 2x downsampling to input tensor.
#
#     upsampling : bool
#         Indicates whether to upsample (2x) feature map during convolution.
#         Overrides asymmetric filetring and dilation settings.
#
#     asymmetric : int
#         Indicates whether to apply asymmetric convolution, i.e.,
#         (1 x asymmetric) filter followed by (asymmetric x 1) filter. Requires
#         dilation equal to 1. Switched off by default (i.e. equal to 0).
#
#     dilation : int
#         Indicates whether to apply dilated convolution instead of normal one,
#         switched off by default (i.e. equal to 1).
#
#     projection_ratio : int
#         Compression factor during compression phase. Input depth is reduced by
#         projection ratio after 1x1 compression convolution.
#
#     argmax_pooling : bool
#         Indicates whether to apply max-pooling with index tracking, i.e.,
#         pooling that returns max values and the corresponding indices (later to
#         be used during unpooling phase).
#
#     unpool_indices : tensorflow.python.framework.ops.Tensor
#         Unpooling indices used for upsampling. Requires argmax_pooling to be
#         switched on.
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Bottleneck core module.
#
#     indices : tensorflow.python.framework.ops.Tensor
#         Argmax pooling indices. Set to None if simple max-pooling applied.
#
#     """
#     input_depth = K.int_shape(x)[-1]
#     assert projection_ratio > 0
#     assert input_depth % projection_ratio == 0
#     assert not (downsampling and upsampling)
#     if argmax_pooling is True and upsampling is True:
#         assert unpool_indices is not None
#     reduced_depth = input_depth // projection_ratio
#
#     # Compression
#     right = knead(x, reduced_depth, name+'_compression', downsampling)
#
#     # Processing
#     right = bottleneck_core(right,
#                             name,
#                             dilation=dilation,
#                             asymmetric=asymmetric,
#                             upsampling=upsampling)
#
#     # Expansion
#     right = knead(right, output_depth, name+'_expansion')
#
#     # Regularizer
#     if dropout is not None:
#         right = SpatialDropout2D(rate=dropout)(right)
#
#     # Left branch
#     if downsampling is True:
#         if argmax_pooling is False:
#             left = MaxPooling2D()(x)
#         else:
#             # According to [1] this should be done with argmax pooling
#             left, indices = MaxPooling2DArgmax()(x)
#
#     elif upsampling is True:
#         # TODO: Match dimensions (squeeze) to match dimensions, simplify merging (just zero padding)
#
#         left = Conv2D(filters=output_depth,
#                       kernel_size=(1, 1),
#                       strides=(1, 1),
#                       padding='same',
#                       use_bias=False,
#                       )(x)
#
#         if argmax_pooling is False:
#             left = UpSampling2D()(left)
#         else:
#             # According to [1] this should be done with unpooling
#             left = MaxUnpooling2D()([left, unpool_indices])
#     else:
#         left = x
#
#     # Merge branches
#     out = bottleneck_merge(left, right)
#     out = PReLU(shared_axes=[1, 2], name=name+'_prelu_final')(out)
#
#     if argmax_pooling is True and downsampling is True:
#         return out, indices
#     else:
#         return out, None
#
#
# def stage1(x, name='', argmax_pooling=False):
#     """ Create stage 1 of ENet architecture (see [1], Table 1).
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to stage 1 module.
#
#     argmax_pooling : bool
#         Indicates whether to use argmax pooling or usual max pooling.
#
#     name : str
#         Prefix for module name (e.g. segmentation_branch).
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Stage 1 of ENet architecture.
#
#     indices : tensorflow.python.framework.ops.Tensor
#         Argmax indices from pooling. Set to None if argmax pooling is off.
#
#     """
#     bottleneck_stage1 = partial(bottleneck,
#                                 output_depth=64,
#                                 dropout=0.01,
#                                 argmax_pooling=argmax_pooling)
#     x, indices = bottleneck_stage1(x, name=name+'bottleneck1.0',
#                                    downsampling=True)
#     x, _ = bottleneck_stage1(x, name=name+'bottleneck1.1')
#     x, _ = bottleneck_stage1(x, name=name+'bottleneck1.2')
#     x, _ = bottleneck_stage1(x, name=name+'bottleneck1.3')
#     x, _ = bottleneck_stage1(x, name=name+'bottleneck1.4')
#     return x, indices
#
#
# def stage2(x, name='', argmax_pooling=False):
#     """ Create stage 2 of ENet architecture (see [1], Table 1).
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to stage 2 module.
#
#     name : str
#         Prefix for module name (e.g. segmentation_branch).
#
#     argmax_pooling : bool
#         Indicates whether to use argmax pooling or usual max pooling.
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Stage 2 of ENet architecture.
#
#     indices : tensorflow.python.framework.ops.Tensor
#         Argmax indices from pooling. Set to None if argmax pooling is off.
#
#     """
#     bottleneck_stage2 = partial(bottleneck,
#                                 output_depth=128,
#                                 dropout=0.1,
#                                 argmax_pooling=argmax_pooling)
#     x, indices = bottleneck_stage2(x, name=name+'bottleneck2.0',
#                                    downsampling=True)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.1')
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.2', dilation=2)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.3', asymmetric=5)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.4', dilation=4)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.5')
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.6', dilation=8)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.7', asymmetric=5)
#     x, _ = bottleneck_stage2(x, name=name+'bottleneck2.8', dilation=16)
#     return x, indices
#
#
# def stage3(x, name=''):
#     """ Create stage 3 of ENet architecture (see [1], Table 1).
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to stage 3 module.
#
#     name : str
#         Prefix for module name (e.g. segmentation_branch).
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Stage 3 of ENet architecture.
#
#     """
#     # Stage 3
#     bottleneck_stage3 = partial(bottleneck, output_depth=128, dropout=0.1)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.1')
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.2', dilation=2)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.3', asymmetric=5)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.4', dilation=4)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.5')
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.6', dilation=8)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.7', asymmetric=5)
#     x, _ = bottleneck_stage3(x, name=name+'bottleneck3.8', dilation=16)
#     return x
#
#
# def stage4(x, name='', unpool_indices=None):
#     """ Create stage 4 of ENet architecture (see [1], Table 1).
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to stage 4 module.
#
#     name : str
#         Prefix for module name (e.g. segmentation_branch).
#
#     unpool_indices : tensorflow.python.framework.ops.Tensor
#         Argmax indices from pooling used for upsampling by unpooling. Simple
#         upsampling is used when unpool_indices set to None.
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Stage 4 of ENet architecture.
#
#     """
#     if unpool_indices is None:
#         argmax_pooling = False
#     else:
#         argmax_pooling = True
#
#     bottleneck_stage4 = partial(bottleneck, output_depth=64, dropout=0.1)
#     x, _ = bottleneck_stage4(x,
#                              name=name+'bottleneck4.0',
#                              upsampling=True,
#                              argmax_pooling=argmax_pooling,
#                              unpool_indices=unpool_indices)
#     x, _ = bottleneck_stage4(x, name=name+'bottleneck4.1')
#     x, _ = bottleneck_stage4(x, name=name+'bottleneck4.2')
#     return x
#
#
# def stage5(x, name='', unpool_indices=None):
#     """ Create stage 4 of ENet architecture (see [1], Table 1).
#
#     Parameters
#     ----------
#     x : tensorflow.python.framework.ops.Tensor
#         Input to stage 1 module.
#
#     name : str
#         Prefix for module name (e.g. segmentation_branch).
#
#     unpool_indices : tensorflow.python.framework.ops.Tensor
#         Argmax indices from pooling used for upsampling by unpooling. Simple
#         upsampling is used when unpool_indices set to None.
#
#     Returns
#     -------
#     x : tensorflow.python.framework.ops.Tensor
#         Stage 5 of ENet architecture.
#
#     """
#     if unpool_indices is None:
#         argmax_pooling = False
#     else:
#         argmax_pooling = True
#
#     bottleneck_stage5 = partial(bottleneck, output_depth=16, dropout=0.1)
#     x, _ = bottleneck_stage5(x,
#                              name=name+'bottleneck5.0',
#                              upsampling=True,
#                              argmax_pooling=argmax_pooling,
#                              unpool_indices=unpool_indices)
#     x, _ = bottleneck_stage5(x, name=name+'bottleneck5.1')
#     return x
#
#
# class ENet():
#
#     @staticmethod
#     def get_model(input_shape, num_classes,
#                       reshape=True, argmax_pooling=False):
#         """ Create ENet architecture.
#
#         Parameters
#         ----------
#         input_shape : tuple
#             Shape of input images as (rows, cols, channels).
#
#         num_classes : int
#             Number of classes to detect *including* background, i.e., value of
#             4 will serve to detect 3 classes (+ background).
#
#         reshape : bool
#             Indicates whether to add last reshape layer for easier training (no
#             need for creating custom loss function, we can use built-in cross
#             entropy). This layer must be removed after training for correct
#             prediction, e.g.,
#                     model = Model(model.inputs, model.layers[-2].output)
#
#         argmax_pooling : bool
#             Indicates whether to use argmax pooling or usual max pooling.
#             TODO: Problem with gradients, carrently cannot be used.
#
#         Returns
#         -------
#         model : keras.models.Model
#             Keras model of ENet architecture.
#
#         """
#         input = Input(input_shape)
#         net = initial_block(input)
#
#         # Stages
#         net, indices_stage1 = stage1(net, argmax_pooling=argmax_pooling)
#         net, indices_stage2 = stage2(net, argmax_pooling=argmax_pooling)
#         net = stage3(net)
#         net = stage4(net, unpool_indices=indices_stage2)
#         net = stage5(net, unpool_indices=indices_stage1)
#
#         # Fullconv
#         net = Conv2DTranspose(filters=num_classes,
#                               kernel_size=(2, 2),
#                               strides=(2, 2),
#                               name='fullconv',
#                               padding='same',
#                               )(net)
#
#         # Softmax activation
#         net = Activation('softmax', name='softmax')(net)
#
#         # Reshape for training
#         if reshape is True:
#             net = Reshape((input_shape[0]*input_shape[1], num_classes))(net)
#
#         model = Model(input, net)
#         return model

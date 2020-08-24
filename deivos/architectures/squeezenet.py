"""Implementation of SqueezeNet as described in [1].

References
----------
[1] F.N. Iandola, S. Han, M.W. Moskewicz, K. Ashraf, W.J. Dally, K. Keutzer,
"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB
model size", arXiv, Nov. 2016.

[2] Online: https://github.com/DeepScale/SqueezeNet

"""

from tensorflow.keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Input,
    MaxPooling2D,
    Reshape,
    Softmax,
    UpSampling2D,
)
from tensorflow.keras.models import Model


def squeeze(x, name, filters):
    """Create squeeze block for SqueezeNet, see Fig. 1 [1].

    Parameters
    ----------
    x : tensor
        Input tensor.

    name : str
        Module name (e.g. squeeze10).

    filters : int
        Number of kernels for 2D convolution. This corresponds to s_{1x1}
        parameter as shown in Table 1 [1].

    Returns
    -------
    squeeze_module : tensor
        SqueezeNet squeeze block.

    """
    assert filters < x.shape[-1], \
           'Squeeze module not squeezing (' + \
           str(x.shape[-1]) + ' --> ' + str(filters) + ')'

    squeeze_module = Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name=name+'squeeze',
                            )(x)
    return squeeze_module


def expand(x, name, filters):
    """Create expand block for SqueezeNet, see Fig. 1 [1].

    Parameters
    ----------
    x : tensor
        Input tensor.

    name : str
        Module name (e.g. expand10).

    filters : int
        Number of kernels for 2D convolutions. This corresponds to e_{1x1} and
        e_{3x3} parameters and it is assumed that e_{1x1} = e_{3x3} as shown
        in Table 1 [1].

    Returns
    -------
    expand_module : tensor
        SqueezeNet expand block.

    """
    assert filters > x.shape[-1], \
           'Expand module not expanding (' + \
           str(x.shape[-1]) + ' --> ' + str(filters) + ')'

    expand_1x1 = Conv2D(filters=filters,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        activation='relu',
                        padding='same',
                        name=name+'_expand_1x1',
                        )(x)
    expand_3x3 = Conv2D(filters=filters,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding='same',
                        name=name+'_expand_3x3',
                        )(x)
    expand_module = Concatenate(name=name+'_concat')([expand_1x1, expand_3x3])
    return expand_module


def fire_module(x,
                name,
                squeeze_filters,
                bypass=False,
                squeeze_expand_ratio=4):
    """Create fire module for SqueezeNet, see Fig. 1 [1].

    Parameters
    ----------
    x : tensor
        Input tensor.

    name : str
        Module name (e.g. fire10).

    squeeze_filters : int
        Number of kernels for 2D squeeze convolutions, i.e., e_{1x1} as shown
        in Table 1 [1].

    bypass : bool
        Indicates whether to add bypass (see Sec. 6 [1]) to SqueezeNet model
        (see Fig. 1 [1]). Bypass can be simple (fire input channels is equal to
        fire output channels) or complex (fire input channels is not equal to
        fire output channels), see Sec. 6 [1].

    squeeze_expand_ratio : int
        Ratio between number of expand filters vs number of squeeze filters,
        i.e., e_{1x1} / s_{1x1} with e_{1x1} = e_{3x3}. Default value
        corresponds to original paper implementation, see Table 1 [1].

    Returns
    -------
    fire : tensor
        SqueezeNet fire module.

    """
    assert squeeze_expand_ratio > 1
    squeezed_tensor = squeeze(x, name=name, filters=squeeze_filters)
    fire = expand(squeezed_tensor,
                  name=name,
                  filters=squeeze_filters*squeeze_expand_ratio)

    if bypass is True:
        num_channels_in = x.shape[-1]
        num_channels_out = fire.shape[-1]

        # Simple bypass
        if num_channels_in == num_channels_out:
            fire = Add()([x, fire])
        # Complex bypass
        else:
            bypass = Conv2D(filters=num_channels_out,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='relu',
                            padding='same',
                            name=name+'_bypass',
                            )(x)
            fire = Add()([bypass, fire])

    return fire


# class SqueezeNet():
#
#     @staticmethod
#     def get_model(num_classes, preprocessing=None, version='1.0'):
#         """ Create SqueezeNet architecture.
#
#         Parameters
#         ----------
#         num_classes : int
#             Number of classes to detect.
#
#         preprocessing : keras.engine.training.Model
#             Input preprocessing module that adapts input shape to what
#             SqueezeNet expects. See get_model_v10 and get_model_v11 for more
#             details.
#
#         version : str, {'1.0', '1.1'}
#             Version of SqueezeNet to use. Version '1.0' corresopnds to original
#             paper implementation [1], version '1.1' is a variation thereof
#             which 2.4x smaller [2].
#
#         Returns
#         -------
#         model : keras.models.Model
#             Keras model of SqueezeNet architecture.
#
#         """
#         if version == '1.0':
#             return SqueezeNet.get_model_v10(num_classes=num_classes,
#                                             preprocessing=preprocessing)
#         elif version == '1.1':
#             return SqueezeNet.get_model_v11(num_classes=num_classes,
#                                             preprocessing=preprocessing)
#         else:
#             raise ValueError('SqueezeNet version not valid.')
#
#
#     @staticmethod
#     def get_model_v10(num_classes, preprocessing=None):
#         """ Create SqueezeNet architecture as described in [1].
#
#         This is v1.0 implementation, it corresponds to original paper
#         description with complex bypass. Check [2] for more implementation
#         information. Also, in Table 1 [1] the output size of input image is
#         incorrect, it should be 227x227 (see [2]) and not 224x224.
#
#         Parameters
#         ----------
#         preprocessing : keras.engine.training.Model
#             Input preprocessing module that adapts input shape to what
#             SqueezeNet expects. If None, SqueezeNet uses Conv2D and MaxPool as
#             described in Table 1 [1] and input shape is fixed to 227x227x3.
#             Preprocessing module must output data with size (55x55x96) which
#             is the input to the first fire module (fire2 in Table 1 [1]).
#             Example: input = Input(shape=(110, 110, 3))
#                      output = Conv2D(96, (3, 3), (2,2))(input)
#                      preprocessing = Model(input, output)
#
#         num_classes : int
#             Number of classes to detect.
#
#         Returns
#         -------
#         model : keras.models.Model
#             Keras model of SqueezeNet architecture.
#
#         """
#         if preprocessing is None:
#             input = Input(shape=(227, 227, 3))
#             net = Conv2D(filters=96,
#                          kernel_size=(7, 7),
#                          strides=(2, 2),
#                          activation='relu',
#                          padding='valid',
#                          name='conv1',
#                          )(input)
#             net = MaxPooling2D()(net)
#         else:
#             input = preprocessing.input
#             net = preprocessing.output
#             assert net.get_shape()[1:] == (55, 55, 96)
#
#
#         net = fire_module(net, name='fire2', squeeze_filters=32, bypass=True)
#         net = fire_module(net, name='fire3', squeeze_filters=32, bypass=True)
#         net = fire_module(net, name='fire4', squeeze_filters=64, bypass=True)
#         net = MaxPooling2D()(net)
#
#         net = fire_module(net, name='fire5', squeeze_filters=64, bypass=True)
#         net = fire_module(net, name='fire6', squeeze_filters=64, bypass=True)
#         net = fire_module(net, name='fire7', squeeze_filters=96, bypass=True)
#         net = fire_module(net, name='fire8', squeeze_filters=128, bypass=True)
#         net = MaxPooling2D()(net)
#
#         net = fire_module(net, name='fire9', squeeze_filters=128, bypass=True)
#         net = Conv2D(filters=num_classes,
#                      kernel_size=(1, 1),
#                      strides=(1, 1),
#                      activation='relu',
#                      padding='same',
#                      name='conv10',
#                      )(net)
#         net = AveragePooling2D(pool_size=(13, 13))(net)
#         net = Reshape((num_classes,))(net)
#         net = Softmax()(net)
#         model = Model(input, net)
#         return model
#
#
#     @staticmethod
#     def get_model_v11(preprocessing=None):
#         raise NotImplementedError()

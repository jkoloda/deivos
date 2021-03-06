"""Implementation of SqueezeNet as described in [1].

References
----------
[1] F.N. Iandola, S. Han, M.W. Moskewicz, K. Ashraf, W.J. Dally, K. Keutzer,
"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB
model size", arXiv, Nov. 2016.

[2] Online: https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.0

"""

from tensorflow.keras.layers import (
    Add,
    GlobalAveragePooling2D,
    Concatenate,
    Conv2D,
    Input,
    MaxPool2D,
    Reshape,
    Softmax,
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
                            name=name+'_squeeze',
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
                        name=name+'_expand_1x1',
                        )(x)
    # Reference uses zero padding for 3x3 filters, see Sec. 3.1.1 [1],
    # here for simplicity 'same' padding is used
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
                            name=name+'_bypass',
                            )(x)
            fire = Add()([bypass, fire])

    return fire


def default_preprocessor(x, version='1.0'):
    """Apply default preprocessing of input data according to [1].

    Parameters
    ----------
    x : tensor
        Default input tensor with dimensions (batch_size, 227, 227, 3).

    Returns
    -------
    processed : tensor
        Processed input tensor (image) with size (batch_size, 55, 55, 96),
        according to [1] (see Table 1).

    """
    if version == '1.0':
        filters = 96
        kernel_size = (7, 7)
    elif version == '1.1':
        filters = 64
        kernel_size = (3, 3)
    else:
        raise ValueError('Not valid SqueezeNet version.')

    assert x.shape[1:] == (227, 227, 3)
    processed = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=(2, 2),
                       activation='relu',
                       padding='valid',
                       name='conv1',
                       )(x)
    processed = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(processed)
    return processed


def set_bypass(bypass_type=None):
    """Set bypass flag for each fire module in SqueezeNet architecture.

    Parameters
    ----------
    bypass_type : {None, 'simple', 'complex'}
        Bypass type to be applied, see Fig. 2 [1] for more detail.

    Returns
    -------
    bypass : dict
        Dictionary of bypass values for each fire module, e.g.,
        if bypass['fire1'] is True then bypass is applied to fire module 1,
        see Fig. 2 [1] for more detail.

    """
    bypass = {'fire1': False, 'fire2': False, 'fire3': False, 'fire4': False,
              'fire5': False, 'fire6': False, 'fire7': False, 'fire8': False,
              'fire9': False}
    if bypass_type in ['simple', 'complex']:
        bypass['fire3'] = True
        bypass['fire5'] = True
        bypass['fire7'] = True
        bypass['fire9'] = True

    if bypass_type == 'complex':
        bypass['fire2'] = True
        bypass['fire4'] = True
        bypass['fire6'] = True
        bypass['fire8'] = True

    return bypass


def get_model_v10(num_classes, preprocessing=None, bypass_type=None):
    """Create SqueezeNet architecture as described in [1], version 1.0.

    This is v1.0 implementation, it corresponds to original paper
    description. Check [2] for more implementation information.

    Parameters
    ----------
    preprocessing : model
        Input preprocessing module that adapts input shape to what
        SqueezeNet expects. If None, SqueezeNet uses Conv2D and MaxPool as
        described in Table 1 [1] and input shape is fixed to 227x227x3.
        Preprocessing module must output data with size (55x55x96) which
        is input to first fire module (fire2 in Table 1 [1]).
        Example: input = Input(shape=(110, 110, 3))
                 output = Conv2D(96, (3, 3), (2,2))(input)
                 preprocessing = Model(input, output)

    num_classes : int
        Number of classes to detect.

    bypass_type : {None, 'simple', 'complex'}
        Bypass type to be applied, see Fig. 2 [1] for more detail.

    Returns
    -------
    model : model
        Keras model of SqueezeNet architecture.

    """
    bypass = set_bypass(bypass_type=bypass_type)
    if preprocessing is None:
        inputs = Input(shape=(227, 227, 3))
        net = default_preprocessor(inputs, version='1.0')
    else:
        pass
        # inputs = preprocessing.input
        # net = preprocessing.output
        # assert net.get_shape()[1:] == (55, 55, 96)
    net = fire_module(net, 'fire2', 16, bypass['fire1'])
    net = fire_module(net, 'fire3', 16, bypass['fire2'])
    net = fire_module(net, 'fire4', 32, bypass['fire3'])
    net = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(net)

    net = fire_module(net, 'fire5', 32, bypass['fire5'])
    net = fire_module(net, 'fire6', 48, bypass['fire6'])
    net = fire_module(net, 'fire7', 48, bypass['fire7'])
    net = fire_module(net, 'fire8', 64, bypass['fire8'])
    net = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(net)

    net = fire_module(net, 'fire9', 64, bypass=bypass['fire9'])
    net = Conv2D(filters=num_classes,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation='relu',
                 name='conv10',
                 )(net)
    net = GlobalAveragePooling2D()(net)
    net = Reshape((num_classes,))(net)
    net = Softmax()(net)
    model = Model(inputs, net)
    return model


def get_model_v11(num_classes, preprocessing=None, bypass_type=None):
    """Create SqueezeNet architecture version 1.1 as described in [2]."""
    bypass = set_bypass(bypass_type=bypass_type)
    if preprocessing is None:
        inputs = Input(shape=(227, 227, 3))
        net = default_preprocessor(inputs, version='1.1')
    else:
        pass
        # inputs = preprocessing.input
        # net = preprocessing.output
        # assert net.get_shape()[1:] == (55, 55, 96)
    net = fire_module(net, 'fire2', 16, bypass['fire1'])
    net = fire_module(net, 'fire3', 16, bypass['fire2'])
    # Padding is set since Caffe pooling works differently from Tensorflow
    # https://stackoverflow.com/questions/40997185/ \
    # differences-between-caffe-and-keras-when-applying-max-pooling
    net = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(net)

    net = fire_module(net, 'fire4', 32, bypass['fire3'])
    net = fire_module(net, 'fire5', 32, bypass['fire5'])
    # Padding is set since Caffe pooling works differently from Tensorflow
    net = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(net)

    net = fire_module(net, 'fire6', 48, bypass['fire6'])
    net = fire_module(net, 'fire7', 48, bypass['fire7'])
    net = fire_module(net, 'fire8', 64, bypass['fire8'])

    net = fire_module(net, 'fire9', 64, bypass=bypass['fire9'])
    net = Conv2D(filters=num_classes,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation='relu',
                 name='conv10',
                 )(net)
    net = GlobalAveragePooling2D()(net)
    net = Reshape((num_classes,))(net)
    net = Softmax()(net)
    model = Model(inputs, net)
    return model


class SqueezeNet():
    """Class to build SqueezeNet models."""

    @staticmethod
    def get_model(num_classes, preprocessing=None, version='1.0'):
        """Create SqueezeNet architecture.

        Parameters
        ----------
        num_classes : int
            Number of classes to detect.

        preprocessing : model
            Input preprocessing module that adapts input shape to what
            SqueezeNet expects. See get_model_v10 and get_model_v11 for more
            details.

        version : str, {'1.0', '1.1'}
            Version of SqueezeNet to use. Version '1.0' corresopnds to original
            paper implementation [1], version '1.1' is a variation thereof
            which is 2.4x smaller [2].

        Returns
        -------
        model : model
            Keras model of SqueezeNet architecture.

        """
        if version == '1.0':
            return get_model_v10(num_classes=num_classes,
                                 preprocessing=preprocessing)
        elif version == '1.1':
            return get_model_v11(num_classes=num_classes,
                                 preprocessing=preprocessing)
        else:
            raise ValueError('SqueezeNet version not valid.')

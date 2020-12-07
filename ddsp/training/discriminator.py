from ddsp import core
from ddsp.training import nn
import gin
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Conv1D, ZeroPadding1D
import tensorflow_addons as tfa

    
class MFCCDiscriminator(tf.keras.Sequential):
    pass


class ConvStack(tf.keras.Sequential):
    def __init__(self, padding, conv_channels, kernel_size, dilation_rate, nonlinear_activation, nonlinear_activation_params):
        super().__init__([
            ZeroPadding1D(padding),
            tfa.layers.WeightNormalization(Conv1D(conv_channels, kernel_size=kernel_size, padding='valid',
                            dilation_rate=dilation_rate)),
             getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
        ])


class ParallelWaveGANDiscriminator(tf.keras.Sequential):
    """Parallel WaveGAN Discriminator module.
    
    Could potentially be replaced bt a resnet
    Could be moved to 
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (int): Whether to use bias parameter in conv.
        """
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        conv_layers = []
        for i in range(layers - 1):
            if i == 0:
                dilation_rate = 1
            else:
                dilation_rate = i
            padding = (kernel_size - 1) // 2 * dilation_rate
            conv_layers += [ConvStack(padding, conv_channels, kernel_size, dilation_rate, nonlinear_activation, nonlinear_activation_params)]
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1D(out_channels, kernel_size=kernel_size, padding='same')
        conv_layers += [last_conv_layer]

        super().__init__(conv_layers)
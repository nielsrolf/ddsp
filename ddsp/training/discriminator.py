from ddsp import core
from ddsp.training import nn, preprocessing
import gin
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Conv1D, ZeroPadding1D
import tensorflow_addons as tfa


class Discriminator(nn.DictLayer):
  """Base class to implement disriminators.
  The discriminator can take a mix of conditional inputs (such as f0_hz) and generated audio.
  The computed score has shape (B, 1)
  """

  def __init__(self, input_keys=None, **kwargs):
    """Constructor."""
    input_keys = input_keys or self.get_argument_names('compute_score')
    super().__init__(input_keys, output_keys=['score'], **kwargs)

  def call(self, *args, **unused_kwargs):
    """Resamples all inputs to the maximal resolution and computes the score"""
    inputs  = [preprocessing.at_least_3d(i) for i in args]
    n_timesteps = max(i.shape[1] for i in inputs)
    inputs = [core.resample(i, n_timesteps) for i in inputs]
    score  = self.compute_score(*inputs)
    score = tf.reduce_mean(score, axis=list(range(1, score.ndim)))
    return score
  
  def compute_score(self, *args):
    raise NotImplementedError()

  
class MFCCDiscriminator(tf.keras.Sequential):
  pass


class ConvStack(tf.keras.Sequential):
  def __init__(self, padding, conv_channels, kernel_size, dilation_rate, nonlinear_activation, nonlinear_activation_params):
    super().__init__([
      ZeroPadding1D(padding),
      tfa.layers.WeightNormalization(
        Conv1D(conv_channels, kernel_size=kernel_size, padding='valid', dilation_rate=dilation_rate)),
      getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
    ])


class ParallelWaveGANDiscriminator(Discriminator):
  """Parallel WaveGAN Discriminator module.
  
  Could potentially be replaced bt a resnet
  Could be moved to 
  """

  def __init__(self,
               input_keys=['discriminator_audio'],
               out_channels=1,
               kernel_size=3,
               layers=10,
               conv_channels=64,
               nonlinear_activation="LeakyReLU",
               nonlinear_activation_params={"alpha": 0.2}):
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
    super().__init__(input_keys=input_keys)
    assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd."

    conv_layers = []
    for i in range(layers - 1):
      if i == 0:
        dilation_rate = 1
      else:
        dilation_rate = i
      padding = (kernel_size - 1) // 2 * dilation_rate
      conv_layers += [ConvStack(padding, conv_channels, kernel_size, dilation_rate, nonlinear_activation, nonlinear_activation_params)]
    padding = (kernel_size - 1) // 2
    last_conv_layer = Conv1D(1, kernel_size=kernel_size, padding='same')
    conv_layers += [last_conv_layer]
    self.conv_layers = conv_layers

  def compute_score(self, *inputs):
    x = tf.concat(inputs, axis=-1)
    for f in self.conv_layers:
      x = f(x)
    return x

  

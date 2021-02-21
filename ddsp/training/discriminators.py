from ddsp import core
from ddsp.training import nn, preprocessing
import gin
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Conv1D, ZeroPadding1D
import tensorflow_addons as tfa
from ddsp import spectral_ops
from ddsp.core import resample


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
    score = tf.reduce_mean(score, axis=list(range(1, len(score.shape))))
    return score
  
  def compute_score(self, *args):
    raise NotImplementedError()


@gin.register  
class MfccTimeConstantRnnDiscriminator(Discriminator):
  """Use MFCCs as latent variables, distribute across timesteps."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               z_time_steps=250,
               input_keys=['discriminator_audio', 'f0_hz', 'ld_scaled'],
               spectral_op='compute_mfcc',
               **kwargs):
    # make the input key that contains audio the first
    input_keys = sorted(input_keys, key=lambda i: not 'audio' in i)
    if len(input_keys) > 1:
        assert 'audio' not in input_keys[1], "This discriminator only handles a single audio input"
    super().__init__(**kwargs, input_keys=input_keys)
    if z_time_steps not in [63, 125, 250, 500, 1000]:
      raise ValueError(
          '`z_time_steps` currently limited to 63,125,250,500 and 1000')
    self.z_audio_spec = {
        '63': {
            'fft_size': 2048,
            'overlap': 0.5
        },
        '125': {
            'fft_size': 1024,
            'overlap': 0.5
        },
        '250': {
            'fft_size': 1024,
            'overlap': 0.75
        },
        '500': {
            'fft_size': 512,
            'overlap': 0.75
        },
        '1000': {
            'fft_size': 256,
            'overlap': 0.75
        }
    }
    self.fft_size = self.z_audio_spec[str(z_time_steps)]['fft_size']
    self.spectral_op = spectral_op
    self.overlap = self.z_audio_spec[str(z_time_steps)]['overlap']

    # Layers.
    self.z_norm = nn.Normalize('layer')
    self.rnn = nn.Rnn(rnn_channels, rnn_type)
    self.dense_out = tfkl.Dense(1)
    self.confidence = tfkl.Dense(1)

  def call(self, audio, *conditioning):
    if self.spectral_op == 'compute_mfcc':
        z = spectral_ops.compute_mfcc(
            audio,
            lo_hz=20.0,
            hi_hz=8000.0,
            fft_size=self.fft_size,
            mel_bins=128,
            mfcc_bins=30,
            overlap=self.overlap,
            pad_end=True)
    elif self.spectral_op == 'compute_logmag':
        z = spectral_ops.compute_logmag(core.tf_float32(audio), size=self.fft_size)
    
    # Normalize.
    z = self.z_norm(z[:, :, tf.newaxis, :])[:, :, 0, :]
    try:
        plt.imshow(z[0].numpy().T)
        plt.show()
        jupyter_utils.show_audio(audio[0])
    except: pass    
    n_timesteps = z.shape[1]
    conditioning = [resample(c, n_timesteps) for c  in conditioning]
    
    z = tf.concat([z] + conditioning, axis=-1)
    # Run an RNN over the latents.
    z = self.rnn(z)
    # Bounce down to compressed z dimensions.
    w = tf.math.sigmoid(self.confidence(z))
    z = self.dense_out(z)
    z = tf.reduce_sum(z * w, axis=1, keepdims=True) / tf.reduce_sum(w, axis=1, keepdims=True)
    return z


class ConvStack(tf.keras.Sequential):
  def __init__(self, padding, conv_channels, kernel_size, dilation_rate, nonlinear_activation, nonlinear_activation_params):
    super().__init__([
      ZeroPadding1D(padding),
      tfa.layers.WeightNormalization(
        Conv1D(conv_channels, kernel_size=kernel_size, padding='valid', dilation_rate=dilation_rate,
          kernel_initializer=tf.keras.initializers.HeNormal())),
      getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
    ])


@gin.register
class ParallelWaveGANDiscriminator(nn.DictLayer):
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
    super().__init__(input_keys=input_keys, output_keys=['score'])
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

  def call(self, *args, **unused_kwargs):
    """Resamples all inputs to the maximal resolution and computes the score"""
    inputs  = [preprocessing.at_least_3d(i) for i in args]
    n_timesteps = max(i.shape[1] for i in inputs)
    inputs = [core.resample(i, n_timesteps) for i in inputs]
    score  = self.compute_score(*inputs)
    score = tf.reduce_mean(score, axis=list(range(1, len(score.shape))))
    return score

  def compute_score(self, *inputs):
    x = tf.concat(inputs, axis=-1)
    print("d:", x.shape)
    for f in self.conv_layers:
      print("d:", x.shape)
      x = f(x)
    print("d:", x.shape)
    return x / 10 

  

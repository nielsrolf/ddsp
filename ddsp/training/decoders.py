# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library of encoder objects."""

from ddsp import core
from ddsp.training import nn
import gin
import tensorflow.compat.v2 as tf
from ddsp.synths import BasicUpsampler

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
class Decoder(tfkl.Layer):
    """Base class to implement any decoder.

    Users should override decode() to define the actual encoder structure.
    Hyper-parameters will be passed through the constructor.
    """

    def __init__(self,
                 output_splits=(('amps', 1), ('harmonic_distribution', 40)),
                 name=None):
        super().__init__(name=name)
        self.output_splits = output_splits
        self.n_out = sum([v[1] for v in output_splits])

    def call(self, conditioning):
        """Updates conditioning with dictionary of decoder outputs."""
        conditioning = core.copy_if_tf_function(conditioning)
        x = self.decode(conditioning)
        outputs = nn.split_to_dict(x, self.output_splits)

        if isinstance(outputs, dict):
            conditioning.update(outputs)
        else:
            raise ValueError('Decoder must output a dictionary of signals.')
        return conditioning

    def decode(self, conditioning):
        """Takes in conditioning dictionary, returns dictionary of signals."""
        raise NotImplementedError


class TimbrePaintingDecoder(Decoder):
  def __init__(self,
               name=None,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('audio', 1)),
               sample_rates = [2000, 4000, 8000, 16000],
               n_samples = 
               ):
    super().__init__(output_splits=output_splits, name=name)
    self.basic_upsampler = BasicUpsampler(sample_rate=sample_rates[0])
    self.upsamplers = []
    for sample_rate in sample_rates:
      upsampler = ParallelWaveGANUpsampler(
        input_keys=upsampler_inputs,
        output_splits=(('audio', 1)),
        name=f"upsampler{sample_rate}"
      )
      self.upsamplers += upsampler

  def decode(self, conditioning):
    """Takes in conditioning dictionary, returns dictionary of signals."""
    conditioning = dict(conditioning)
    conditioning['audio'] = self.basic_upsampler(conditioning)

    for upsampler in range(self.upsamplers):
        conditioning['audio'] = upsampler(conditioning)

    return conditioning['audio']



class ParallelWaveGANUpsampler(tfkl.Layer):
  """Single Upsampler that performs:
  (loudness, f0, z, audio_in) -> (audio_out) 

    The upsamplers get trained one after another.

    Original pytorch implementation: https://github.com/mosheman5/timbre_painting/blob/f18c709a16a2111f7a76224a5e235c8481a9c67f/models/networks.py#L22
  """

  def __init__(self,
                input_keys=('ld_scaled', 'f0_scaled', 'z', 'audio'),
                output_splits=(('signal', 1)),
                name=None,
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                layers=30,
                stacks=3,
                residual_channels=64,
                gate_channels=128,
                skip_channels=64,
                aux_channels=80,
                aux_context_window=2,
                dropout=0.0,
                use_weight_norm=True,
                use_causal_conv=False,
                # upsample_conditional_features=None,
                # upsample_net="ConvInUpsampleNetwork",
                # upsample_params={"upsample_scales": [4, 4, 4, 4]},
                # affine = False,
                norm=True):
    """Initialize Parallel WaveGAN Generator module. -> this is already a single upsampler
    Args:
      input_keys:  (tuple[string]) required keys for the conditioning dict
      output_splits: names for the splits of the outputs and the number of dimensions used for each key
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      kernel_size (int): Kernel size of dilated convolution.
      layers (int): Number of residual block layers.
      stacks (int): Number of stacks i.e., dilation cycles.
      residual_channels (int): Number of channels in residual conv.
      gate_channels (int):  Number of channels in gated conv.
      skip_channels (int): Number of channels in skip conv.
      aux_channels (int): Number of channels for auxiliary feature conv.
      aux_context_window (int): Context window size for auxiliary feature.
      dropout (float): Dropout rate. 0.0 means no dropout applied.
      use_weight_norm (bool): Whether to use weight norm.
        If set to true, it will be applied to all of the conv layers.
      use_causal_conv (bool): Whether to use causal structure.
      upsample_conditional_features (bool): Whether to use upsampling network.
      upsample_net (str): Upsampling network architecture.
      upsample_params (dict): Upsampling network parameters.
    """
    pass
    # 
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.aux_channels = aux_channels
    self.layers = layers
    self.stacks = stacks
    self.kernel_size = kernel_size

    # check the number of layers and stacks
    assert layers % stacks == 0
    layers_per_stack = layers // stacks
    
  #   # define first convolution
  #   self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

  #   # define conv + upsampling network
  #   if upsample_conditional_features:
  #     upsample_params = dict(upsample_params)
  #     upsample_params.update({
  #       "use_causal_conv": use_causal_conv,
  #     })
  #     if upsample_net == "ConvInUpsampleNetwork":
  #       upsample_params.update({
  #         "aux_channels": aux_channels,
  #         "aux_context_window": aux_context_window,
  #       })
  #     self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
  #   else:
  #     self.upsample_net = None

  #   # define residual blocks
  #   self.conv_layers = torch.nn.ModuleList()
  #   for layer in range(layers):
  #     dilation = 2**(layer % layers_per_stack)
  #     conv = ResidualBlock(
  #       kernel_size=kernel_size,
  #       residual_channels=residual_channels,
  #       gate_channels=gate_channels,
  #       skip_channels=skip_channels,
  #       aux_channels=aux_channels,
  #       dilation=dilation,
  #       dropout=dropout,
  #       bias=True,  # NOTE: magenda uses bias, but musyoku doesn't
  #       use_causal_conv=use_causal_conv,
  #     )
  #     self.conv_layers += [conv]

  #   # define output layers
  #   self.last_conv_layers = torch.nn.ModuleList([
  #     torch.nn.ReLU(inplace=True),
  #     Conv1d1x1(skip_channels, skip_channels, bias=True),
  #     torch.nn.ReLU(inplace=True),
  #     Conv1d1x1(skip_channels, out_channels, bias=True),
  #   ])

  #   if norm:
  #     self.loudness_prenet = torch.nn.Sequential(
  #       torch.nn.InstanceNorm1d(1, affine=affine),
  #       Conv1d1x1(1, aux_channels, bias=True)
  #     )
  #   else:
  #     self.loudness_prenet = Conv1d1x1(1, aux_channels, bias=True)
  
  # def forward(self, x, c=None):
  #   """Calculate forward propagation.
  #   Args:
  #     x (Tensor): Input noise signal (B, 1, T).
  #     c (Tensor): Local conditioning auxiliary features (B, C ,T').
  #   Returns:
  #     Tensor: Output tensor (B, out_channels, T)
  #   """
  #   B, _, T = x.size()

  #   # perform upsampling
  #   if c is not None and self.upsample_net is not None:
  #     c = self.loudness_prenet(c)
  #     c = self.upsample_net(c)
  #     assert c.size(-1) == x.size(-1)

  #   # encode to hidden representation
  #   x = self.first_conv(x)
  #   skips = 0
  #   for f in self.conv_layers:
  #     x, h = f(x, c)
  #     skips += h
  #   skips *= math.sqrt(1.0 / len(self.conv_layers))

  #   # apply final layers
  #   x = skips
  #   for f in self.last_conv_layers:
  #     x = f(x)

  #   return x

  @staticmethod
  def _get_receptive_field_size(layers, stacks, kernel_size,
                  dilation=lambda x: 2**x):
    assert layers % stacks == 0
    layers_per_cycle = layers // stacks
    dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
    return (kernel_size - 1) * sum(dilations) + 1

  @property
  def receptive_field_size(self):
    """Return receptive field size."""
    return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

  def decode(self, conditioning):
      """Takes in conditioning dictionary, returns dictionary of signals."""
      conditioning = dict(conditioning)
      conditioning['audio'] = self.initial_audio(conditioning)

      for i in range(self.trained_upsamplers):
          conditioning['audio'] = self.upsampler[i](conditioning)

      return conditioning['audio']


@gin.register
class RnnFcDecoder(Decoder):
    """RNN and FC stacks for f0 and loudness."""

    def __init__(self,
                 rnn_channels=512,
                 rnn_type='gru',
                 ch=512,
                 layers_per_stack=3,
                 input_keys=('ld_scaled', 'f0_scaled', 'z'),
                 output_splits=(('amps', 1), ('harmonic_distribution', 40)),
                 name=None):
        super().__init__(output_splits=output_splits, name=name)
        def stack(): return nn.FcStack(ch, layers_per_stack)
        self.input_keys = input_keys

        # Layers.
        self.input_stacks = [stack() for k in self.input_keys]
        self.rnn = nn.Rnn(rnn_channels, rnn_type)
        self.out_stack = stack()
        self.dense_out = tfkl.Dense(self.n_out)

        # Backwards compatability.
        self.f_stack = self.input_stacks[0] if len(
            self.input_stacks) >= 1 else None
        self.l_stack = self.input_stacks[1] if len(
            self.input_stacks) >= 2 else None
        self.z_stack = self.input_stacks[2] if len(
            self.input_stacks) >= 3 else None

    def decode(self, conditioning):
        # Initial processing.
        inputs = [conditioning[k] for k in self.input_keys]
        inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

        # Run an RNN over the latents.
        x = tf.concat(inputs, axis=-1)
        x = self.rnn(x)
        x = tf.concat(inputs + [x], axis=-1)

        # Final processing.
        x = self.out_stack(x)
        return self.dense_out(x)


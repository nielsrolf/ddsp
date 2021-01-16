# Copyright 2021 The DDSP Authors.
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
"""Library of decoder layers."""

from ddsp import core
from ddsp.training import nn
from ddsp import core
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
              input_keys=('amplitudes', 'f0_hz', 'z'),
              sample_rates=[2000, 4000, 8000, 16000],
              n_total=64000):
    super().__init__(output_splits=(('audio_tensor', 1),), name=name)
    n_initial = int(sample_rates[0]/sample_rates[-1]*n_total)
    self.basic_upsampler = BasicUpsampler(n_samples=n_initial,sample_rate=sample_rates[0])
    self.upsamplers = []
    for sample_rate in sample_rates:
      upsampler = Upsampler(
        int(sample_rate/sample_rates[-1]*n_total),
        input_keys=input_keys+('audio_tensor',),
        name=f"upsampler{sample_rate}"
      )
      self.upsamplers += [upsampler]

  def decode(self, conditioning):
    """Takes in conditioning dictionary, returns dictionary of signals."""
    conditioning = dict(conditioning)
    audio = self.basic_upsampler(conditioning['amplitudes'], conditioning['f0_hz'])
    conditioning['audio_tensor'] = tf.expand_dims(audio, 2)

    for upsampler in self.upsamplers:
      conditioning['audio_tensor'] = upsampler.decode(conditioning)

    return conditioning['audio_tensor']


class Upsampler(Decoder):
  """Featurewise FcStack -> Resample -> DilatedConvs -> Dense"""

  def __init__(self,
         n_timesteps,
         conv_layers=10,
         kernel=3,
         ch=512,
         layers_per_input_stack=3,
         input_keys=('ld_scaled', 'f0_scaled', 'z', 'audio'),
         name=None):
    super().__init__(output_splits=(('audio', 1),), name=name)
    assert 'audio_tensor' in input_keys, f"Upsampler requires one input to be named audio. Got input_keys: {str(input_keys)}"

    def fc_stack():
      return nn.FcStack(ch, layers_per_input_stack)

    def conv_stack(dilation_rate):
      return nn.DilatedConvLayer(ch, kernel, dilation_rate)

    self.n_timesteps = n_timesteps
    self.input_keys = [k for k in input_keys if k != 'audio_tensor']

    # Layers.
    self.input_stacks = [fc_stack() for k in input_keys if k != 'audio_tensor']
    self.conv_layers = [conv_stack(2**i) for i in range(1, conv_layers + 1)]
    self.dense_out = tfkl.Dense(1)

  def decode(self, conditioning):
    # Initial processing.
    audio = conditioning.pop("audio_tensor")
    audio = core.resample(audio, self.n_timesteps)

    inputs = [conditioning[k] for k in self.input_keys]
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

    # Resample all inputs to the target sample rate
    inputs = [core.resample(x, self.n_timesteps) for x in inputs]
    # Conv layers
    # TODO copy original implementation or do a correct resnet
    x = tf.concat(inputs, axis=-1)
    for conv_layer in self.conv_layers:
      x = conv_layer(x)
    # Final processing.
    return self.dense_out(x) + audio



# @gin.register
class RnnFcDecoder(nn.OutputSplitsLayer):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    super().__init__(
        input_keys=input_keys, output_splits=output_splits, **kwargs)
    stack = lambda: nn.FcStack(ch, layers_per_stack)

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

  def compute_output(self, *inputs):
    # Initial processing.
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]
    print("RnnFcDecoder:")
    print({k: x.shape for k, x in zip(self.input_keys, inputs)})

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    return self.out_stack(x)


@gin.register
class MidiDecoder(nn.DictLayer):
  """Decodes MIDI notes (& velocities) to f0 (& loudness)."""

  def __init__(self,
               net=None,
               f0_residual=True,
               center_loudness=True,
               norm=True,
               **kwargs):
    """Constructor."""
    super().__init__(**kwargs)
    self.net = net
    self.f0_residual = f0_residual
    self.center_loudness = center_loudness
    self.dense_out = tfkl.Dense(2)
    self.norm = nn.Normalize('layer') if norm else None

  def call(self, z_pitch, z_vel, z=None) -> ['f0_midi', 'loudness']:
    """Forward pass for the MIDI decoder.

    Args:
      z_pitch: Tensor containing encoded pitch in MIDI scale. [batch, time, 1].
      z_vel: Tensor containing encoded velocity in MIDI scale. [batch, time, 1].
      z: Additional non-MIDI latent tensor. [batch, time, n_z]

    Returns:
      f0_midi, loudness: Reconstructed f0 and loudness.
    """
    # pylint: disable=unused-argument
    # x = tf.concat([z_pitch, z_vel], axis=-1)  # TODO(jesse): Allow velocity.
    x = z_pitch
    x = self.net(x) if z is None else self.net([x, z])

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    f0_midi = x[..., 0:1]
    loudness = x[..., 1:2]

    if self.f0_residual:
      f0_midi += z_pitch

    if self.center_loudness:
      loudness = loudness * 30.0 - 70.0

    return f0_midi, loudness


@gin.register
class MidiToHarmonicDecoder(nn.DictLayer):
  """Decodes MIDI notes (& velocities) to f0, amps, hd, noise."""

  def __init__(self,
               net=None,
               f0_residual=True,
               norm=True,
               output_splits=(('f0_midi', 1),
                              ('amplitudes', 1),
                              ('harmonic_distribution', 60),
                              ('magnitudes', 65)),
               **kwargs):
    """Constructor."""
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    output_keys = [v[0] for v in output_splits] + ['f0_hz']
    super().__init__(output_keys=output_keys, **kwargs)

    # Layers.
    self.net = net
    self.f0_residual = f0_residual
    self.dense_out = tfkl.Dense(self.n_out)
    self.norm = nn.Normalize('layer') if norm else None

  def call(self, z_pitch, z_vel, z=None):
    """Forward pass for the MIDI decoder.

    Args:
      z_pitch: Tensor containing encoded pitch in MIDI scale. [batch, time, 1].
      z_vel: Tensor containing encoded velocity in MIDI scale. [batch, time, 1].
      z: Additional non-MIDI latent tensor. [batch, time, n_z]

    Returns:
      A dictionary to feed into a processor group.
    """
    # pylint: disable=unused-argument
    # x = tf.concat([z_pitch, z_vel], axis=-1)  # TODO(jesse): Allow velocity.
    x = z_pitch
    x = self.net(x) if z is None else self.net([x, z])

    if self.norm is not None:
      x = self.norm(x)

    x = self.dense_out(x)

    outputs = nn.split_to_dict(x, self.output_splits)

    if self.f0_residual:
      outputs['f0_midi'] += z_pitch

    outputs['f0_hz'] = core.midi_to_hz(outputs['f0_midi'])
    return outputs


@gin.register
class DilatedConvDecoder(nn.OutputSplitsLayer):
  """WaveNet style 1-D dilated convolution with optional conditioning."""

  def __init__(self,
               ch=256,
               layers_per_stack=5,
               stacks=2,
               dilation=2,
               norm_type='layer',
               resample=False,
               resample_stride=None,
               stacks_per_resample=None,
               input_keys=('ld_scaled', 'f0_scaled'),
               output_splits=(('amps', 1), ('harmonic_distribution', 60)),
               conditioning_keys=('z'),
               precondition_stack=None,
               **kwargs):
    """Constructor, combines input_keys and conditioning_keys."""
    self.conditioning_keys = ([] if conditioning_keys is None else
                              list(conditioning_keys))
    input_keys = list(input_keys) + self.conditioning_keys
    super().__init__(input_keys, output_splits, **kwargs)

    # Conditioning.
    self.n_conditioning = len(self.conditioning_keys)
    self.conditional = bool(self.conditioning_keys)
    if not self.conditional and precondition_stack is not None:
      raise ValueError('You must specify conditioning keys if you specify'
                       'a precondition stack.')

    # Layers.
    self.precondition_stack = precondition_stack
    self.dilated_conv_stack = nn.DilatedConvStack(
        ch=ch,
        layers_per_stack=layers_per_stack,
        stacks=stacks,
        dilation=dilation,
        norm_type=norm_type,
        resample_type='upsample' if resample else None,
        resample_stride=resample_stride,
        stacks_per_resample=stacks_per_resample,
        conditional=self.conditional)

  def _parse_inputs(self, inputs):
    """Split x and z inputs and run preconditioning."""
    if self.conditional:
      x = tf.concat(inputs[:-self.n_conditioning], axis=-1)
      z = tf.concat(inputs[-self.n_conditioning:], axis=-1)
      if self.precondition_stack is not None:
        z = self.precondition_stack(z)
      return [x, z]
    else:
      return tf.concat(inputs, axis=-1)

  def compute_output(self, *inputs):
    stack_inputs = self._parse_inputs(inputs)
    return self.dilated_conv_stack(stack_inputs)

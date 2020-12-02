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
              input_keys=('amplitudes', 'f0_hz', 'z'),
              sample_rates=[2000, 4000, 8000, 16000],
              n_total=64000
              ):
    super().__init__(output_splits=(('audio', 1),), name=name)
    n_initial = int(sample_rates[0]/sample_rates[-1]*n_total)
    self.basic_upsampler = BasicUpsampler(n_samples=n_initial,sample_rate=sample_rates[0])
    self.upsamplers = []
    for sample_rate in sample_rates:
      upsampler = Upsampler(
        int(sample_rate/sample_rates[-1]*n_total),
        input_keys=input_keys+('audio',),
        name=f"upsampler{sample_rate}"
      )
      self.upsamplers += [upsampler]

  def decode(self, conditioning):
    """Takes in conditioning dictionary, returns dictionary of signals."""
    conditioning = dict(conditioning)
    audio = self.basic_upsampler(conditioning['amplitudes'], conditioning['f0_hz'])
    conditioning['audio'] = tf.expand_dims(audio, 2)

    for upsampler in self.upsamplers:
      conditioning['audio'] = upsampler.decode(conditioning)

    return conditioning['audio']


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
    assert 'audio' in input_keys, f"Upsampler requires one input to be named audio. Got input_keys: {str(input_keys)}"

    def fc_stack():
      return nn.FcStack(ch, layers_per_input_stack)

    def conv_stack(dilation_rate):
      return nn.DilatedConvLayer(ch, kernel, dilation_rate)

    self.n_timesteps = n_timesteps
    self.input_keys = [k for k in input_keys if k != 'audio']

    # Layers.
    self.input_stacks = [fc_stack() for k in input_keys if k != 'audio']
    self.conv_layers = [conv_stack(2**i) for i in range(1, conv_layers + 1)]
    self.dense_out = tfkl.Dense(1)

  def decode(self, conditioning):
    # Initial processing.
    audio = conditioning.pop("audio")
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


  #@gin.register
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
    print("RnnFcDecoder:")
    print({k: x.shape for k, x in zip(self.input_keys, inputs)})

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    x = self.out_stack(x)
    return self.dense_out(x)


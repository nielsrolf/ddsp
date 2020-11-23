# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Model that outputs coefficeints of an additive synthesizer."""

import ddsp
from ddsp.training.models.model import Model
import tensorflow as tf

class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)
    discriminators = [loss.__dict__.get('discriminator') for loss in losses]
    discriminators = [d for d in discriminators if d is not None]
    assert len(discriminators) <= 1, "There should only be one adversarial loss"
    self._discriminator = discriminators[0] if len(discriminators) == 1 else None

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    if self.preprocessor is not None:
      conditioning = self.preprocessor(features, training=training)
    else:
      conditioning = features
    if self.encoder is not None:
      z_dict = self.encoder(conditioning)
      conditioning.update(z_dict)
    return conditioning

  def decode(self, conditioning, training=True):
    """Get generated audio by decoding than processing."""
    processor_inputs = self.decoder(conditioning, training=training)
    return self.processor_group(processor_inputs)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return self.processor_group.get_signal(outputs)

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    conditioning = self.encode(features, training=training)
    processor_inputs = self.decoder(conditioning, training=training)
    outputs = self.processor_group.get_controls(processor_inputs)
    outputs['audio_synth'] = self.processor_group.get_signal(outputs)
    if training:
      self._update_losses_dict(
          self.loss_objs, features['audio'], outputs['audio_synth'])
    return outputs

  @tf.function
  def step_fn(self, batch):
    """Per-Replica training step."""
    with tf.GradientTape() as tape:
      outputs, losses = self(batch, return_losses=True, training=True)
    # Clip and apply gradients.
    signal = self.get_audio_from_outputs(outputs)
    grads = tape.gradient(losses['total_loss'], self.generator_variables)
    return signal, losses, grads




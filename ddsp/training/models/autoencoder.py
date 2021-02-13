# Copyright 2021 The DDSP Authors.
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
"""Model that encodes audio features and decodes with a ddsp processor group."""

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
               discriminator=None,
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)
    self._discriminator = discriminator

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    if self.preprocessor is not None:
      features.update(self.preprocessor(features, training=training))
    if self.encoder is not None:
      features.update(self.encoder(features))
    return features

  def decode(self, features, training=True):
    """Get generated audio by decoding than processing."""
    features.update(self.decoder(features, training=training))
    return self.processor_group(features)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return outputs['audio_synth']
  
  def _add_discriminator_loss(self, outputs):
    if not self.is_gan:
      return
    batch = {k if k != 'audio_synth' else 'discriminator_audio': v for k, v in outputs.items()}
    scores = self.discriminator(batch)['score']
    self._losses_dict['adversarial_loss'] = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(scores), scores, 'L2'))

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    features = self.encode(features, training=training)
    features.update(self.decoder(features, training=training))

    # Run through processor group.
    pg_out = self.processor_group(features, return_outputs_dict=True)

    # Parse outputs
    outputs = pg_out['controls']
    outputs['audio_synth'] = pg_out['signal']

    if training:
      self._update_losses_dict(
          self.loss_objs, features['audio'], outputs['audio_synth'])
      self._add_discriminator_loss(outputs)

    return outputs

  @tf.function
  def step_fn(self, batch):
    """Per-Replica training step."""
    with tf.GradientTape() as tape:
      outputs, losses = self(batch, return_losses=True, training=True)
    grads = tape.gradient(losses['total_loss'], self.generator_variables)
    return outputs, losses, grads

  # @tf.function
  # def discriminator_step_fn(self, batch):
  #   """At this point, the batch already contains the generator output.
  #   The samples in batch['audio'] and batch['audio_synth'] correspond to each other.
  #   In order to prevent overfitting on a pattern that is realistic by itself but 
  #   different from the original sample, it is randomly sampled wether to use the 
  #   original or synthesized version of a sample.
  #   """
  #   outputs = {}
  #   use_real_sample = tf.round(tf.random.uniform([batch['audio'].shape[0], 1], maxval=1))
  #   batch = dict(**batch)
  #   batch['discriminator_audio'] = use_real_sample * batch['audio'] + (1 - use_real_sample) * batch['audio_synth']
  #   use_real_sample = tf.squeeze(use_real_sample)
  #   with tf.GradientTape() as tape:
  #     scores = self.discriminator(batch)['score']
  #     outputs['discriminator_loss'] = mean_difference(use_real_sample, scores, 'L2')
  #   mean_pred_real = tf.reduce_sum(scores * use_real_sample) / tf.reduce_sum(use_real_sample)
  #   mean_pred_synth = tf.reduce_sum(scores * (1 - use_real_sample)) / tf.reduce_sum(1 - use_real_sample)
  #   grads = tape.gradient(outputs['discriminator_loss'], self.discriminator_variables)
  #   losses = {
  #     'discriminator_loss': outputs['discriminator_loss'],
  #     'mean_pred_real': mean_pred_real,
  #     'mean_pred_synth': mean_pred_synth}

  #   return losses, grads
  def discriminator_step_fn(self, batch):
        """At this point, the batch already contains the generator output.
        The samples in batch['audio'] and batch['audio_synth'] correspond to each other.
        In order to prevent overfitting on a pattern that is realistic by itself but 
        different from the original sample, it is randomly sampled wether to use the 
        original or synthesized version of a sample.
        """
        losses = {}
        
        real_batch = dict(**batch)
        real_batch['discriminator_audio'] = real_batch['audio']
        fake_batch = dict(**batch)
        fake_batch['discriminator_audio'] = fake_batch['audio_synth']
        
        with tf.GradientTape() as tape:
            scores_real = self.discriminator(real_batch)['score']
            scores_fake = self.discriminator(fake_batch)['score']
            d = tf.reduce_mean(tf.abs(real_batch['discriminator_audio']) -  fake_batch['discriminator_audio'])
            losses['discriminator_loss_total'] = tf.reduce_mean(scores_real) - tf.reduce_mean(scores_fake)
            # losses['discriminator_loss_real'] = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(scores_real), scores_real))
            # losses['discriminator_loss_fake'] = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.zeros_like(scores_fake), scores_fake))
            # losses['discriminator_loss_total'] = losses['discriminator_loss_real'] + losses['discriminator_loss_fake']
            losses['discriminator_pred_real'] = tf.reduce_mean(scores_real)
            losses['discriminator_pred_fake'] = tf.reduce_mean(scores_fake)
            losses['difference_input'] = d
            grads = tape.gradient(losses['discriminator_loss_total'], self.discriminator_variables)
        
        return losses, grads





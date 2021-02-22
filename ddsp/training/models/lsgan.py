import ddsp
from ddsp.training.models.model import Model
import tensorflow as tf


def mse(a, b):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(a, b))


class LSGAN(Model):
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
    self._losses_dict['adversarial_loss'] = mse(tf.ones_like(scores), scores)

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

  @tf.function
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
      losses['d_loss_real'] = mse(tf.ones_like(scores_real), scores_real)
      losses['d_loss_fake'] = mse(tf.zeros_like(scores_fake), scores_fake)
      losses['d_loss_total'] = losses['d_loss_real'] + losses['d_loss_fake']
      grads = tape.gradient(losses['d_loss_total'], self.discriminator_variables)
    
    return losses, grads

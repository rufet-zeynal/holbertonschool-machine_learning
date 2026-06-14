#!/usr/bin/env python3
"""Wasserstein GAN with weight clipping"""
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """A Wasserstein GAN whose critic is kept 1-Lipschitz by clipping
    its weights into [-1, 1] after every update"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # generator loss : -E[ D(G(z)) ]
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # discriminator (critic) loss : E[ D(fake) ] - E[ D(real) ]
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(x) - tf.reduce_mean(y))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Returns generator(latent sample) of given size"""
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Returns a random batch from the real examples"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """disc_iter critic updates (clipped), then one generator
        update"""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                disc_real = self.discriminator(real_sample, training=True)
                disc_fake = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(disc_fake, disc_real)

            disc_grads = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_variables))

            # enforce the Lipschitz constraint by clipping the weights
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            disc_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(disc_fake)

        gen_grads = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

#!/usr/bin/env python3
"""
Defines a function that builds a variational autoencoder model.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.
    """
    # 1. ENCODER WITH SAMPLING
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean), mean=0., stddev=1.
        )
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,),
        name='z'
    )([mu, log_sig])

    encoder = keras.Model(encoder_inputs, [z, mu, log_sig], name='encoder')

    # 2. DECODER
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_inputs, outputs, name='decoder')

    # 3. FULL VARIATIONAL AUTOENCODER
    auto_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(encoder_inputs, auto_outputs, name='vae')

    # 4. LOSS CALCULATION VIA KERAS BACKEND
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        auto_outputs
    )
    reconstruction_loss *= input_dims

    # Adjusted indentation to cleanly align mathematical operations
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_sig - keras.backend.square(mu) -
        keras.backend.exp(log_sig),
        axis=-1
    )
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto

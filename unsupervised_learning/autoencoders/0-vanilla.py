#!/usr/bin/env python3
"""
Defines a function that builds a vanilla autoencoder model.
"""
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model.
    """
    # 1. ENCODER
    encoder_inputs = tf.keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)
    latent_space = tf.keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = tf.keras.Model(encoder_inputs, latent_space, name='encoder')

    # 2. DECODER
    decoder_inputs = tf.keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = tf.keras.Model(decoder_inputs, outputs, name='decoder')

    # 3. FULL AUTOENCODER
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = tf.keras.Model(encoder_inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

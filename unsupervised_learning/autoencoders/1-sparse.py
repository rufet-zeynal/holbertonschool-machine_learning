#!/usr/bin/env python3
"""
Defines a function that builds a sparse autoencoder model.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder model using L1 activity regularization.
    """
    # 1. ENCODER
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    reg = keras.regularizers.l1(lambtha)
    latent_space = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=reg
    )(x)

    encoder = keras.Model(encoder_inputs, latent_space, name='encoder')

    # 2. DECODER
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_inputs, outputs, name='decoder')

    # 3. FULL AUTOENCODER
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

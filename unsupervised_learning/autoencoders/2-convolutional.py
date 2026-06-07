#!/usr/bin/env python3
"""
Defines a function that builds a convolutional autoencoder model.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.
    """
    # 1. ENCODER
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(encoder_inputs, x, name='encoder')

    # 2. DECODER
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs

    # Reverse the list of filters to build backward path cleanly
    rev_filters = filters[::-1]

    # Process all layers except for the last two custom ones
    for i in range(len(filters) - 1):
        x = keras.layers.Conv2D(
            rev_filters[i], (3, 3), activation='relu', padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution layer uses 'valid' padding
    x = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu', padding='valid'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Last convolution layer maps exactly to input image channels
    channels = input_dims[-1]
    outputs = keras.layers.Conv2D(
        channels, (3, 3), activation='sigmoid', padding='same'
    )(x)

    decoder = keras.Model(decoder_inputs, outputs, name='decoder')

    # 3. FULL AUTOENCODER
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

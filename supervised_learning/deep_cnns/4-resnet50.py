#!/usr/bin/env python3
"""ResNet-50 architecture"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Input shape: (224, 224, 3)
    Returns: keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv1 — 7x7, 64 filters, stride 2, then MaxPool
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2: Conv2_x — 1 projection block + 2 identity blocks
    # Output channels go: 64 → 64 → 256
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3: Conv3_x — 1 projection block + 3 identity blocks
    # Output channels go: 128 → 128 → 512, stride 2 to halve spatial dims
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4: Conv4_x — 1 projection block + 5 identity blocks
    # Output channels go: 256 → 256 → 1024, stride 2
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5: Conv5_x — 1 projection block + 2 identity blocks
    # Output channels go: 512 → 512 → 2048, stride 2
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling + fully connected (Dense 1000 for ImageNet)
    X = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(X)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(X)

    model = K.models.Model(inputs=X_input, outputs=X)
    return model

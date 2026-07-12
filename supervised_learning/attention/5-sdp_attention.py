#!/usr/bin/env python3
"""Scaled Dot Product Attention module"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale by the square root of the key dimension
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights

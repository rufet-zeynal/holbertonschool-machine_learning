#!/usr/bin/env python3
"""Builds the encoder/decoder masks """
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    """

    def padding_mask(seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    encoder_mask = padding_mask(inputs)
    decoder_mask = padding_mask(inputs)


    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    target_padding_mask = padding_mask(target)
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask

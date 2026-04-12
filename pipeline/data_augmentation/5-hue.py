#!/usr/bin/env python3
"""
Data Augmentation - Hue Changing
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Hue Changing of an image
    """
    return tf.image.random_hue(image, delta)

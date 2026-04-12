#!/usr/bin/env python3
"""
Data Augmentation - Hue Changingg
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Hue Changing of an image
    """
    return tf.image.adjust_hue(image, delta)

#!/usr/bin/env python3
"""
Data Augmentation - Contrasting
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Contrast of an image
    """
    contrasted_image = tf.image.random_contrast(image,
                        lower, upper)
    return contrasted_image

#!/usr/bin/env python3
"""
Data Augmentation - Crop Image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.
    """
    image = tf.image.random_crop(image, size)
    return image

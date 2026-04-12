#!/usr/bin/env python3
"""
Data Augmentation - Flipping horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Flipping image
    """
    image = tf.image.flip_left_right(image)
    return image

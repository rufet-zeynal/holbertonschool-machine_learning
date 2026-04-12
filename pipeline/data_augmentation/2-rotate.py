#!/usr/bin/env python3
"""
Data Augmentation - Rotation
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotate an image
    """
    rotated_image = tf.image.rot90(image)
    return rotated_image

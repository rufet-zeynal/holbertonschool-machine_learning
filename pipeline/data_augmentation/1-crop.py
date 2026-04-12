#!/usr/bin/env python3
"""
Data Augmentation - Cropping
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Cropping function
    """
    return tf.random_crop(image, size)

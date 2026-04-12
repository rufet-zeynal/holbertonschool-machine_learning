#!/usr/bin/env python3
"""
Data Augmentation - Brightness
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Brightness augmentation
    """
    brightened_image = tf.image.random_brightness(image, max_delta)
    return brightened_image

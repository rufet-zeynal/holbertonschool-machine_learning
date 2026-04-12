#!/usr/bin/env python3
"""
Data Augmentation - Hue Changing
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Hue Changing of an image
    """
    hue_adjusted_image = tf.image.adjust_hue(image,
                        delta=0.1)
    return hue_adjusted_image

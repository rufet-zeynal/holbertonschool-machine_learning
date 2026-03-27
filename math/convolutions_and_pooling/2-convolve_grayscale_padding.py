#!/usr/bin/env python3
"""Convolution with custom padding on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs convolution with custom padding

    images  - (m, h, w)
    kernel  - (kh, kw)
    padding - (ph, pw) how many zeros to add on each side

    Returns: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # output size with custom padding
    out_h = h + 2*ph - kh + 1
    out_w = w + 2*pw - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

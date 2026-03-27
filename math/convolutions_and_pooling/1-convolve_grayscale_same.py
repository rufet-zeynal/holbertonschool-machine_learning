#!/usr/bin/env python3
"""Same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs same convolution — output same size as input

    images  - (m, h, w)
    kernel  - (kh, kw)

    Returns: convolved images same size as input
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # padding needed to keep output same size as input
    # for odd kernels: pad = (kernel-1)/2
    # for even kernels: pad more on bottom/right
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # pad images with zeros
    # (0,0) = no padding on m dimension
    # (ph, ph) = pad top and bottom
    # (pw, pw) = pad left and right
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # output same size as input
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

#!/usr/bin/env python3
"""Strided convolution with all padding options"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution with stride and any padding type

    images  - (m, h, w)
    kernel  - (kh, kw)
    padding - 'same', 'valid', or (ph, pw)
    stride  - (sh, sw)

    Returns: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # calculate padding based on type
    if padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # output size with stride
    # floor division because stride can skip positions
    out_h = (h + 2*ph - kh) // sh + 1
    out_w = (w + 2*pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    # loop over output positions
    # i and j are output indices
    # multiply by stride to get input position
    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

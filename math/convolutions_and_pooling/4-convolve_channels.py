#!/usr/bin/env python3
"""Convolution on images with color channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution on images with channels

    images  - (m, h, w, c)   c = number of channels
    kernel  - (kh, kw, c)    same number of channels as image
    padding - 'same', 'valid', or (ph, pw)
    stride  - (sh, sw)

    Returns: convolved images (m, out_h, out_w)
    """
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # pad — add zeros to h and w dimensions only, not channels
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    out_h = (h + 2*ph - kh) // sh + 1
    out_w = (w + 2*pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # region shape: (m, kh, kw, c)
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            # sum across height, width AND channels (1,2,3)
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output

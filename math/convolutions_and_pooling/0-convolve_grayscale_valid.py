#!/usr/bin/env python3
"""Valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs valid convolution — no padding, output is smaller

    images  - (m, h, w)    m grayscale images
    kernel  - (kh, kw)     convolution kernel

    Returns: convolved images
    """
    m, h, w    = images.shape
    kh, kw     = kernel.shape

    # output size: valid means no padding
    # each dimension shrinks by (kernel_size - 1)
    out_h = h - kh + 1
    out_w = w - kw + 1

    # empty output array
    output = np.zeros((m, out_h, out_w))

    # slide kernel across image — only 2 loops allowed
    for i in range(out_h):
        for j in range(out_w):
            # extract region from ALL images at once (no loop over m)
            region = images[:, i:i+kh, j:j+kw]

            # multiply region by kernel and sum
            # region shape: (m, kh, kw)
            # kernel shape: (kh, kw)
            # result shape: (m,)
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

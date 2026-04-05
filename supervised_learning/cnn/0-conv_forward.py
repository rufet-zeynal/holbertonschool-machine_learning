#!/usr/bin/env python3
"""Forward propagation over a convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Forward propagation over a convolutional layer

    A_prev     - (m, h_prev, w_prev, c_prev) input
    W          - (kh, kw, c_prev, c_new) kernels
    b          - (1, 1, 1, c_new) biases
    activation - activation function
    padding    - 'same' or 'valid'
    stride     - (sh, sw)

    Returns: activated output of convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new          = W.shape
    sh, sw                    = stride

    # calculate padding
    if padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    else:
        ph, pw = 0, 0

    # pad input
    A_padded = np.pad(A_prev,
                      ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')

    # output dimensions
    out_h = (h_prev + 2*ph - kh) // sh + 1
    out_w = (w_prev + 2*pw - kw) // sw + 1

    # output array
    Z = np.zeros((m, out_h, out_w, c_new))

    # convolve — loop over output height and width
    for i in range(out_h):
        for j in range(out_w):
            # extract region from all images at once
            region = A_padded[:,
                               i*sh : i*sh+kh,
                               j*sw : j*sw+kw,
                               :]
            # region shape: (m, kh, kw, c_prev)
            # W shape:      (kh, kw, c_prev, c_new)
            # for each kernel k: sum(region * W[:,:,:,k])
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    region * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    # add bias and apply activation
    Z += b
    return activation(Z)

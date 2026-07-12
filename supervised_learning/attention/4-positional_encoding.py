#!/usr/bin/env python3
"""Positional Encoding module"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer"""
    PE = np.zeros((max_seq_len, dm))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    # angle rates depend on dimension index i
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    # sin on even indices, cos on odd indices
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE

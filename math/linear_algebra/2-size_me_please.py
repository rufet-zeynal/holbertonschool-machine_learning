#!/usr/bin/env python3
"""
Shape
"""


def matrix_shape(matrix):
    """
    Matrix shaping
    """
    matshape = []
    while isinstance(matrix, list):
        matshape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return matshape

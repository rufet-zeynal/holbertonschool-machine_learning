#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Matrix shaping
    """
    matrix_shape = []
    while isinstance(matrix, list):
        matrix_shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]

#!/usr/bin/env python3
"""
Transpose matrix
"""


def matrix_transpose(matrix):
    """
    Finding the transpose of the matrix
    """
    if not matrix:
        return []
    if not all(len(row) == len(matrix[0]) for row in matrix):
        return None
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

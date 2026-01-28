#!/usr/bin/env python3
"""
Determinant of a matrix
"""


def determinant(mat):
    """
    Calculation of determinant of a matrix
    """
    if (not isinstance(mat, list) or
            any(not isinstance(row, list) for row in mat)):
        raise TypeError("matrix must be a list of lists")
    if mat == [[]]:
        return 1
    x = len(mat)
    if any(len(row) != x for row in mat):
        raise ValueError("matrix must be a square matrix")
    if x == 1:
        return mat[0][0]
    if x == 2:
        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]

    return sum(
        (-1) ** k * mat[0][k] *
        determinant([row[:k] + row[k + 1:] for row in mat[1:]])
        for k in range(x)
    )

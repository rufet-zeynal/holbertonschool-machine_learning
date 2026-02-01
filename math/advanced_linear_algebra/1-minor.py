#!/usr/bin/env python3
"""
Minor of a matrix
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


def minor(matrix):
    """
    Defining the minor of a matrix
    """
    if (not(isinstance(matrix, list)) or len(matrix) == 0 or
           any(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    x = len(matrix)
    if any(len(row) != x for row in matrix) or len(matrix) == 0:
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1

    minor_matrix = []
    for i in range(x):
        minor_row = []
        for j in range(x):
            mat = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(mat))
        minor_matrix.append(minor_row)
    return minor_matrix

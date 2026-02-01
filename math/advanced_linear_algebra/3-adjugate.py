#!/usr/bin/env python3
"""
Adjugate
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

def adjugate(matrix):
    """
    Adjugate matrix
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    x = len(matrix)
    if x == 0 or any(len(row) != x for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if x == 1:
        return [[1]]

    adjugate_matrix = [[0 for i in range(x)] for j in range(x)]
    for i in range(x):
        for j in range(x):
            sub_mat = [row[:j] + row[j + 1:] for row in
                       (matrix[:i] + matrix[i + 1:])]

            sign = (-1) ** (i + j)
            cofactor_value =sign * determinant(sub_mat)
            adjugate_matrix[j][i] = cofactor_value
    return adjugate_matrix

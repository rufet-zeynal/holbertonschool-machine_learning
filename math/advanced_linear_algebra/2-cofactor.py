#!/usr/bin/env python3
"""
Cofactor
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


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    x = len(matrix)
    if x == 0 or any(len(row) != x for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if x == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(x):
        cofactor_row = []
        for j in range(x):
            # Minor calculation: submatrix by removing row i and column j
            sub_mat = [row[:j] + row[j + 1:] for row in
                       (matrix[:i] + matrix[i + 1:])]

            # Apply the sign (-1)^(i+j) to the determinant of the submatrix
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * determinant(sub_mat))

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix

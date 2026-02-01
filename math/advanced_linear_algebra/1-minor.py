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

    # Base case for empty matrix in recursion
    if mat == [[]]:
        return 1

    x = len(mat)
    if any(len(row) != x for row in mat):
        raise ValueError("matrix must be a square matrix")

    # Base cases for determinant calculation
    if x == 1:
        return mat[0][0]
    if x == 2:
        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]

    # Recursive Laplace expansion
    return sum(
        (-1) ** k * mat[0][k] *
        determinant([row[:k] + row[k + 1:] for row in mat[1:]])
        for k in range(x)
    )


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    # 1. Check if it's a list of lists
    # Note: Logic changed to 'if not list OR if any row is not list'
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    x = len(matrix)

    # 2. Check if it's empty or non-square
    if x == 0 or any(len(row) != x for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 3. Special case for 1x1 matrix: minor is defined as 1
    if x == 1:
        return [[1]]

    minor_matrix = []
    for i in range(x):
        minor_row = []
        for j in range(x):
            # Create the (n-1)x(n-1) submatrix by removing row i and column j
            sub_mat = [row[:j] + row[j+1:] for row in
                       (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_mat))
        minor_matrix.append(minor_row)

    # 4. Correct indentation: return must be aligned with the outer 'for' loop
    return minor_matrix

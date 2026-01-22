#!/usr/bin/env python3
"""
Accross the planes
"""


def add_matrices2D(mat1, mat2):
    """
    Adding matrices element-wise
    """
    if len(mat1) != len(mat2):
        return None
    new_matrice = []
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        new_matrice.append(row)
    return new_matrice

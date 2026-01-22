#!/usr/bin/env python3
"""
matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Matrix multiplication
    """
    if len(mat1[0] == len(mat2)):
            new_matrice =[]
            for row1 in range(len(mat1)):
                m = []
                for col2 in range(len(mat2[0])):
                    value = 0
                    for col1 in range(len(mat1)):
                        value += (mat1[row1][col1] * mat2[row1][col2])
                    m.append(value)
                new_matrice.append(m)
            return new_matrice
    else:
        return None

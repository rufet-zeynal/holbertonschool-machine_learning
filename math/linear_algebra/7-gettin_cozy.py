#!/usr/bin/env python3
"""
Gettin' cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenating two matrices along a specific axis
     """
     if (len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0])):
         return None

     if (len(mat1[0] == len(mat2[0])) and (axis == 0):
             return mat1 + mat2

    elif (len(mat1) == len(mat2)) and (axis == 1):
        return [mat1[i] + mat2[i] for i in range(len(mat1)]
    else:
        return None

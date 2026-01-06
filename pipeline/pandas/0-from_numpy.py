#!/usr/bin/env python3
"""
Module containing function from numpy
"""
import pandas as pd
def from_numpy(array):
    """
        Creates a pd.DataFrame from a np.ndarray
        The columns are labeled in alphabetical order and capitalized
        """
    # Get the number of columns in the array
    cols = array.shape[1]

    # Generate labels A, B, C... using ASCII values (65 is 'A')
    col_names = [chr(65 + i) for i in range(cols)]

    # Create the DataFrame using the provided array and generated names
    return pd.DataFrame(array, columns=col_names)
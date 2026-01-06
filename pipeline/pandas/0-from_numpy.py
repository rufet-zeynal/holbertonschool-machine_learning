#!/usr/bin/env python3
"""
Module to convert numpy ndarray to pandas DataFrame
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    """
    # Get the number of columns in the array
    cols = array.shape[1]

    # Generate labels: A, B, C... based on the number of columns
    # ord('A') is 65; we go from 65 to 65 + cols
    alphabet = [chr(i) for i in range(65, 65 + cols)]

    # Create the DataFrame with the specified column labels
    df = pd.DataFrame(array, columns=alphabet)

    return df

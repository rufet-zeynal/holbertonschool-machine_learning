#!/usr/bin/env python3
"""
Importing pandas library
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Function that reads a csv file and returns a pandas dataframe
    """
    return pd.read_csv(filename, sep=delimiter)

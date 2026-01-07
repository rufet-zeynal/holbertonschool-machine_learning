#!/usr/bin/env python3
"""
Importing pandas library
"""
import pandas as pd


def from_file(filename, delimiter):
    return pd.read_csv(filename, sep=delimiter)

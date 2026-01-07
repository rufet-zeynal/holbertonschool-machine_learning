#!/usr/bin/env python3
"""
First line
"""


def index(df):
    """
    Setting Timestamp as index of dataframe
    """
    df = df.set_index("Timestamp")
    return df

#!/usr/bin/env python3
"""
First line
"""


def prune(df):
    """
    Removing entries when Close has NaN values
    """
    df = df.dropna(subset=['Close'])
    return df

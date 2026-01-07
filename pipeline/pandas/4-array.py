#!/usr/bin/env python3
"""
First line
"""


def array(df):
    """
    10 rows of High and Close columns
    """
    df = df[['High', 'Close']].tail(10)
    return df.to_numpy()

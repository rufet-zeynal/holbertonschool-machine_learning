#!/usr/bin/env python3
"""
First line
"""


def high(df):
    """
    Sorting df by High price in desc
    """
    df = df.sort_values(by=['High'], ascending=False)
    return df

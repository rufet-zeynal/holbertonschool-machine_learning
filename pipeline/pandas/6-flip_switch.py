#!/usr/bin/env python3
"""
First line
"""


def flip_switch(df):
    """
    Sorting data in reverse order
    """
    df = df.sort_values(ascending=False)
    transposed_df=df.T
    return transposed_df

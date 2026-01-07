#!/usr/bin/env python3
"""
First line
"""


def index(df):
    """
    Setting Timestamp as index of dataframe
    """
    df['Timestamp'] = df.index
    return df

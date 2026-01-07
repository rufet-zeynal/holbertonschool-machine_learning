#!/usr/bin/env python3
"""
First line
"""


def slice(df):
    """
    Selecting every 60th raw
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df.iloc[::60]

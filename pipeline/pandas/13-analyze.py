#!/usr/bin/env python3
"""
First line
"""


def analyze(df):
    """
    Description of columns except Timestamp
    """
    df = df.describe
    df = df.drop(columns=['Timestamp'])
    return df

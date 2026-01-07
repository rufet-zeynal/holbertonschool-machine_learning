#!/usr/bin/env python3
"""
Importing pandas library
"""
import pandas as pd


def array(df):
    """
    10 rows of High and Close columns
    """
    df=df[['High', 'Close']].tail(10)
    df.to_numpy()
    return df

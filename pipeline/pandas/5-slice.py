#!/usr/bin/env python3
"""
First line
"""
import pandas as pd


def slice(df):
    df = df[['High', 'Low', 'Close', 'Volume_BTC']]
    return df.iloc[::60]

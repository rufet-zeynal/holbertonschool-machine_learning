#!/usr/bin/env python3
"""
First line
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Hierarchy function
    """
    df1 = index(df1)
    df2 = index(df2)
    initial, last = 1417411980, 1417417980
    df1_new = df1.loc[initial:last]
    df2_new = df2.loc[initial:last]
    df = pd.concat([df2_new, df1_new], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0,1)
    df = df.sort_index()
    return df

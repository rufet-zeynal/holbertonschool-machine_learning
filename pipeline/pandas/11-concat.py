#!/usr/bin/env python3
"""
First line
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatting two dataframes
    """
    df1 = df1.set_index("Timestamp")
    df2 = df2.set_index("Timestamp")
    df2 = df2[df2.index <= 1417411920]
    df = pd.concat([df1, df2], keys=['bitstamp', 'coinbase'])
    return df

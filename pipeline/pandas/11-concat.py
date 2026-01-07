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
    df1 = index(df1)
    df2 = index(df2)
    df2_new = df2.loc[:1417411920]
    df = pd.concat([df2_new, df1], keys=['bitstamp', 'coinbase'])
    return df

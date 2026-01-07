#!/usr/bin/env python3
"""
First line
"""
from operator import index


def hierarchy(df1, df2):
    """
    Hierarchy function
    """
    df1 = index(df1)
    df2 = index(df2)
    df1_new = df1.loc[1417411980:1417417980]
    df2_new = df2.loc[1417417980:1417417980]
    df = pd.concat([df2_new, df1_new], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0,1)
    df = df.sort_index()
    return df

#!/usr/bin/env python3
"""
First line
"""


def fill(df):
    """
    Drop 'Weighted_Price' column
    Fill missing 'Close' with previous row
    Fill missing 'High', 'Low', 'Open' with 'Close'
    Set missing 'Volume_(BTC)' and 'Volume_(Currency)' to 0
    """
    df = df.drop(columns='Weighted_Price')
    df['Close'] = df['Close'].fillna(method='ffill')
    for i in ['Open', 'High', 'Low']:
        df[i] = df[i].fillna(df['Close'])
    for i in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[i] = df[i].fillna(0)
    return df

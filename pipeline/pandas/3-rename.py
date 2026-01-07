#!/usr/bin/env python3
"""
Importing pandas library
"""
import pandas as pd


def rename(df):
    """
    Rename columns of pandas dataframe and converting timestamp values to datetime format
    """
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[["Datetime", "Close"]]

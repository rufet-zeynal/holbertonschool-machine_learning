#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
"""
Fuction creating df from np.ndarray
"""
    df = pd.DataFrame(array)
    df.columns = [chr(65+1) for i in range(df.shape[1])]
    return df
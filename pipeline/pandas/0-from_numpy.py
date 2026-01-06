#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
"""
Fuction creating df from np.ndarray
"""
    pd.set_option("display.expand_frame_repr", True)
    pd.set_option("display.max_columns", None)
    df = pd.DataFrame(array)
    df.columns = [chr(65 + i) for i in range(array.shape[1])]
    return df
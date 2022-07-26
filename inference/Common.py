import pandas as pd
import numpy as np

def get_NA_index(df):
    nan_cond = pd.isna(df).any(axis=1)
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5)) | \
                   ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89)) | \
                   ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
                    (df['Ndir'] > 720))
    indices_na = np.where(nan_cond | invalid_cond)
    indices_right = np.where(~nan_cond & ~invalid_cond)
    return list(indices_na[0]), list(indices_right[0])


def get_diff(row):
    diff = np.array(row['Etmp_seq']) - np.array(row['Itmp_seq'])
    return list(diff)

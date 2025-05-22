import numpy as np
import pandas as pd

def normalize(df):
    """
    Normalize a DataFrame using min-max normalization.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be normalized.

    Returns:
    - pd.DataFrame: the normalized DataFrame.
    """
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    df = (df - mins) / (maxs - mins)
    return df, mins, maxs

def denormalize(df, mins, maxs):
    """
    Denormalize a DataFrame using min-max normalization.
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be denormalized.
    - mins (pd.Series): The minimum values used for normalization.
    - maxs (pd.Series): The maximum values used for normalization.
    Returns:
    - pd.DataFrame: the denormalized DataFrame.
    """
    df = df * (maxs - mins) + mins
    return df
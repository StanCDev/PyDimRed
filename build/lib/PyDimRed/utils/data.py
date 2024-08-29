"""
Basic utility module to load and save csv files to different formats
"""

import pandas as pd
import numpy as np


def load_data_np(path) -> np.array:
    """
    Load a csv file with a given file and format it as a numpy array

    Args:
    _____
        path (file | str | pathlib.Path): path to file

    Returns:
    --------
        data (np.array)
    """
    X_tilde = np.loadtxt(path, delimiter=",")
    return X_tilde


def load_data_df(path) -> pd.DataFrame:
    """
    Load a csv file with a given file and format it as a pandas data frame

    Args:
    _____
        path (file | str | pathlib.Path): path to file

    Returns:
    --------
        data (pd.df)
    """
    X_tilde = pd.read_csv(path, sep=",")
    return X_tilde


def save_data(path, arr: np.array) -> None:
    """
    Save a numpy array to a .csv file

    Args:
    _____
        path (file | str | pathlib.Path): path to file

    Returns:
    --------
        None
    """
    return np.savetxt(path, arr, delimiter=",")

"""
Dimensionality reduction utility functions
"""

from pathlib import Path
import os
from os.path import basename, splitext

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import numpy as np

from .data import save_data
from PyDimRed.exceptions import checkCondition, checkDimensionCondition
from PyDimRed.transform import TransformWrapper


def reduce_data_with_params(
    X: np.array,
    y: np.array,
    *method_params: dict,
    save: bool = False,
    base_save_folder: Path = None,
    n_jobs: int = 1,
):
    """
    Given a data set (X, y) and dict(s) containing parameter to value mapping, obtain all reduced data sets.


    Args:
    -----
        X (np.array): N x D dimensional data array

        y (np.array): N dimensional labels array

        \*method_params (dict): dictionary input / set_params input to define DR models that will reduce data.
        The resulting parameters will be the cross product of all combinations of parameters in a dictonary, and the union of
        disjoint dictionaries. See sklearn.model_selection.ParameterGrid for more information.

        save (bool): if True save data in disk

        base_save_folder (pathlib.Path): directory where data will be saved if save is True

        n_jobs (int): Number of jobs for the joblib backend. Default = 1. Setting n_jobs = -1 is equivalent to
            setting to the maximum number of jobs system can handle. Note that for small data sets or few parameter
            combinations it can be quicker to set n_jobs = 1

    Returns:
    --------
        tuple : list of reduced data, list of all paths (if save = False list of all names returned)

    Examples:
    ---------
    Examples of values for methodParams:
    Single parameter
    
    >>> param = {"method" : ["TSNE", "UMAP"] , "n_nbrs" : [20,60,80]}
    >>> reduce_data_with_params(X,y,param)

    Multiple parameters

    >>> params = [{"method" : ["TRIMAP"] , "n_nbrs" : [20,60,80] , "n_outliers" : [10,20,30]} , {"method" = ["PCA"]} ]
    >>> reduce_data_with_params(X,y,*params)

    """
    checkDimensionCondition(
        X.shape[0] == y.shape[0],
        "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
        X.shape,
        y.shape,
    )
    checkCondition(
        not (save and base_save_folder is None),
        "Cannot save if directory to save csv files in is not specified",
    )
    checkCondition(
        len(method_params) > 0,
        "Must have at least one parameter configuration")

    if save:
        assert os.path.isdir(
            str(base_save_folder)
        ), f"baseSaveFolder:\
        {base_save_folder} ({str(base_save_folder)}) is not a directory!"

    x_reds_paths = Parallel(n_jobs=n_jobs)(
        delayed(__reduce_one_model)(params, X, y, save, base_save_folder)
        for params in ParameterGrid(method_params)
    )

    x_reds, paths = zip(*x_reds_paths)

    return list(x_reds), list(paths)


def __reduce_one_model(
    params: dict, X: np.array, y: np.array, save: bool, base_save_folder: Path
):
    model: TransformWrapper = TransformWrapper.from_params(**params)

    new_params = params.copy()
    method = new_params.get("method")
    if method is not None:
        new_params.pop("method")
    else:
        method = new_params.pop("base_model")

    file_name = (
        str(method)
        + "-"
        + "-".join(
            f"{param_name}{param_value}"
            for param_name, param_value in new_params.items()
        )
    )
    if save:
        file_name = base_save_folder / f"{file_name}.csv"

    x_red = model.fit_transform(X, y)
    if save:
        save_data(str(file_name), x_red)

    return x_red, file_name


def path_to_names(paths: list[str]) -> list[str]:
    """
    Small utility methods to extract filename without file extension from a list of paths

    Args:
    -----
        paths (list[str]): list of paths

    Returns:
    --------
        list[str]
    """
    return [splitext(basename(path))[0] for path in paths]

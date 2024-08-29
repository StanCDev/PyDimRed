import pytest
import pathlib
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris

from PyDimRed.utils.data import load_data_np
from PyDimRed.utils.dr_utils import reduce_data_with_params

@pytest.fixture(scope="module")
def load_data_train():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture(scope="module")
def get_params():
    params = [{"method" : ["UMAP", "TSNE", "TRIMAP"] , "n_nbrs" : [5,10,15]}, {"method" : ["TRIMAP"], "n_nbrs" : [10,12], "n_outliers" : [3,4]} , {"method" : ["PACMAP"]}]
    return params

def test_reduce_all(load_data_train,get_params):
    """
    Test reduce on a list of params
    """
    X,y = load_data_train
    params = get_params

    Xreds, paths = reduce_data_with_params(X,y, *params,n_jobs=-1)
    assert len(Xreds) == len(ParameterGrid(params)) and len(Xreds) == len(paths)
    
    for Xred, path in zip(Xreds, paths):
        assert Xred.shape == (X.shape[0], 2), "Reduced data of improper dimension"
        assert ('.csv' not in path) and ('/' not in path), "When save is false should not be a path but a name"
    return

def test_reduce_all_save(load_data_train,get_params,tmp_path):
    """
    Test reduce on a list of params
    """
    X,y = load_data_train
    params = get_params

    Xreds, paths = reduce_data_with_params(X,y, *params,save=True, base_save_folder=tmp_path,n_jobs=-1)
    assert len(Xreds) == len(ParameterGrid(params)) and len(Xreds) == len(paths)
    
    for Xred in Xreds:
        assert Xred.shape == (X.shape[0], 2)

    for path, Xred in zip(paths, Xreds):
        XredFile = pathlib.Path(path)
        assert XredFile.is_file(), "File with this name is invalid / not generated"

        XredLoaded = load_data_np(path)

        assert np.array_equal(XredLoaded, Xred), "Saved array and returned array are not equal!"

    return
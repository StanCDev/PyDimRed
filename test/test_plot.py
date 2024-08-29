import pytest
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from os.path import basename, splitext
import re
from sklearn.datasets import load_iris

from PyDimRed.plot import display, display_group, display_heatmap, display_heatmap_df, display_training_validation
from PyDimRed.utils.dr_utils import reduce_data_with_params, path_to_names
from PyDimRed.transform import TransformWrapper

TEST_SIZE = 0.3

@pytest.fixture(scope="module")
def load_data_train():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture(scope="module")
def load_data_train_and_test():
    X, y = load_iris(return_X_y=True)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=TEST_SIZE)
    return Xtrain, Xtest, ytrain, ytest


@pytest.fixture(scope="module") # return type is list[dict[string,list]]
def get_params():
    params = [{"method" : ["TSNE", "UMAP", "TRIMAP"] , "n_nbrs" : [5,10,15,20]} , {"method" : ["PACMAP"]}]
    return params


def test_display_single(load_data_train):
    X , y = load_data_train
    print(f"Shape of X and y are {X.shape} and {y.shape} respectively")
    Xred = TransformWrapper(method="TRIMAP",n_nbrs=12).fit_transform(X,y)
    display(Xred,y,hue_label="digit")
    assert True
    return

def test_display_group(tmp_path,load_data_train, get_params):
    X, y = load_data_train
    params = get_params

    XtrainReds, paths = reduce_data_with_params(X,y,*params,save=True,base_save_folder=tmp_path)

    names = path_to_names(paths)

    display_group(names=names,X_train_list=XtrainReds,y_train=y,nbr_cols=4,nbr_rows=4)

    display_group(names=names,X_train_list=XtrainReds,y_train=y,nbr_cols=4,nbr_rows=4,figsize=(8,8))

    assert True

    return

def test_display_group_test_data(load_data_train_and_test, get_params, tmp_path):
    Xtrain, Xtest, ytrain, ytest = load_data_train_and_test

    params = get_params

    XtrainReds, paths = reduce_data_with_params(Xtrain, ytrain, *params, save=True, base_save_folder=tmp_path)
    names = path_to_names(paths)

    XtestReds , _ = reduce_data_with_params(Xtest, ytest, *params)

    display_group(names=names,X_train_list=XtrainReds,y_train=ytrain,X_test_list=XtestReds, y_test=ytest ,nbr_cols=4,nbr_rows=4)

    assert True

    return

def test_display_group_grid(tmp_path,load_data_train):
    X, y = load_data_train
    range1 = [5,10,15,20]
    params = {"method" : ["TRIMAP"] , "n_nbrs" : range1, "n_outliers" : range1}

    Xreds, paths = reduce_data_with_params(X,y,params,save=True, base_save_folder=tmp_path)

    names = [splitext(basename(path))[0] for path in paths]
    pattern = re.compile(r'[0-9]+')
    names = ['-'.join(pattern.findall(name)) for name in names]
    
    cols = len(params["n_nbrs"])
    rows = len(params["n_outliers"])

    display_group(names=None,X_train_list=Xreds,y_train=y,nbr_cols=cols,nbr_rows=rows,marker_size=10,
                grid_x_label=[f"n_inliers={i}" for i in range1],
                grid_y_label=[f"n_outliers={i}" for i in range1],
                title="TRIMAP with variation in number of inliers and number of outliers")

    assert True

    return

def test_displayHeatmap():
    d1, d2 = 9, 10
    X = np.random.rand(d1,d2)

    display_heatmap(X, range(d1), range(d2), "x", "y")

    assert True
    return

def test_displayHeatmapDF():
    d1, d2 = 9, 10

    f = lambda x : (x % d2) ** 2 + (x // d2) ** 2 + 1

    # Create the range arrays for i and j
    xRange = np.arange(d1)
    yRange = np.arange(d2)

    # Create the meshgrid
    x, y = np.meshgrid(xRange, yRange, indexing='ij')
    ### ravel flattens the arrays
    data = {
    'x': x.ravel(),
    'y': y.ravel(),
    'z': list(map(f ,range(d1 * d2)))
}

    df = pd.DataFrame(data)
    display_heatmap_df(df, 'x', 'y', 'z')
    assert True
    return

def test_displayTrainVal():
    parameters = [5, 10, 15, 20] # list of n_nbrs
    y = {
        "TSNE" : [94.0990, 93.4519, 91.9385, 92.4469], 
        "TRIMAP" : [82.5421, 82.1633, 82.5700, 82.38504968853520]
    }
    y = pd.DataFrame(y)
    display_training_validation(parameters, y)
    assert True
    return
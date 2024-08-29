import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np ### do I need this?
from sklearn.datasets import load_iris, load_diabetes

import umap
import trimap
from sklearn import decomposition
import pacmap

from PyDimRed.utils.dr_utils import reduce_data_with_params
from PyDimRed.evaluation import one_NN_accuracy, ModelEvaluator


TEST_SIZE = 0.3

@pytest.fixture(scope="module")
def load_data_train():
    X, y = load_iris(return_X_y=True)
    return X, y

@pytest.fixture(scope="module")
def load_data_train_cts():
    X, y = load_diabetes(return_X_y=True)
    ### train test split is called to reduce size of data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4)
    return X_train, y_train

@pytest.fixture(scope="module")
def load_data_train_and_test():
    X, y = load_iris(return_X_y=True)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=TEST_SIZE)
    return Xtrain, Xtest, ytrain, ytest

@pytest.fixture(scope="module") # return type is list[dict[string,list]]
def get_params(request):
    match request.param:
        case "SMALL":
            params = {"method" : ["TSNE", "UMAP"] , "n_nbrs" : [20,30]}
            return params
        case "MODEL":
            params = [
                {"base_model" : [umap.UMAP()], "n_neighbors" : [2,5,8]},
                {"base_model" : [decomposition.PCA()], "n_components" : [2,3]},
                {"base_model" : [trimap.TRIMAP()], "n_inliers" : [5,8], "n_outliers" : [6,9]},
                {"base_model" : [pacmap.PaCMAP()], "n_neighbors" : [2,5,8], "save_tree" : [True]}
                ]
            return params
        case "COMPLEX":
            params = [{"method" : ["UMAP", "TRIMAP"] , "n_nbrs" : [5,10]} , {"method" : ["PACMAP"]}]
            return params
        case "SMALL_LIST":
            params = [{"method" : ["TSNE", "UMAP"] , "n_nbrs" : [20,30]}]
            return params
        case _:
            return []

def floatEquals(x, y, epsilon=1e-9):
    return abs(x - y) <= epsilon

@pytest.mark.parametrize(
    "get_params",
    [
        param_type for param_type in "MODEL COMPLEX SMALL_LIST".split()
    ],
    indirect=True
)
def test_oneNNAcc_class(load_data_train_and_test, get_params):
    ### Things to check:
    #### 1. All values in dictionary are in range 0 -> 100
    Xtrain, Xtest, ytrain, ytest = load_data_train_and_test
    params = get_params

    Xredstrain, _ = reduce_data_with_params(Xtrain,ytrain, *params)
    Xredstest, _  = reduce_data_with_params(Xtest,ytest, *params)

    for Xredtrain, Xredtest in zip(Xredstrain, Xredstest):
        acc = one_NN_accuracy(X_train=Xredtrain,y_train=ytrain, X_val=Xredtest, y_val = ytest)
        assert acc >= 0 and acc <= 1, "accuracy must be in correct range"
    return

from sklearn.datasets import make_swiss_roll

@pytest.mark.parametrize(
    "get_params",
    [
        param_type for param_type in "MODEL COMPLEX SMALL_LIST".split()
    ],
    indirect=True
)
def test_oneNNAcc_reg(get_params):
    ### Things to check:
    #### 1. All values in dictionary are in range 0 -> 100
    X , y = make_swiss_roll(n_samples = 1000, noise=1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y)
    params = get_params

    Xredstrain, _ = reduce_data_with_params(Xtrain,ytrain, *params)
    Xredstest, _  = reduce_data_with_params(Xtest,ytest, *params)

    for Xredtrain, Xredtest in zip(Xredstrain, Xredstest):
        score = one_NN_accuracy(X_train=Xredtrain,y_train=ytrain, X_val=Xredtest, y_val = ytest,task_type="REGRESSION")
        ### If we use R^2 it should be in range -1, 1
        ### If we use MSE it should be geq. 0
        print(f"### Score = {score}")
        #assert score >= -1 and score <= 1 , "score must be in correct range"
    return


from sklearn.metrics import mean_absolute_error

def pipeline(reg_or_class):
    match reg_or_class:
        case "REG":
            return Pipeline([("Scaler", StandardScaler()), ("OneNN", KNeighborsRegressor(1))])
        case "CLASS":
            return Pipeline([("Scaler", StandardScaler()), ("OneNN", KNeighborsClassifier(1))])
        case _:
            raise ValueError("must have arg = 'REG' or arg = 'CLASS'")

def _predicate(param):
    value = param[1]
    if value is None:
        return False
    if isinstance(value, float):
        return not np.isnan(value)
    return True

@pytest.mark.parametrize(
    #"K, n_repeats, estimator, regr_or_class, scorer",
    "K, n_repeats, get_params, estimator, regr_or_class, scorer",
    [(k, n_repeats, param_type, estimator, regr_or_class, scorer) 
        for k in [2,4]
        for n_repeats in [1,3]
        for param_type in "MODEL COMPLEX".split()
        for estimator, regr_or_class, scorer in [(pipeline("REG"), "REG", None), (pipeline("REG"), "REG", mean_absolute_error), (pipeline("CLASS"), "CLASS", None)]
    ],
    indirect=["get_params"]
)
def test_grid_search_dr_performance_complex(K, n_repeats, get_params, estimator, regr_or_class, scorer, load_data_train, load_data_train_cts):
    X,y = None, None
    if regr_or_class == "CLASS":
        X, y = load_data_train
    else:
        X, y = load_data_train_cts
    params = get_params

    model_eval = ModelEvaluator(X=X,y=y,parameters=params,estimator=estimator,scorer = scorer, K=K, n_repeats=n_repeats,n_jobs=-1)

    maxScore, maxParams, dfOut = model_eval.grid_search_dr_performance()
    print()
    print(dfOut.to_markdown())

    if regr_or_class == "CLASS":
        assert maxScore >= 0 and maxScore <= 100, "accuracy must be in correct range"

    testMaxScore = 0 if regr_or_class == "CLASS" else -10
    testMaxScoreVariance = 0
    testBestParams = None

    for index, row in dfOut.iterrows():
        acc = row['score']
        var = row['variance']
        if acc > testMaxScore:
            testMaxScore = acc
            testMaxScoreVariance = var
            testBestParams: dict = row.to_dict()
        if regr_or_class == "CLASS":
            assert acc >= 0 and acc <= 100, "accuracy must be in correct range"

    assert floatEquals(testMaxScore, maxScore), "Returned maximum accuracy is not true maximum accuracy"
    maxParams['score'] = maxScore
    maxParams['variance'] = testMaxScoreVariance
    testBestParamsFiltered = dict(
        filter(
            _predicate, 
            testBestParams.items()
            )
        )
    assert (maxParams == testBestParamsFiltered), "Function doesn't return correct parameters"
    return


@pytest.mark.parametrize(
    #"K, n_repeats, estimator, regr_or_class, scorer",
    "get_params, estimator, regr_or_class, scorer",
    [(param_type, estimator, regr_or_class, scorer) 
        for param_type in "SMALL MODEL COMPLEX".split()
        for estimator, regr_or_class, scorer in [(pipeline("REG"), "REG", None), (pipeline("REG"), "REG", mean_absolute_error), (pipeline("CLASS"), "CLASS", None)]
    ],
    indirect=["get_params"]
)
def test_crossVal(get_params, estimator, regr_or_class, scorer, load_data_train):
    X, y = load_data_train
    params = get_params

    model_eval = ModelEvaluator(X=X,y=y,parameters=params, estimator=estimator, K=3,n_repeats=2, scorer=scorer,n_jobs=-1)

    maxScore, maxParams, dfOut = model_eval.cross_validation()

    testMaxScore = -10e9
    testBestParams = None

    for index, row in dfOut.iterrows():
        score = row['mean_test_score']
        if score > testMaxScore:
            testMaxScore = score
            testBestParams: dict = row.to_dict()
        # assert acc >= 0 and acc <= 100, "accuracy must be in correct range"

    assert floatEquals(testMaxScore, maxScore), "Returned maximum accuracy is not true maximum accuracy"

    maxParamsRelevant : dict = testBestParams['params']
    assert (maxParams == maxParamsRelevant), "Function doesn't return correct parameters"
    return

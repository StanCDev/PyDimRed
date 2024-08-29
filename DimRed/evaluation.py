"""
This module provides a class to evaluate the performance of DR methods.
The two main methods are cross validation and a variation of it where seperate models are trained
on a train and validation set due to some DR models only having a 'fit_transform' method.
"""

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, RepeatedKFold
from sklearn.metrics import make_scorer

import pandas as pd
from joblib import Parallel, delayed
from .exceptions import checkDimensionCondition, checkCondition
from .transform import TransformWrapper


def one_NN_accuracy(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    task_type: str = "CLASSIFICATION",
) -> float:
    """Trains a 1-NN algorithm on Xtrain and ytrain to output accuracy of prediction on Xtest dataset

    Args:
    -----
        X_train (np.array): N x D dimensional array of training data. N data points, D features

        y_train (np.array): N dimensional array of training labels

        X_val (np.array): N x D dimensional array of test data

        y_val (np.array):  N dimensional array of test labels

        task_type (str): Type of task - 'classification' or 'regression'.

    Returns:
    --------
        accuracy (float): accuracy of test data with 1-NN
    """
    checkDimensionCondition(
        X_train.shape[0] == y_train.shape[0],
        "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
        X_train.shape,
        y_train.shape,
    )
    checkDimensionCondition(
        X_val.shape[0] == y_val.shape[0],
        "Test data doesn't have the same number of samples (dimension 0 doesn't correspond)",
        X_val.shape,
        y_val.shape,
    )
    checkCondition(
        task_type == "CLASSIFICATION" or task_type == "REGRESSION",
        f"task_type must correspond to either 'CLASSIFICATION' or 'REGRESSION' task_type = {task_type}",
    )

    scorers = {
        "CLASSIFICATION": KNeighborsClassifier(n_neighbors=1),
        "REGRESSION": KNeighborsRegressor(n_neighbors=1),
    }
    scaler = StandardScaler()
    one_nn = scorers[task_type]
    # Pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("one_nn", one_nn)])
    # Train and prediction
    pipe.fit(X_train, y_train.reshape(-1))
    score = pipe.score(X=X_val, y=y_val)

    return score


class ModelEvaluator:
    """ModelEvaluator is a class that evaluates the performance of DR methods on a given X, y data set."""

    def __init__(
        self,
        X: np.array,
        y: np.array,
        parameters: dict[str, list] | list[dict[str, list]],
        estimator=Pipeline(
            steps=[("Scaler", StandardScaler()),
                ("OneNN", KNeighborsClassifier(1))]
        ),
        scorer=None,
        K: int = 5,
        max_or_min: str = "MAX",
        n_repeats: int = 1,
        n_jobs: int = 1,
    ) -> None:
        """
        ModelEvaluator constructor

        Args:
        -----
            X (np.array): N x D dimensional dataset

            y (np.array): N dimensional features

            parameters (dict[str, list]): dictionary that maps from parameter name (str) to values parameter will be set to (list)

            estimator : estimator that determines fitness of model. The estimator must implement score() and fit() methods.

            scorer : Optional function to override estimator.score() function. For example default estimator has R squared score function
                than can be replaced by sklearn's Mean Square Error scoring function. Must have signature: score_func(y, y_pred, \*\*kwargs).
                Default is None

            K (int): number of folds in K-fold cross validation like split. Default = 5

            max_or_min (str) : 'MAX' if best score is maximum, else 'MIN'

            n_repeats (int): number of repeats for K_fold. Default = 1 (no repeats)

            n_jobs (int): Number of jobs for the joblib backend. Default = 1. Setting n_jobs = -1 is equivalent to
            setting to the maximum number of jobs system can handle

        Returns:
        --------
            None
        """
        checkDimensionCondition(
            X.shape[0] == y.shape[0],
            "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
            X.shape,
            y.shape,
        )
        N = X.shape[0]
        checkCondition(
            K >= 2 and K < N,
            f"In K - Fold cross validation value of K must be bound between 2 and the number of data points, \
        in this case K = {K} and [1,N] = [1,{N}]",
        )
        checkCondition(len(parameters) >= 1, "parameters dict cannot be empty")
        checkCondition(
            n_repeats >= 1,
            f"Cannot have less than n_repeats = 1 as this corresponds to running K-Fold Validation once: n_repeats = {n_repeats} ",
        )
        checkCondition(
            max_or_min == "MAX" or max_or_min == "MIN",
            f"max_or_min is not 'MAX' or 'MIN' but {max_or_min}",
        )
        checkCondition(
            hasattr(
                estimator,
                "score") and callable(
                getattr(
                    estimator,
                    "score")),
            "estimator does not have a 'score()' function",
        )
        checkCondition(
            hasattr(estimator, "fit") and callable(getattr(estimator, "fit")),
            "scorer does not have a fit_transform function",
        )
        if scorer is not None:
            checkCondition(
                callable(
                    scorer), "If scorer is not None it must be a callable function"
            )
        self.X = X
        self.y = y
        self.parameters = parameters
        self.estimator = estimator
        self.scorer = scorer
        self.K = K
        self.max_or_min = max_or_min
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        return

    @staticmethod
    def __iter_grid_search(
        estimator,
        scorer,
        X: np.array,
        y: np.array,
        i: int,
        param: dict,
        j: int,
        split: tuple,
    ):
        """Helper function that transforms train and test data seperately, fits estimator on train and obtains
        predictions on test, calculates a score via estimator.scorer(), or scorer if not None. Method is static as joblib
        does not work with methods from self

        Args:
        -----
            estimator : estimator that determines fitness of model. The estimator must implement score() and fit() methods.

            scorer : Optional function to override estimator.score() function. For example default estimator has R squared score function
                than can be replaced by sklearn's Mean Square Error scoring function. Must have signature: score_func(y, y_pred, **kwargs).
                Default is None

            X (np.array): N x D dimensional dataset

            y (np.array): N dimensional features

            i (int): Passed by argument to conform to joblib API, is returned to know order of call

            param (dict): map of parameter name to parameter value for TransformWrapper

            j (int): Passed by argument to conform to joblib API, is returned to know order of call

            split (tuple): tuple of np.array indices for train and validation data

        Returns:
        --------
            score (float): model score of transforming x_train and x_test seperately
        """
        train_index, test_index = split

        Xtrain = X[train_index]
        Xval = X[test_index]
        ytrain = y[train_index]
        yval = y[test_index]

        # Transformed train data
        model: TransformWrapper = TransformWrapper.from_params(**param)
        XtrainRed = model.fit_transform(Xtrain, ytrain)

        # Transformed test data
        model: TransformWrapper = TransformWrapper.from_params(**param)
        XtestRed = model.fit_transform(Xval, yval)

        # Compute Score / accuracy
        estimator.fit(XtrainRed, ytrain.reshape(-1))
        score = 0
        if scorer is not None:
            yvalpred = estimator.predict(X=XtestRed)
            score = scorer(yvalpred, yval)
        else:
            score = estimator.score(X=XtestRed, y=yval)
        return (i, j, score)

    def __get_all_param_names(self):
        """
        Finds each different parameter in self.params and returns it in a set

        Returns:
        --------
            set[int]
        """
        temp_parameters = None
        match self.parameters:
            case [head, *tail]:
                temp_parameters = self.parameters
            case _:
                temp_parameters = [self.parameters]

        all_parameters: set[str] = set()

        for param_dict in temp_parameters:
            for param_name in param_dict:
                all_parameters.add(param_name)

        return all_parameters

    def grid_search_dr_performance(self):
        """
        Evaluate the performance of dimensionality reduction (DR) models.

        This method performs the following steps to assess the quality of DR models like TSNE and TRIMAP:

        1. Fits a DR model on the training data and transforms the training data.
        2. Fits a new DR model on the validation data and transforms the validation data.
        3. Uses an estimator to obtain performance value. For classification an sklearn pipeline with a Standard Scaler and 1-Nearest Neighbour classifier is used by default. For regression a 1-NN regressor can be used instead of a classifier

        Process is repeated according to 'K' fold cross validation with 'n_repeats' repetitions per fold

        Returns:
        --------
            best_score (float): best score (accuracy in this case)
            best_params (dict): best parameters
            results (pd.Dataframe): dataframe of results. There is a column for each parameter and two extra columns:
            one for empirical mean score with values of parameters on that line, and one for the empirical variance
        """
        all_param_combinations = ParameterGrid(self.parameters)
        N = len(all_param_combinations)  # Number of parameter combinations

        all_parameters = self.__get_all_param_names()

        output_grid = {param: [0] * N for param in all_parameters}
        output_grid["score"] = [0] * N
        output_grid["variance"] = [0] * N

        kf = RepeatedKFold(n_splits=self.K, n_repeats=self.n_repeats)

        params_split = (
            (i, param, j, split)
            for i, param in enumerate(all_param_combinations)
            for j, split in enumerate(kf.split(self.X))
        )

        par_out = Parallel(n_jobs=self.n_jobs)(
            delayed(ModelEvaluator.__iter_grid_search)(self.estimator, self.scorer, self.X, self.y, i, param, j, split) for (i, param, j, split) in params_split
        )

        # Simply unflattening list
        temp = [[0] * (self.K * self.n_repeats) for _ in range(N)]

        for i, j, score in par_out:
            temp[i][j] = score

        best_score = None
        best_params = None

        for i, sample in enumerate(temp):
            mean_score = np.mean(sample)
            var_score = np.var(sample)
            params = all_param_combinations[i]

            for param in all_parameters:
                value = params.get(param)
                output_grid[param][i] = value

            output_grid["score"][i] = mean_score
            output_grid["variance"][i] = var_score

            if (best_score is None and best_params is None) or (
                mean_score > best_score and self.max_or_min == "MAX"
            ):
                best_score = mean_score
                best_params = params
            elif (best_score is None and best_params is None) or (
                mean_score < best_score and self.max_or_min == "MIN"
            ):
                best_score = mean_score
                best_params = params

        return best_score, best_params, pd.DataFrame(output_grid)

    def cross_validation(self):
        """Grid search cross validation for dimensionality reduction. 'parameters' defines a map from parameter name to all values that
        parameter takes. Fitness / scoring of DR model is determined by estimator or scorer (must have a score function). Note that all DR models passed
        must implement a 'fit' and 'transform' method to be coherent with the sklearn API

        Returns:
        --------
            best_score (float): best score (accuracy in this case)
            best_params (dict): best parameters
            results (pd.Dataframe): dataframe of results
        """
        # Pipeline, note we set model to pca as a placeholder, will be changed
        pipe = Pipeline(
            steps=[
                ("Transform", TransformWrapper(method="PCA")),
                ("scorer", self.estimator),
            ]
        )

        # Parameter grid
        param_grid = {}
        if isinstance(self.parameters, list):
            for d in self.parameters:
                for param, values in d.items():
                    param_grid[f"Transform__{param}"] = values
        else:
            for param, values in self.parameters.items():
                param_grid[f"Transform__{param}"] = values

        # repeated K - Fold crossval
        kf = RepeatedKFold(n_splits=self.K, n_repeats=self.n_repeats)

        # Scorer
        wrapped_scorer = None
        if self.scorer is not None:
            wrapped_scorer = make_scorer(
                self.scorer, greater_is_better=(self.max_or_min == "MAX")
            )

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            n_jobs=self.n_jobs,
            cv=kf,
            scoring=wrapped_scorer,
        )

        search.fit(self.X, self.y.reshape(-1))

        results = search.cv_results_
        df = pd.DataFrame(results)

        return search.best_score_, search.best_params_, df

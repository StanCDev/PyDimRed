"""
This module provides a wrapper class for any dimensionality reduction (DR) technique
such as PCA, UMAP, TSNE, TRIMAP, and PACMAP. The TransformWrapper class standardizes the
interface for these techniques, making it easier to integrate and use them in
machine learning workflows.
"""

from copy import deepcopy

import numpy as np
from sklearn import decomposition  # manifold for sklearn TSNE
from sklearn.base import BaseEstimator, TransformerMixin
import umap
import trimap
import pacmap
from openTSNE.sklearn import TSNE

from .exceptions import checkCondition


class TransformWrapper(TransformerMixin, BaseEstimator):
    """
    A wrapper class that provides a consistent interface for multiple dimensionality reduction methods.
    """

    def __init__(
        self,
        base_model=None,
        method: str = None,
        n_nbrs: int = 10,
        d: int = 2,
        default_model: bool = False,
        n_outliers: int = 4,
    ):
        """
        Initialize the TransformWrapper with the specified DR method and hyperparameters. Note that either
        base_model or method must be None as they both define the wrapped Transformer. If a 'TransformWrapper' is 
        initialized via the method parameters the 'set_params' and 'get_params' methods will use this classes attributes.
        If a 'TransformWrapper' is initialized via the base_model parameters the 'set_params' and 'get_params' methods
        will call the wrapped objects method. Using the method parameter is more restricting as this class
        does not have acess to every models underlying parameters but it allows for different models to share common parameter
        names like n_nbrs. This allows for easier use of parameter changing in evaluation methods like cross validation.

        Args:
        -----
            base_model : default is None

            method (str): a supported dimensionality reduction method. Available method values are
            'PCA' / 'UMAP' / 'TSNE' / 'TRIMAP' / 'PACMAP'. Default is None

            n_nbrs (int): Common parameter value in DR methods. Can represent n_neighbours / n_inliers / perplexity. default is 10

            d (int): Dimension data is reduced to. Default is 2

            default_model (bool): If True wrapped model with default values is set. Default is False
            
            n_outliers (int): Number of outlier data points. Is used in trimap.TRIMAP. Default is 4

        Returns:
        -------
            TransformWrapper

        Default pacmap is set to true means that pacmap will determine n_nbrs on its own.
        """
        checkCondition(
            (base_model is None) != (method is None),
            "Either model or method argument must be None, but not both",
        )

        self.method: str = method
        self.n_nbrs: int = n_nbrs
        self.n_outliers: int = n_outliers
        self.d: int = d
        self.default_model: bool = default_model
        self.base_model = base_model
        self.__init_model()

    def __check_attributes(self):
        checkCondition(
            self.n_nbrs > 0,
            "n_nbrs (number of neighbours / inliers) must be greater than 0",
        )
        checkCondition(
            self.n_outliers > 0,
            "n_outliers (number of outliers) must be greater than 0",
        )
        checkCondition(
            self.d > 0,
            "d (reduction dimension) must be greater than 0")
        if self.base_model is not None:
            checkCondition(
                not isinstance(self.base_model, TransformWrapper),
                "Cannot pass a TransformWrapper as argument to a TransformWrapper",
            )
            checkCondition(
                hasattr(self.base_model, "fit_transform")
                and callable(getattr(self.base_model, "fit_transform")),
                "base_model does not have a fit_transform function",
            )
            checkCondition(
                hasattr(self.base_model, "get_params")
                and callable(getattr(self.base_model, "get_params")),
                "base_model does not have a get_params function",
            )
            checkCondition(
                hasattr(self.base_model, "set_params")
                and callable(getattr(self.base_model, "set_params")),
                "base_model does not have a set_params function",
            )

    def __init_model(self):
        self.__check_attributes()

        if self.method is not None:
            match self.method:
                case "PCA":
                    self.base_model = decomposition.PCA(n_components=self.d)
                case "UMAP":
                    self.base_model = (
                        umap.UMAP(n_neighbors=self.n_nbrs, n_components=self.d)
                        if not self.default_model
                        else (umap.UMAP(n_components=self.d))
                    )
                case "TSNE":
                    self.base_model = (
                        TSNE(n_components=self.d, perplexity=self.n_nbrs)
                        if not self.default_model
                        else (TSNE(n_components=self.d))
                    )
                case "TRIMAP":
                    self.base_model = (
                        trimap.TRIMAP(
                            n_dims=self.d,
                            n_inliers=self.n_nbrs,
                            n_outliers=self.n_outliers,
                        )
                        if not self.default_model
                        else (trimap.TRIMAP(n_dims=self.d))
                    )
                case "PACMAP":
                    self.base_model = (
                        pacmap.PaCMAP(
                            n_components=self.d,
                            n_neighbors=None if self.default_model else self.n_nbrs,
                            save_tree=True,
                        )
                        if not self.default_model
                        else (pacmap.PaCMAP(n_components=self.d, save_tree=True))
                    )
                case _:
                    raise ValueError(
                        "Not an implemented dimensionality reduction techniques"
                    )
        else:  # In this case model not None
            params = self.base_model.get_params()
            new_model = None
            if self.default_model:
                new_model = type(self.base_model)()
            else:
                new_model = type(self.base_model)(**params)
            new_model.set_params(**params)
            self.base_model = new_model

    def reset_model(self) -> None:
        """
        Reset the model with previously passed parameters.
        For example models like trimap.TRIMAP can only have fit transform called once!

        Returns:
        --------
            None
        """
        self.__init_model()

    def fit_transform(self, X: np.array, y: np.array) -> np.array:
        """
        Calls the wrapped model's fit transform method

        Args:
        -----
            X (np.array): N x D dimensional array to fit model with

            y (np.array): unused

        Returns:
        --------
            np.array: transformed data
        """
        return self.base_model.fit_transform(X, None)

    def fit(self, X: np.array, y: np.array = None):
        """
        Call the fit function on wrapped model. Not all models implement this function, if not the case
        value error is thrown

        Args:
        -----
            X (np.array): N x D array used to fit the model

            y (np.array): default = None, usually unused but label array for corresponding X data
        
        Returns:
        --------
            TransformWrapper: fitted transform wrapper
        """
        self.__check_function(self.fit.__name__)
        self.base_model = self.base_model.fit(
            X, None
        )  # .reshape(1, -1) if using pacmap for y apparently
        return self

    def transform(self, X: np.array):
        """
        Call the transform function on wrapped model. Not all models implement this function, if not the case
        value error is thrown

        Args:
        -----
            X (np.array): N x D array of data to be transformed
        
        Returns:
        --------
            X_reduced (np.array): transformed data
        """

        self.__check_function(self.transform.__name__)
        return self.base_model.transform(X)

    def __check_function(self, func: str):
        """
        Checks if self.base_model implements the function passed by argument

        Args:
        _____
            func (str): name of function to check
        
        Returns:
        --------
            None
        """

        checkCondition(
            hasattr(
                self.base_model,
                func) and callable(
                getattr(
                    self.base_model,
                    func)),
            f"This model does not have a {func} function",
        )

    def get_params(self, deep=True):
        if self.method is None:
            return self.base_model.get_params(deep)
        return {
            "method": self.method,
            "n_nbrs": self.n_nbrs,
            "n_outliers": self.n_outliers,
            "d": self.d,
            "default_model": self.default_model,
            # "base_model": self.base_model,
        }

    def set_params(self, **params):
        if self.method is None:
            edited_params = deepcopy(params)
            edited_params.pop("base_model")

            new_base_model = self.base_model.set_params(**edited_params)
            self.base_model = new_base_model
            return self
        for key, value in params.items():
            setattr(self, key, value)
        self.__init_model()
        return self

    @classmethod
    def from_params(cls, **params) -> "TransformWrapper":
        """
        Takes the same arguments as set params, and returns a new instance of type TransformWrapper.
        Factory / Class method.
        
        Args:
        -----
            \**params : name value pairs that correspond to TransformWrapper attribute
        
        Returns:
        --------
            TransformWrapper
        """
        method = params.get("method")
        base_model = params.get("base_model")
        out = cls(base_model=base_model, method=method)
        out.set_params(**params)
        return out
import pytest
from DimRed.transform import TransformWrapper
from sklearn.datasets import load_iris
from sklearn import decomposition

TEST_SIZE = 0.3

@pytest.fixture(scope="module")
def load_data_train():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_fit_transform_all_methods_default_params(load_data_train):
    """
    Test all default fit_transform methods
    """
    X, y = load_data_train

    methods = ["PCA","UMAP","TSNE","TRIMAP","PACMAP"]
    for method in methods:
        model : TransformWrapper = TransformWrapper(method=method,default_model=True)
        Xred = model.fit_transform(X,y)
        assert Xred.shape == (X.shape[0], 2)
    return

def test_fit_and_transform_all_methods_default_params(load_data_train):
    """
    Test fit and transform independantly on methods that work and on methods that dont work
    """
    X,y = load_data_train

    methodWorks =  ["PCA","UMAP","TSNE","PACMAP"]
    methodDoesntWork = ["TRIMAP"]

    for method in methodWorks:
        model : TransformWrapper = TransformWrapper(method=method,default_model=True)
        model.fit(X,y)
        Xred = model.transform(X)
        assert Xred.shape == (X.shape[0], 2)

    for method in methodDoesntWork:
        model : TransformWrapper = TransformWrapper(method=method, default_model=True)
        model.fit(X,y)
        with pytest.raises(ValueError, match="This model does not have a transform function"):
            model.transform(X)
    return


from sklearn import decomposition, manifold # manifold for sklearn TSNE
import umap
import trimap
import pacmap
from openTSNE.sklearn import TSNE


def test_transform_with_pre_existing_model(load_data_train):
    X , y = load_data_train

    models = [decomposition.PCA(2), umap.UMAP(), pacmap.PaCMAP(), TSNE(), trimap.TRIMAP()]

    for model in models:
        wrappedModel : TransformWrapper = TransformWrapper(base_model=model)

        Xred = wrappedModel.fit_transform(X,y)
        assert Xred.shape == (X.shape[0], 2)

        wrappedModel.reset_model()

        Xred2 = wrappedModel.fit_transform(X,y)
        assert Xred2.shape == (X.shape[0], 2)
    return

def test_transform_with_base_model(load_data_train):
    X,y = load_data_train

    model = TransformWrapper(base_model= decomposition.PCA(n_components=2))

    model.fit(X,y)

    model.set_params(base_model = decomposition.PCA(n_components=2),n_components = 3, random_state= 42)

    print(model.get_params())

    model = TransformWrapper(base_model= decomposition.PCA(n_components=2))

    model.fit(X,y)

    model.set_params(base_model = TSNE(), n_components = 2)

    assert True

    return
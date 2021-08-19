from lyncs_io.convert import to_array, from_array
from torch.nn import Conv1d
from torch import Tensor
import numpy as np
import dask.array as da
from pandas import DataFrame
from scipy import sparse
from lyncs_io.utils import layers_are_equal


def test_to_from_array():

    # TODO: [x] sparse matrices
    # TODO: [x] ndarrays
    # TODO: [x] built-ins
    # TODO: [x] dask
    # TODO: [x] torch
    # TODO: [x] Dataframes

    # ??
    # TODO: [ ] keras
    # TODO: [ ] tensorflow

    # Test DataFrames
    df = DataFrame({"A": [1, 2], "B": [3, 4]})
    arr, attrs = to_array(df)
    new_df = from_array(arr, attrs)

    assert (arr == np.array(df)).all()
    assert isinstance(new_df, type(df))
    assert (df.all() == new_df.all()).all()

    # Test sparse matrices
    formats = ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"]

    for f in formats:
        matrix = sparse.random(4, 4, format=f)
        arr, attrs = to_array(matrix)
        new_m = from_array(arr, attrs)

        # TODO:

        assert (arr == matrix.toarray()).all()
        assert isinstance(new_m, type(matrix))
        assert (matrix != new_m).nnz == 0
        assert np.allclose(matrix.A, new_m.A)

        # "For dense arrays >>> np.allclose
        # is a good way of testing equality.
        # And if the sparse arrays aren't too large, that might be good as well"

    # Test ndarrays
    ndarr = np.random.rand(2, 2)
    arr, attrs = to_array(ndarr)
    new_ndarr = from_array(arr, attrs)
    assert (arr == np.array(ndarr)).all()
    assert (ndarr == new_ndarr).all()
    assert isinstance(new_ndarr, type(ndarr))

    # Test dask
    darr = da.random.random((10, 10))
    arr, attrs = to_array(darr)
    new_darr = from_array(arr, attrs)
    assert (arr == np.array(darr)).all()
    assert (darr == new_darr).all()
    assert isinstance(new_darr, type(darr))

    conv1d = Conv1d(4, 4, 3)
    arr, attrs = to_array(conv1d)
    new_conv = from_array(arr, attrs)
    # assert numpy array
    assert isinstance(arr, np.ndarray)
    assert layers_are_equal(conv1d, new_conv)

    tensor = Tensor(4, 4, 3)
    arr, attrs = to_array(tensor)
    new_tens = from_array(arr, attrs)
    # assert numpy array
    assert isinstance(arr, np.ndarray)
    assert (tensor == new_tens).all()

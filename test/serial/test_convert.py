from lyncs_io.convert import to_array, from_array
import numpy as np
from pandas import DataFrame
from scipy import sparse


def test_to_from_array():
    df = DataFrame({"A": [1, 2], "B": [3, 4]})
    arr, attrs = to_array(df)
    new_df = from_array(arr, attrs)

    assert type(df) == type(new_df)
    assert (df.all() == new_df.all()).all()

    csr = sparse.random(4, 4, format="csr")
    csc = sparse.random(4, 4, format="csc")
    cc = sparse.random(4, 4, format="coo")

    sparse_matrices = [csr, csc, cc]

    for m in sparse_matrices:
        arr, attrs = to_array(m)

        new_m = from_array(arr, attrs)

        assert type(m) == type(new_m)
        assert (m != new_m).nnz == 0

        # "For dense arrays >>> np.allclose
        # is a good way of testing equality.
        # And if the sparse arrays aren't too large, that might be good as well"

        assert np.allclose(m.A, new_m.A)

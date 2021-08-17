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
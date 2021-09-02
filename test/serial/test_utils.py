"""
Tests for utils.py
"""

# ignore missing doc-string warnings
# pylint: disable=C0116

import tarfile
import pytest
import numpy as np
from pandas import DataFrame
from scipy import sparse
from lyncs_io.utils import (
    find_file,
    get_depth,
    find_member,
    format_key,
    is_sparse_matrix,
    from_reduced,
    from_state,
)
from lyncs_io.testing import tempdir
from lyncs_io.base import save
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    coo_matrix,
    bsr_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
)


def test_find_file(tempdir):
    open(tempdir + "data.npy", "w")
    assert find_file(tempdir + "data") == tempdir + "data.npy"
    assert find_file(tempdir + "/data") == tempdir + "data.npy"
    assert find_file(tempdir + "data.npy") == tempdir + "data.npy"

    # Now two files starting with data exist
    open(tempdir + "data.h5", "w")
    with pytest.raises(ValueError):
        find_file(tempdir + "data")

    # Not existing file
    with pytest.raises(FileNotFoundError):
        find_file(tempdir + "/d_data")

    # test FileLike objects
    with open(tempdir + "data.npy", "w") as data:
        assert find_file(data) is data


def test_find_member(tempdir):
    arr = None
    path = tempdir + "tarball.tar"
    key = "arr.npy"
    save(arr, path + "/" + key)
    with tarfile.open(path, "r") as tar:
        assert find_member(tar, "arr") == tar.getmember(key)
        assert find_member(tar, key) == tar.getmember(key)

        with pytest.raises(KeyError):
            find_member(tar, "non_existing")

    path = tempdir + "newtarball.tar"
    save(arr, path + "/" + key)
    save(arr, path + "/" + key)
    with tarfile.open(path, "r") as tar:

        with pytest.raises(KeyError):
            find_member(tar, "arr")


def test_format_key():
    key = "bar/bar/bar"
    assert format_key(key) == "bar/bar/bar/"
    assert format_key("") == "/"
    assert format_key("/") == "/"


def test_get_depth():
    path = "home/user/bar/foo.npy"
    key = "/"
    assert get_depth(path, key) == 3

    path = "user//bar//foo.npy"
    key = "user"
    assert get_depth(path, key) == 1

    key = "user/bar/.."
    assert get_depth(path, key) == 1

    key = "user/bar/../././"
    assert get_depth(path, key) == 1

    key = "user/bar/.././../"
    assert get_depth(path, key) == 2

    key = "user/bar/.."
    assert get_depth(path, key) == 1


def test_is_sparse_matrix():
    formats = ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"]

    for f in formats:
        matrix = sparse.random(4, 4, format=f)
        assert is_sparse_matrix(matrix) == True


def test_from_reduced():
    objs = [
        DataFrame({}),
        # np.ndarray,
        sparse.random(1, 1),
    ]

    for obj in objs:
        assert from_reduced(obj.__reduce__()) == True

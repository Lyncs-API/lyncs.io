import sys
import os
import tempfile
import pytest
from pathlib import Path
from lyncs_io.utils import find_file, get_depth
from lyncs_io.testing import tempdir


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

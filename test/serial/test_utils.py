import sys
import os
import tempfile
import pytest
from pathlib import Path
from lyncs_io.utils import find_file
from lyncs_io.testing import tempdir


def test_find_file(tempdir):
    with open(tempdir + "/data.npy", "w") as data:
        assert find_file(tempdir + "/data") == tempdir + "/data.npy"

    with open(tempdir + "/data.npy", "w") as data_npy:
        with open(tempdir + "/data.h5", "w") as data_h5:
            with pytest.raises(ValueError):
                find_file(tempdir + "/data")

    with open(tempdir + "/data.npy", "w") as data_npy:
        with pytest.raises(FileNotFoundError):
            find_file(tempdir + "/d_data")

    # test FileLike objects
    with open(tempdir + "/data.npy", "w") as data:
        assert find_file(data) == tempdir + "/data.npy"

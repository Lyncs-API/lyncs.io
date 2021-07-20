import sys
import os
import tempfile
import pytest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lyncs_io.utils import find_file

def test_find_file():
    with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
        with open(tempdir + "/data.npy", "w") as data:
            assert find_file(tempdir + '/data') == tempdir + '/data.npy'

    with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
        with open(tempdir + "/data.npy", "w") as data_npy:
            with open(tempdir + "/data.h5", "w") as data_h5:
                with pytest.raises(Exception):
                    find_file(tempdir + '/data')

    with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
        with open(tempdir + "/data.npy", "w") as data_npy:
            with pytest.raises(Exception):
                find_file(tempdir + '/d_data')

    # test FileLike objects
    with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
        with open(tempdir + "/data.npy", "w") as data:
            assert find_file(data) == tempdir + '/data.npy'


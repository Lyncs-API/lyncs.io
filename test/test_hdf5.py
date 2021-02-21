from pathlib import Path
import lyncs_io as io
import numpy as np
import tempfile


def test_numpy():
    arr = np.random.rand(100)
    with tempfile.TemporaryDirectory() as tmp:
        ftmp = tmp + "/foo.h5/random"

        io.save(arr, ftmp)
        assert (arr == io.load(ftmp)).all()
        assert (arr == io.load(ftmp, format="hdf5")).all()

        path = Path(ftmp)
        assert (arr == io.load(path.parent)[path.name]).all()

        # Testing default name
        ftmp = tmp + "/foo.h5"
        io.save(arr, ftmp)
        assert (arr == io.load(ftmp)["arr0"]).all()
        io.save(arr * 2, ftmp)
        assert (arr * 2 == io.load(ftmp)["arr1"]).all()

        # Testing Head
        assert list(io.load(ftmp).keys()) == list(io.head(ftmp).keys())
        assert io.load(ftmp)["arr0"].shape == io.head(ftmp)["arr0"]["shape"]

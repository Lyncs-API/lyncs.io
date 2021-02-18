import lyncs_io as io
import numpy as np
import tempfile


def test_numpy():
    arr = np.random.rand(100)
    with tempfile.TemporaryDirectory() as tmp:
        ftmp = tmp + "/foo.npy"
        io.save(arr, ftmp)
        assert (arr == io.load(ftmp)).all()
        assert (arr == io.load(ftmp, format="numpy")).all()

        ftmp = tmp + "/foo.txt"
        io.save(arr, ftmp)
        assert (arr == io.load(ftmp)).all()
        assert (arr == io.load(ftmp, format="ascii")).all()

        ftmp = tmp + "/foo.npz"
        io.save(arr, ftmp)
        assert (arr == io.load(ftmp)["arr_0"]).all()
        assert (arr == io.load(ftmp + "/arr_0", format="numpyz")).all()

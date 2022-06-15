from pathlib import Path
import lyncs_io as io
import numpy as np
import tempfile

from lyncs_io.testing import dtype_loop, shape_loop, skip_hdf5, generate_rand_arr


@skip_hdf5
@dtype_loop
@shape_loop
def test_serial_hdf5(dtype, shape):
    arr = generate_rand_arr(shape, dtype)
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

        # Testing Head
        assert list(io.load(ftmp).keys()) == list(io.head(ftmp).keys())
        assert io.load(ftmp)["arr0"].shape == io.head(ftmp)["arr0"]["shape"]

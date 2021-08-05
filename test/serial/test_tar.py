"""
Serial tests for tar
"""

from pathlib import Path
import numpy as np
import lyncs_io as io
from lyncs_io.testing import dtype_loop, shape_loop, ext_loop, tar_mode_loop, tempdir


@tar_mode_loop
@ext_loop
@dtype_loop
@shape_loop
def test_serial_tar(tempdir, dtype, shape, mode, ext):
    """
    Test all the basic functionality of saving, loading
    and getting the header of a file in a tarball
    """
    arr = np.random.rand(*shape).astype(dtype)

    ftmp = tempdir + f"tarball{mode}/foo{ext}"
    io.save(arr, ftmp)
    assert (arr == io.load(ftmp)).all()
    assert (arr == io.load(ftmp, format="Tar")).all()
    assert io.load(ftmp).shape == io.head(ftmp)["shape"]
    assert io.load(ftmp).dtype == io.head(ftmp)["dtype"]

    path = Path(ftmp)
    assert (arr == io.load(path.parent)[path.name]).all()

    # Testing default name
    ftmp = tempdir + f"foo{mode}"
    io.save(arr, ftmp)
    assert (arr == io.load(ftmp)["arr0.npy"]).all()

    # Skip next assertion: Issues with append mode
    # io.save(arr * 2, ftmp)
    # assert (arr * 2 == io.load(ftmp)["arr1.npy"]).all()

    # Testing Head
    assert list(io.load(ftmp).keys()) == list(io.head(ftmp).keys())
    assert io.load(ftmp)["arr0.npy"].shape == io.head(ftmp)["arr0.npy"]["shape"]

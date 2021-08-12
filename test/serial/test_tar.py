"""
Serial tests for tar
"""

import pytest
from pathlib import Path
import numpy as np
import lyncs_io as io
from lyncs_io.testing import dtype_loop, shape_loop, ext_loop, tar_mode_loop, tempdir, generate_rand_arr


@tar_mode_loop
@ext_loop
@dtype_loop
@shape_loop
def test_serial_tar(tempdir, dtype, shape, mode, ext):
    """
    Test all the basic functionality of saving, loading
    and getting the header of a file in a tarball
    """
    arr = generate_rand_arr(shape, dtype)

    ftmp = tempdir + f"tarball{mode}/foo{ext}"
    io.save(arr, ftmp)

    assert (arr == io.load(ftmp)).all()
    assert (arr == io.load(ftmp, format="Tar")).all()
    assert io.load(ftmp).shape == io.head(ftmp)["shape"]
    assert io.load(ftmp).dtype == io.head(ftmp)["dtype"]

    path = Path(ftmp)
    assert (arr == io.load(path.parent)[path.name]).all()

    # Test that an exception is raised when
    # trying to append to a compressed tarball
    with pytest.raises(ValueError):
        io.save(arr, tempdir + f"tarball.tar.gz/foo{ext}")
        io.save(arr, tempdir + f"tarball.tar.gz/foo{ext}")

    # Testing Head
    ftmp = tempdir + f"tarball{mode}/"
    assert list(io.head(ftmp).keys()) == list(io.load(ftmp).keys())
    assert io.load(ftmp)["foo.npy"].shape == io.head(ftmp)["foo.npy"]["shape"]

    ftmp = tempdir + "map_tar.tar"
    mydict = {
        "random": {
            "arr0.npy": generate_rand_arr(shape, dtype),
            "arr1.npy": generate_rand_arr(shape, dtype),
        },
        "zeros.npy": np.zeros((4, 4, 4, 4)),
    }
    io.save(mydict, ftmp)

    loaded_dict = io.load(ftmp, all_data=True)

    assert (mydict["random"]["arr0.npy"] == loaded_dict["random"]["arr0.npy"]).all()
    assert (mydict["random"]["arr1.npy"] == loaded_dict["random"]["arr1.npy"]).all()
    assert (mydict["zeros.npy"] == loaded_dict["zeros.npy"]).all()

import lyncs_io as io
from lyncs_io.testing import dtype_loop, shape_loop, tempdir
from os import remove
from pathlib import Path
import pytest
import tempfile
import tarfile
import numpy as np


modes = ['', '.bz2', '.gz', '.xz']
#.txt raises NotImplementedError
# .h5: nested archives not supported yet
formats = ['.npy',]# '.txt',]# '.h5']


@dtype_loop
@shape_loop
def test_serial_tar(tempdir, dtype, shape):
    arr = np.random.rand(*shape).astype(dtype)

    ftmp = tempdir + "tarball.tar/foo.npy"
    io.save(arr, ftmp)
    assert (arr == io.load(ftmp)).all()
    assert (arr == io.load(ftmp, format='Tar')).all()
    assert io.load(ftmp).shape == io.head(ftmp)["shape"]
    assert io.load(ftmp).dtype == io.head(ftmp)["dtype"]

    path = Path(ftmp)
    assert (arr == io.load(path.parent)[path.name]).all()

    # Testing default name
    ftmp = tempdir + "foo.tar"
    io.save(arr, ftmp)
    assert (arr == io.load(ftmp)["arr0.npy"]).all()
    io.save(arr * 2, ftmp)
    assert (arr * 2 == io.load(ftmp)["arr1.npy"]).all()

    # Testing Head
    # assert list(io.load(ftmp).keys()) == list(io.head(ftmp).keys())
    # assert io.load(ftmp)["arr0.npy"].shape == io.head(ftmp)["arr0.npy"]["shape"]

    # arr = np.random.rand(10)

    # for mode in modes:
    #     for form in formats:

    #         path = f'{tempdir}temp.tar{mode}/data{form}'
    #         save(arr, path, 'Tar')
    #         f = load(path, 'Tar')
    #         remove(f'{tempdir}temp.tar{mode}')

    #         # remove the tarball; append doesn't support compression
    #         assert np.array_equal(arr, f)

    # save(arr, f'{tempdir}temp.tar{mode}/')

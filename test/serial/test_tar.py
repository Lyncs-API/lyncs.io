from lyncs_io.base import load, save
from lyncs_io.testing import tempdir
from os import remove
import pytest
import tempfile
import tarfile
import numpy as np


def test_save(tempdir):

    modes = ['', '.bz2', '.gz', '.xz']
    formats = ['.npy', '.txt', '.h5']

    f = 'Tar'

    arr = np.random.rand(10, 10)

    for mode in modes:
        for form in formats:

            path = f'{tempdir}temptar.tar{mode}/data{form}'

            save(arr, path, format=f)
            assert np.array_equal(arr, load(path, format=f))

            # remove the tarball: append doesn't support compression
            remove(f'{tempdir}temptar.tar{mode}')


# def test_load_without_compression(tempdir):
#     arr_a = ['a', 'b', 'z']
#     with tarfile.open(tempdir + '/tarball.tar', "w:") as tar:
#         with open(tempdir + "/datafile.txt", "w") as dat:
#             for elt in arr_a:
#                 dat.write(str(elt) + '\n')
#             dat.flush()
#             tar.add(dat.name, arcname='datafile.txt')

#     assert load(tempdir + '/tarball.tar/datafile.txt') == arr_a

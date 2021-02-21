"""
Customizing support for numpy-z
"""

__all__ = [
    "loadz",
    "savez",
]

import numpy
from numpy.lib.npyio import NpzFile
from numpy.lib.format import read_magic, _check_version, _read_array_header
from lyncs_utils import is_keyword
from .archive import split_filename, Data, Loader, Archive


def _get_data(npz, key):
    "Reads the header of a numpy file"

    with npz.zip.open(key + ".npy") as tmp:
        version = read_magic(tmp)
        _check_version(version)
        shape, fortran_order, dtype = _read_array_header(tmp, version)

    return Data(
        {
            "shape": shape,
            "dtype": dtype,
            "_numpy_version": version,
            "_fortran_order": fortran_order,
        }
    )


def loadz(filename, key=None, **kwargs):
    "Numpy-z load function"

    filename, key = split_filename(filename, key)

    loader = Loader(loadz, filename, kwargs=kwargs)

    with numpy.load(filename, **kwargs) as npz:
        assert isinstance(npz, NpzFile), "Broken support for Numpy-z"
        if key:
            return npz[key.lstrip("/")]
        return Archive({key: _get_data(npz, key) for key in npz}, loader=loader)


def savez(data, filename, key=None, compressed=False, **kwargs):
    "Numpy-z save function"

    # TODO: numpy overwrites files. Support to numpy-z should be done through zip format
    filename, key = split_filename(filename, key)

    savez = numpy.savez if not compressed else numpy.savez_compressed

    if key:
        if not is_keyword(key):
            raise ValueError("Numpy-z supports only keys that are a valid keyword")
        return savez(filename, **{key: data}, **kwargs)
    return savez(filename, data, **kwargs)

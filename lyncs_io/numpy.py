"""
Customizing support for numpy-z
"""

__all__ = [
    "head",
    "load",
    "save",
    "loadtxt",
    "savetxt",
    "headz",
    "loadz",
    "savez",
]

from io import UnsupportedOperation, BytesIO
from functools import wraps
import numpy
from numpy.lib.npyio import NpzFile
from numpy.lib.format import (
    read_magic,
    _check_version,
    _read_array_header,
    _write_array_header,
    header_data_from_array_1_0,
)
from lyncs_utils import is_keyword
from .archive import split_filename, Data, Loader, Archive
from .header import Header
from .utils import swap, open_file

from .mpi_io import MpiIO, check_comm
from .dask_io import DaskIO, is_dask_array

loadtxt = numpy.loadtxt
savetxt = swap(numpy.savetxt)


@wraps(numpy.load)
def load(filename, chunks=None, comm=None, **kwargs):
    """
    High level interface function for numpy load.
    Loads a numpy array from file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    filename : str
        Filename of the numpy array to be loaded.
    chunks: list
        How to divide the data domain. This enables the Dask API.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.


    Returns:
    --------
    local_array : list
        Returns a numpy array representing the local elements of the domain.
    """

    if comm is not None and chunks is not None:
        raise ValueError("chunks and comm parameters cannot be both set")

    if chunks is not None:

        metadata = head(filename)
        daskio = DaskIO(filename)

        return daskio.load(
            metadata["shape"],
            metadata["dtype"],
            metadata["_offset"],
            chunks=chunks,
            order="F" if metadata["_fortran_order"] else "C",
        )

    if comm is not None:
        check_comm(comm)

        metadata = head(filename)
        with MpiIO(comm, filename, mode="r") as mpiio:
            return mpiio.load(
                metadata["shape"],
                metadata["dtype"],
                "F" if metadata["_fortran_order"] else "C",
                metadata["_offset"],
            )

    return numpy.load(filename, **kwargs)


@wraps(numpy.save)
def save(array, filename, comm=None, **kwargs):
    """
    High level interface function for numpy save.
    Writes a numpy array to file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    local_array : list
        A numpy array representing the local elements of the domain.
    filename : str
        Filename of the numpy array to be loaded.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.

    """

    if is_dask_array(array):
        daskio = DaskIO(filename)
        header = _get_header_bytes(array, fortran_order=False)
        return daskio.save(array, header=header)

    if comm is not None:
        check_comm(comm)

        with MpiIO(comm, filename, mode="w") as mpiio:
            global_shape, _, _ = mpiio.decomposition.compose(array.shape)
            header = _get_header_bytes(array, shape=global_shape)
            return mpiio.save(array, header=header)

    return numpy.save(filename, array, **kwargs)


def _get_header(array, **header):
    if "shape" not in header:
        header["shape"] = array.shape
    if "fortran_order" not in header:
        header["fortran_order"] = numpy.isfortran(array)
    if "descr" not in header:
        header["descr"] = numpy.lib.format.dtype_to_descr(array.dtype)
    return header


def _get_header_bytes(array, **kwargs):
    stream = BytesIO()
    header = _get_header(array, **kwargs)
    numpy.lib.format._write_array_header(stream, header)
    return stream.getvalue()


def _get_offset(npy):
    try:
        return npy.tell()
    except UnsupportedOperation:
        return None


def _get_head(npy):
    "Returns the header of a numpy file"
    version = read_magic(npy)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(npy, version)

    return Header(
        {
            "shape": shape,
            "dtype": dtype,
            "_offset": _get_offset(npy),
            "_numpy_version": version,
            "_fortran_order": fortran_order,
        }
    )


head = open_file(_get_head)


def _get_headz(npz, key):
    "Reads the header of a numpy file"

    with npz.zip.open(key + ".npy") as npy:
        return _get_head(npy)


def headz(filename, key=None, **kwargs):
    "Numpy-z head function"

    filename, key = split_filename(filename, key)

    with numpy.load(filename, **kwargs) as npz:
        assert isinstance(npz, NpzFile), "Broken support for Numpy-z"
        if key:
            return _get_headz(npz, key.lstrip("/"))
        return Archive({key: _get_headz(npz, key) for key in npz})


def loadz(filename, key=None, **kwargs):
    "Numpy-z load function"

    filename, key = split_filename(filename, key)

    loader = Loader(loadz, filename, kwargs=kwargs)

    with numpy.load(filename, **kwargs) as npz:
        assert isinstance(npz, NpzFile), "Broken support for Numpy-z"
        if key:
            return npz[key.lstrip("/")]
        return Archive({key: Data(_get_headz(npz, key)) for key in npz}, loader=loader)


def savez(data, filename, key=None, compressed=False, **kwargs):
    "Numpy-z save function"

    # TODO: numpy overwrites files. Support to numpy-z should be done through zip format
    filename, key = split_filename(filename, key)

    _savez = numpy.savez if not compressed else numpy.savez_compressed

    if key:
        if not is_keyword(key):
            raise ValueError("Numpy-z supports only keys that are a valid keyword")
        return _savez(filename, **{key: data}, **kwargs)
    return _savez(filename, data, **kwargs)

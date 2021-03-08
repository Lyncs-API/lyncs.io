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

from io import UnsupportedOperation
from functools import wraps
import numpy
from numpy.lib.npyio import NpzFile
from numpy.lib.format import read_magic, _check_version, _read_array_header
from lyncs_utils import is_keyword
from .archive import split_filename, Data, Loader, Archive
from .header import Header
from .utils import swap, open_file

# import .mpi_io

# save = swap(numpy.save)
loadtxt = numpy.loadtxt
savetxt = swap(numpy.savetxt)


@wraps(numpy.load)
def load(filename, chunks=None, comm=None, **kwargs):
    """
    chunks = number of chunks per dir
    comm = cartesian MPI_Comm
    """
    from mpi4py import MPI

    if comm is None or comm.size == 1:
        # serial
        return numpy.load(filename, **kwargs)

    # Open File
    fh = MPI.File.Open(comm, filename, amode=MPI.MODE_RDONLY)

    # Each process reads header
    metadata = head(filename)

    if isinstance(comm, MPI.Cartcomm):
        # decompose data over cartesian communicator
        sizes, subsizes, starts = domain_decomposition_cart(metadata["shape"], comm)
    else:
        # decompose data over 1d for normal communicator
        sizes, subsizes, starts = domain_decomposition_1D(metadata["shape"], comm)

    etype = mpi_io.to_mpi_type(numpy.dtype(metadata["dtype"]).char)

    # construct the filetype, use fixed data-type
    filetype = etype.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
    filetype.Commit()

    # set the file view - skip header
    pos = fh.Get_position() + metadata["_offset"]
    # move file pointer to beginning of array data
    fh.Set_view(pos, etype, filetype, datarep="native")

    # allocate space for local_array to hold data read from file
    local_array = numpy.empty(subsizes, dtype=metadata["dtype"], order="C")

    # collectively read the array from file
    fh.Read_all(local_array)

    # close the file
    fh.Close()

    return local_array


@wraps(numpy.save)
def save(filename, array, comm=None, **kwargs):

    if not chunks:
        return numpy.save(filename, array, **kwargs)

    raise NotImplementedError("chunking not supported yet")


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

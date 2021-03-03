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

from mpi4py import MPI

# save = swap(numpy.save)
loadtxt = numpy.loadtxt
savetxt = swap(numpy.savetxt)


@wraps(numpy.load)
def load(filename, chunks=None, comm=None, **kwargs):
    """
    chunks = number of chunks per dir
    comm = cartesian MPI_Comm
    """
    if comm is None or comm.size == 1:
        return numpy.load(filename, **kwargs)

    # TODO: inspect chunks and determine if data can be distributed on processes

    # Open File
    fh = MPI.File.Open(comm, filename, amode=MPI.MODE_RDONLY)

    # Each process reads header
    metadata = head(filename)

    # decompose data over 1d for normal communicator
    # sizes, subsizes, starts = domain_decomposition_1d(metadata["shape"], comm)
    # decompose data over cartesian communicator
    sizes, subsizes, starts = domain_decomposition_cart(metadata["shape"], comm)

    mpi_type = MPI._typedict[numpy.dtype(metadata["dtype"]).char]

    # construct the filetype, use fixed data-type
    filetype = mpi_type.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_C)
    filetype.Commit()

    # set the file view - skip header
    pos = fh.Get_position() + metadata["_offset"]
    # move file pointer to beginning of array data
    fh.Set_view(pos, mpi_type, filetype, datarep="native")

    # allocate space for local_array to hold data read from file
    local_array = numpy.empty(subsizes, dtype=metadata["dtype"], order="C")

    # collectively read the array from file
    fh.Read_all(local_array)

    # close the file
    fh.Close()

    return local_array


def split_work(load, workers, id):
    """
    Performs 1D decomposition of the domain over columns
    """
    part = int(load / workers)  # uniform distribution
    rem = load - part * workers

    # reverse round robbin assignment of the remaining work
    if id >= (workers - rem):
        part += 1
        low = part * id - (workers - rem)
        hi = part * (1 + id) - (workers - rem)
    else:
        low = part * id
        hi = part * (1 + id)

    return low, hi


def domain_decomposition_1D(domain, comm):

    subsizes = list(domain)
    starts = [0] * len(domain)
    dim = 0

    low, hi = split_work(domain[dim], comm.size, comm.rank)

    sizes = domain
    subsizes[dim] = hi - low
    starts[dim] = low

    return sizes, subsizes, starts


def domain_decomposition_cart(domain, comm):

    subsizes = list(domain)
    starts = [0] * len(domain)

    mpi_dims = MPI.Compute_dims(comm.size, 2)
    coords = comm.Get_coords(comm.rank)
    sizes = domain

    for dim in range(len(domain)):
        low, hi = split_work(domain[dim], mpi_dims[dim], coords[dim])
        subsizes[dim] = hi - low
        starts[dim] = low

    return sizes, subsizes, starts


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

"""
Import this file only if in a testing environment
"""

__all__ = [
    "mark_mpi",
    "shape_loop",
    "dtype_loop",
    "chunksize_loop",
    "lshape_loop",
    "workers_loop",
    "parallel_loop",
    "parallel_format_loop",
]

import re
import tempfile
from itertools import product
from pytest import fixture, mark
import numpy

from lyncs_utils import factors, prod
from .formats import formats
from .base import save
from .mpi_io import _tempdir_MPI, with_mpi
from .dask_io import with_dask
from .openqcd import with_openqcd

mark_mpi = mark.skipif(not with_mpi, reason="mpi not available")
mark_dask = mark.skipif(not with_dask, reason="dask not available")

parallel_format_loop = mark.parametrize(
    "format",
    [
        "numpy",
        "lime",
        "tar",
        "hdf5",
    ],
)

with_hdf5 = formats["hdf5"].error is None
skip_hdf5 = mark.skipif(not with_hdf5, reason="hdf5 not available")
if with_hdf5:
    from .hdf5 import mpi as with_hdf5_mpi
skip_hdf5_mpi = mark.skipif(
    not with_hdf5 or not with_hdf5_mpi, reason="parallel hdf5 not available"
)
if not skip_hdf5_mpi.args[0]:
    parallel_format_loop.args[1].append("hdf5")

skip_openqcd = mark.skipif(not with_openqcd, reason="openqcd not available")

shape_loop = mark.parametrize(
    "shape",
    [
        (10,),
        (10, 10),
        (10, 10, 10),
        (10, 10, 10, 10),
    ],
)

# includes all dtypes as dtype_loop but float16 as it's not supported
dtype_mpi_loop = mark.parametrize(
    "dtype",
    [
        "float32",
        "float64",
        "float128",
        "complex64",
        "complex128",
        "complex256",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ],
)


dtype_loop = mark.parametrize(
    "dtype",
    [
        "float16",
        "float32",
        "float64",
        "float128",
        "complex64",
        "complex128",
        "complex256",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ],
)

chunksize_loop = mark.parametrize(
    "chunksize",
    [3, 5, 6, 10],
)

lshape_loop = mark.parametrize(
    "lshape",
    [
        (6, 4, 2, 2),
        (3, 5, 2, 1),
    ],
)

workers_loop = mark.parametrize(
    "workers",
    [1, 2, 4, 7, 12],
)

tar_mode_loop = mark.parametrize(
    "mode",
    [
        ".tar.gz",
        ".taz",
        ".tgz",
        ".tar.bz2",
        ".tb2",
        ".tbz",
        ".tbz2",
        ".tz2",
        ".tar.xz",
        ".txz",
        ".tar",
    ],
)

ext_loop = mark.parametrize(
    "ext",
    [
        ".npy",
        # TODO: ".txt", ".h5"
    ],
)


def generate_rand_arr(shape, dtype):

    if type(shape) == int:
        shape = (shape,)

    if dtype != "bool":
        bits = re.findall("[0-9]+", dtype)[0]
    fdtype = re.findall("[a-zA-Z]+", dtype)[0]

    if fdtype == "bool":
        arr = numpy.random.rand(*shape)
        return numpy.around(arr).astype("bool8")

    if fdtype == "int":
        high = 2 ** (int(bits) - 1) - 1
        low = -high - 1
        return numpy.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if fdtype == "complex":
        bits = str(round(int(bits) / 2))
        reals = numpy.random.rand(*shape).astype("float" + bits)
        imag = numpy.random.rand(*shape).astype("float" + bits) * 1j
        return reals + imag

    return numpy.random.rand(*shape).astype(dtype)


@fixture(scope="session")
def client():
    """
    Enables dask client session-wise during testing to minimize overheads
    """
    # pylint: disable=C0415
    from dask.distributed import Client

    clt = Client(n_workers=12, threads_per_worker=1)
    yield clt
    clt.shutdown()


def mpi():
    """
    Imports MPI upon request
    """
    # pylint: disable=C0415
    from mpi4py import MPI

    return MPI


@fixture
def tempdir():
    """
    Creates a temporary directory to be used during testing
    """

    tmp = tempfile.TemporaryDirectory()
    path = tmp.__enter__()

    yield path + "/"

    tmp.__exit__(None, None, None)


if with_mpi:
    tempdir_MPI = fixture(_tempdir_MPI)
else:
    tempdir_MPI = tempdir


def write_global_array(comm, filename, lshape, dtype="int64", format="numpy"):
    """
    Writes the global array from a local domain using MPI
    """
    if comm.rank == 0:
        if not comm.is_topo:
            mult = tuple(comm.size if i == 0 else 1 for i in range(len(lshape)))
        else:
            dims = comm.dims
            mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))

        gshape = tuple(a * b for a, b in zip(lshape, mult))
        master_array = numpy.random.rand(*gshape).astype(dtype)
        save(master_array, filename, format=format)
    comm.Barrier()  # make sure file is created and visible by all


def get_comm():
    """
    Get the MPI communicator
    """
    return mpi().COMM_WORLD


def get_cart(procs=None, comm=None):
    """
    Get the MPI cartesian communicators
    """
    if comm is None:
        comm = mpi().COMM_WORLD
    if procs is None:
        procs = [comm.size]
    return comm.Create_cart(dims=procs)


def get_procs_list(comm_size=None, max_size=None, repeat=1):
    """
    Gets a processor list with all the combinations for the cartesian topology
    """
    if comm_size is None:
        comm_size = mpi().COMM_WORLD.size

    facts = {1} | set(factors(comm_size))
    procs = []

    for rep in range(1, repeat + 1):
        procs.append(
            list(
                procs
                for procs in product(facts, repeat=rep)
                if prod(procs) == comm_size
            )
        )
    # flattens the list of processes
    procs = list(j for sub_procs in procs for j in sub_procs)

    if not max_size:
        return procs
    return procs[:max_size]


if with_mpi:
    parallel_loop = mark.parametrize("procs", get_procs_list(repeat=4, max_size=2))
else:
    parallel_loop = mark.skipif(True, reason="MPI not installed")

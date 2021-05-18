"""
Import this file only if in a testing environment
"""

__all__ = [
    "mark_mpi",
    "shape_loop",
    "chunksize_loop",
    "lshape_loop",
    "workers_loop",
    "topo_dim_loop",
    "parallel_loop",
]

import os
import tempfile
from itertools import product
from pytest import fixture, mark
import numpy

from lyncs_utils import factors, prod

mark_mpi = mark.mpi(min_size=1)

shape_loop = mark.parametrize(
    "shape",
    [
        (10,),
        (10, 10),
        (10, 10, 10),
        (10, 10, 10, 10),
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

topo_dim_loop = mark.parametrize(
    "topo_dim",
    [1, 2, 3, 4],
)

dtype_loop = mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
    ],
)


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
def tempdir_MPI():
    """
    Creates a temporary directory to be used during testing
    """
    comm = get_comm()
    if comm.rank == 0:
        tmp = tempfile.TemporaryDirectory()
        name = tmp.__enter__()
    else:
        name = ""
    path = comm.bcast(name, root=0)

    # test path exists for all
    has_access = os.path.exists(path) and os.access(path, os.R_OK | os.W_OK)
    all_access = comm.allreduce(has_access, op=mpi().LAND)
    if not all_access:
        raise ValueError(
            "Some processes are unable to access the temporary directory. \n\
                Set TMPDIR, TEMP or TMP environment variables with the temporary \n\
                directory to be used across processes. "
        )

    yield path

    # make sure file exists until everyone is done
    comm.Barrier()
    if comm.rank == 0:
        tmp.__exit__(None, None, None)


@fixture
def tempdir():
    """
    Creates a temporary directory to be used during testing
    """

    tmp = tempfile.TemporaryDirectory()
    path = tmp.__enter__()

    yield path

    tmp.__exit__(None, None, None)


def write_global_array(comm, filename, lshape, dtype="int64", mult=None):
    """
    Writes the global array from a local domain using MPI
    """
    if comm.rank == 0:
        gshape = lshape

        if mult:
            gshape = tuple(a * b for a, b in zip(gshape, mult))

        master_array = numpy.random.rand(*gshape).astype(dtype)
        numpy.save(filename, master_array)
    comm.Barrier()  # make sure file is created and visible by all


def get_comm():
    """
    Get the MPI communicator
    """
    return mpi().COMM_WORLD


def get_cart(procs=None, comm=None):
    """
    Get the MPI cartesian communicator
    """
    if comm is None:
        comm = mpi().COMM_WORLD
    return comm.Create_cart(dims=procs)


def get_topology_dims(comm, ndims):
    """
    Gets the MPI dimensions
    """
    return mpi().Compute_dims(comm.size, ndims)


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


# TODO: Substitute topo_dim_loop with parallel_loop
# such that routines work with arbitrary dimensionality ordering
parallel_loop = mark.parametrize("procs", get_procs_list(repeat=4))

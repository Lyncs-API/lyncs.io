"""
Import this file only if in a testing environment
"""

__all__ = [
    "domain_loop",
    "parallel_loop",
    "mark_mpi",
]

from itertools import product
from pytest import fixture, mark
from lyncs_utils import factors, prod
import numpy


domain_loop = mark.parametrize(
    "domain",
    [
        # (2, 2, 2, 2),
        # (3, 3, 3, 3),
        (4, 4, 4, 4),
        (8, 8, 8, 8),
    ],
)

dtype_loop = mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        # "float16",
    ],
)


def get_procs_list(comm_size=None, max_size=None):
    if comm_size is None:
        from mpi4py import MPI

        comm_size = MPI.COMM_WORLD.size
    facts = {1} | set(factors(comm_size))
    procs = list(
        set(procs for procs in product(facts, repeat=4) if prod(procs) == comm_size)
    )
    if not max_size:
        return procs
    return procs[:max_size]


def get_cart(procs=None, comm=None):
    if not QUDA_MPI or procs is None:
        return None
    if comm is None:
        comm = lib.MPI.COMM_WORLD
    return comm.Create_cart(procs)


parallel_loop = mark.parametrize("procs", get_procs_list(max_size=1))

mark_mpi = mark.mpi(min_size=1)


@fixture
def tempdir():
    import tempfile
    import os
    from mpi4py import MPI

    comm = comm_world()
    if comm.rank == 0:
        tmp = tempfile.TemporaryDirectory()
        name = tmp.__enter__()
    else:
        name = ""
    path = comm.bcast(name, root=0)

    # test path exists for all
    has_access = os.path.exists(path) and os.access(path, os.R_OK | os.W_OK)
    all_access = comm.allreduce(has_access, op=MPI.LAND)
    if not all_access:
        raise ValueError(
            "Some processes are unable to access the temporary directory. \n\
                Set TMPDIR, TEMP or TMP environment variables with the temporary \n\
                directory to be used across processes. "
        )

    yield path
    if comm.rank == 0:
        tmp.__exit__(None, None, None)


def order(header):
    if header["_fortran_order"] is True:
        ordering = "Fortran"
    else:
        ordering = "C"

    return ordering


def write_global_array(comm, filename, *args):

    if comm.rank == 0:
        master_array = numpy.random.rand(*args)
        numpy.save(filename, master_array)
    comm.Barrier()  # make sure file is created and visible by all


def comm_world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def comm_dims(comm, ndims):
    from mpi4py import MPI

    return MPI.Compute_dims(comm.size, ndims)


def hlen():
    return 6


def vlen():
    return 4
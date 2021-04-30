"""
Import this file only if in a testing environment
"""

__all__ = ["mark_mpi", "ldomain_loop", "topo_dim_loop"]

from pytest import fixture, mark
import numpy

mark_mpi = mark.mpi(min_size=1)

ldomain_loop = mark.parametrize(
    "ldomain",
    [
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        (6, 4, 2, 2),
        (3, 5, 2, 1),
    ],
)

topo_dim_loop = mark.parametrize(
    "topo_dim",
    [1, 2, 3, 4],
)


@fixture
def tempdir():
    import tempfile
    import os
    from mpi4py import MPI

    comm = get_comm()
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


def write_global_array(comm, filename, ldomain, mult=None):

    if comm.rank == 0:
        gdomain = ldomain

        if mult:
            gdomain = tuple(a * b for a, b in zip(gdomain, mult))

        master_array = numpy.random.rand(*gdomain)
        numpy.save(filename, master_array)
    comm.Barrier()  # make sure file is created and visible by all


def get_comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def get_cart(procs=None, comm=None):
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.Create_cart(procs)


def get_topology_dims(comm, ndims):
    from mpi4py import MPI

    return MPI.Compute_dims(comm.size, ndims)

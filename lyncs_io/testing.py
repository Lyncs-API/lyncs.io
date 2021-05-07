"""
Import this file only if in a testing environment
"""

__all__ = [
    "mark_mpi",
    "domain_loop",
    "chunksize_loop",
    "ldomain_loop",
    "workers_loop",
    "topo_dim_loop",
    "parallel_loop",
]

from pytest import fixture, mark
import numpy
from itertools import product
from lyncs_utils import factors, prod

mark_mpi = mark.mpi(min_size=1)

domain_loop = mark.parametrize(
    "domain",
    [
        (10,),
        (10, 10),
        (10, 10, 10),
        (10, 10, 10, 10),
    ],
)

# NOTE: currently testing with uniform chunks
chunksize_loop = mark.parametrize(
    "chunksize",
    [3, 4, 5, 6, 10],
)

ldomain_loop = mark.parametrize(
    "ldomain",
    [
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        (6, 4, 2, 2),
        (3, 5, 2, 1),
    ],
)

workers_loop = mark.parametrize(
    "workers",
    [1, 2, 3, 4, 5, 6, 7, 8],
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
    return comm.Create_cart(dims=procs)


def get_topology_dims(comm, ndims):
    from mpi4py import MPI

    return MPI.Compute_dims(comm.size, ndims)


def get_procs_list(comm_size=None, max_size=None, repeat=1):
    from mpi4py import MPI

    if comm_size is None:
        comm_size = MPI.COMM_WORLD.size

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


# TODO: Substitute topo_dim_loop with parallel_loop such that routines work with arbitrary dimensionality ordering
parallel_loop = mark.parametrize("procs", get_procs_list(repeat=4))

"""
Import this file only if in a testing environment
"""

__all__ = ["mark_mpi"]

from pytest import fixture, mark
import numpy

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
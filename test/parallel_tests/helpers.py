import numpy


def get_tmpdir(comm):
    """
    Trying something like:
        with tempfile.TemporaryDirectory() as tmp:
            with io.MpiIO(comm, ftmp, mode="w") as mpiio:
    will result in deadlock as processes will try to write collectively
    to files in different directories collectively.
    Hence need to ensure all the processes will have the same tmpdir.
    """
    import tempfile

    if comm.rank == 0:
        tmpdir = tempfile.TemporaryDirectory()
    else:
        tmpdir = None

    return comm.bcast(tmpdir, root=0)


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

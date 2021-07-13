import numpy
import lyncs_io as io

from lyncs_io.testing import (
    mark_mpi,
    tempdir_MPI,
    lshape_loop,
    dtype_loop,
    parallel_loop,
    get_comm,
    get_cart,
    write_global_array,
)


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
def test_MPI_numpy_load_comm(tempdir_MPI, dtype, lshape):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/foo_numpy_mpiio_load_comm.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(lshape)))
    write_global_array(comm, ftmp, lshape, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
def ttest_MPI_numpy_load_cart(tempdir_MPI, dtype, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()
    ftmp = tempdir_MPI + "/foo_numpy_mpiio_load_cart.npy"

    mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))
    write_global_array(comm, ftmp, lshape, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )
    assert (global_array[slices] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
def test_MPI_numpy_save_comm(tempdir_MPI, dtype, lshape):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/foo_numpy_mpiio_save_comm.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(lshape)))
    write_global_array(comm, ftmp, lshape, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    local_array = global_array[slc]

    io.save(local_array, ftmp, comm=comm)

    global_array = numpy.load(ftmp)
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
def test_MPI_numpy_save_cart(tempdir_MPI, dtype, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()
    ftmp = tempdir_MPI + "/foo_numpy_mpiio_save_cart.npy"

    mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))
    write_global_array(comm, ftmp, lshape, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )
    local_array = global_array[slices]

    io.save(local_array, ftmp, comm=comm)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slices]).all()

import numpy
import lyncs_io as io

from lyncs_io.testing import (
    mark_mpi,
    tempdir_MPI,
    lshape_loop,
    dtype_loop,
    parallel_loop,
    parallel_format_loop,
    get_comm,
    get_cart,
    write_global_array,
)


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_format_loop
def test_MPI_load_comm(tempdir_MPI, dtype, lshape, format):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/mpiio_load_comm"
    if format == "hdf5":
        ftmp += ".h5/data"

    write_global_array(comm, ftmp, lshape, dtype=dtype, format=format)
    global_array = io.load(ftmp, format=format)
    local_array = io.load(ftmp, comm=comm, format=format)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
@parallel_format_loop
def test_MPI_load_cart(tempdir_MPI, dtype, lshape, procs, format):

    comm = get_cart(procs=procs)
    rank = comm.rank
    coords = comm.coords
    ftmp = tempdir_MPI + "/mpiio_load_cart"
    if format == "hdf5":
        ftmp += ".h5/data"

    write_global_array(comm, ftmp, lshape, dtype=dtype, format=format)
    global_array = io.load(ftmp, format=format)
    local_array = io.load(ftmp, comm=comm, format=format)

    slices = tuple(
        slice(coord * size, (coord + 1) * size) for coord, size in zip(coords, lshape)
    )
    assert (global_array[slices] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_format_loop
def test_MPI_save_comm(tempdir_MPI, dtype, lshape, format):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/mpiio_save_comm"
    if format == "hdf5":
        ftmp += ".h5/data"

    write_global_array(comm, ftmp, lshape, dtype=dtype, format=format)
    global_array = io.load(ftmp, format=format)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    local_array = global_array[slc]

    io.save(local_array, ftmp, comm=comm, format=format)

    global_array = io.load(ftmp, format=format)
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
@parallel_format_loop
def test_MPI_save_cart(tempdir_MPI, dtype, lshape, procs, format):

    comm = get_cart(procs=procs)
    coords = comm.coords
    ftmp = tempdir_MPI + "/mpiio_save_cart"
    if format == "hdf5":
        ftmp += ".h5/data"

    write_global_array(comm, ftmp, lshape, dtype=dtype, format=format)
    global_array = io.load(ftmp, format=format)

    slices = tuple(
        slice(coord * size, (coord + 1) * size) for coord, size in zip(coords, lshape)
    )
    local_array = global_array[slices]

    io.save(local_array, ftmp, comm=comm, format=format)

    global_array = io.load(ftmp, format=format)
    assert (local_array == global_array[slices]).all()

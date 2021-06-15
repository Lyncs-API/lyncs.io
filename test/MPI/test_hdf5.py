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
)


def construct_global_shape(comm, lshape, mult=None):

    gshape = lshape

    if mult:
        gshape = tuple(a * b for a, b in zip(gshape, mult))

    return gshape


def write_global_dataset(comm, filename, shape, dataset=None, dtype="int64"):
    """
    Writes the global dataset using MPI Master Process
    """
    if comm.rank == 0:
        if dataset is None:
            master_array = numpy.random.rand(*shape).astype(dtype)
        else:
            master_array = dataset
        io.save(master_array, filename)
    comm.Barrier()  # make sure file is created and visible by all


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
def test_hdf5_load_dataset_comm(tempdir_MPI, dtype, lshape):

    comm = get_comm()
    rank = comm.rank

    ftmp = tempdir_MPI + "/test_hdf5_load_comm.h5/random"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(lshape)))
    gshape = construct_global_shape(comm, lshape, mult=mult)
    write_global_dataset(comm, ftmp, gshape, dtype=dtype)
    global_array = io.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()

    # Testing default name
    ftmp = tempdir_MPI + "/test_hdf5_load_comm.h5"
    write_global_dataset(comm, ftmp, gshape, dataset=global_array, dtype=dtype)
    assert (global_array[slc] == io.load(ftmp, comm=comm)["arr0"]).all()
    write_global_dataset(comm, ftmp, gshape, dataset=global_array * 2, dtype=dtype)
    assert (global_array[slc] * 2 == io.load(ftmp, comm=comm)["arr1"]).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
def test_hdf5_load_dataset_cart(tempdir_MPI, dtype, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()
    ftmp = tempdir_MPI + "/test_hdf5_load_cart.h5/random"

    mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))
    gshape = construct_global_shape(comm, lshape, mult=mult)
    write_global_dataset(comm, ftmp, gshape, dtype=dtype)
    global_array = io.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )
    assert (global_array[slices] == local_array).all()

    # Testing default name
    ftmp = tempdir_MPI + "/test_hdf5_load_cart.h5"
    write_global_dataset(comm, ftmp, gshape, dataset=global_array, dtype=dtype)
    assert (global_array[slices] == io.load(ftmp, comm=comm)["arr0"]).all()
    write_global_dataset(comm, ftmp, gshape, dataset=global_array * 2, dtype=dtype)
    assert (global_array[slices] * 2 == io.load(ftmp, comm=comm)["arr1"]).all()
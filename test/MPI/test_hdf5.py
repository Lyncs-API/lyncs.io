from pathlib import Path
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


def construct_global_shape(lshape, mult=None):

    gshape = lshape

    if mult:
        gshape = tuple(a * b for a, b in zip(gshape, mult))

    return gshape


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
def test_hdf5_load_dataset_comm(tempdir_MPI, dtype, lshape):

    comm = get_comm()
    rank = comm.rank

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(lshape)))
    gshape = construct_global_shape(lshape, mult=mult)
    global_array = numpy.random.rand(*gshape).astype(dtype)

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))

    ftmp = tempdir_MPI + "/test_hdf5_load_comm.h5/random"
    io.save(global_array[slc], ftmp, comm=comm)

    assert (global_array[slc] == io.load(ftmp, comm=comm)).all()
    assert (global_array[slc] == io.load(ftmp, format="hdf5", comm=comm)).all()

    path = Path(ftmp)
    assert (global_array[slc] == io.load(path.parent, comm=comm)[path.name]).all()

    # Testing default name
    ftmp = tempdir_MPI + "/foo.h5"
    io.save(global_array[slc], ftmp, comm=comm)
    assert (global_array[slc] == io.load(ftmp, comm=comm)["arr0"]).all()
    io.save(global_array[slc] * 2, ftmp, comm=comm)
    assert (global_array[slc] * 2 == io.load(ftmp, comm=comm)["arr1"]).all()


@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
def test_hdf5_load_dataset_cart(tempdir_MPI, dtype, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()

    mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))
    gshape = construct_global_shape(lshape, mult=mult)
    global_array = numpy.random.rand(*gshape).astype(dtype)

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )

    ftmp = tempdir_MPI + "/test_hdf5_load_cart.h5/random"
    io.save(global_array[slices], ftmp, comm=comm)

    assert (global_array[slices] == io.load(ftmp, comm=comm)).all()
    assert (global_array[slices] == io.load(ftmp, format="hdf5", comm=comm)).all()

    path = Path(ftmp)
    assert (global_array[slices] == io.load(path.parent, comm=comm)[path.name]).all()

    # Testing default name
    ftmp = tempdir_MPI + "/foo.h5"
    io.save(global_array[slices], ftmp, comm=comm)
    assert (global_array[slices] == io.load(ftmp, comm=comm)["arr0"]).all()
    io.save(global_array[slices] * 2, ftmp, comm=comm)
    assert (global_array[slices] * 2 == io.load(ftmp, comm=comm)["arr1"]).all()

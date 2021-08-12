from pathlib import Path
import numpy
from h5py import h5
import lyncs_io as io

from lyncs_io.testing import (
    mark_mpi,
    tempdir_MPI,
    lshape_loop,
    dtype_loop,
    parallel_loop,
    get_comm,
    get_cart,
    skip_hdf5_mpi,
    generate_rand_arr,
)

mpi = h5.get_config().mpi


def construct_global_shape(lshape, mult=None):

    gshape = lshape

    if mult:
        gshape = tuple(a * b for a, b in zip(gshape, mult))

    return gshape


def get_local_array_slice(dims, lshape, dtype, coords):

    mult = tuple(dims[i] if i < len(dims) else 1 for i in range(len(lshape)))
    gshape = construct_global_shape(lshape, mult=mult)
    global_array = generate_rand_arr(gshape, dtype)

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )

    return global_array[slices]


@skip_hdf5_mpi
@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
def test_MPI_hdf5_load_dataset_comm(tempdir_MPI, dtype, lshape):

    comm = get_comm()
    rank = comm.rank

    local_slice = get_local_array_slice([comm.size], lshape, dtype, [rank])

    ftmp = tempdir_MPI + "/test_hdf5_load_comm.h5/random"
    io.save(local_slice, ftmp, comm=comm)

    assert (local_slice == io.load(ftmp, comm=comm)).all()
    assert (local_slice == io.load(ftmp, format="hdf5", comm=comm)).all()

    path = Path(ftmp)
    assert (local_slice == io.load(path.parent, comm=comm)[path.name]).all()

    # Testing default name
    ftmp = tempdir_MPI + "/foo.h5"
    io.save(local_slice, ftmp, comm=comm)
    assert (local_slice == io.load(ftmp, comm=comm)["arr0"]).all()
    io.save(local_slice * 2, ftmp, comm=comm)
    assert (local_slice * 2 == io.load(ftmp, comm=comm)["arr1"]).all()


@skip_hdf5_mpi
@mark_mpi
@dtype_loop
@lshape_loop  # enables local domain
@parallel_loop
def test_MPI_hdf5_load_dataset_cart(tempdir_MPI, dtype, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()

    local_slice = get_local_array_slice(dims, lshape, dtype, coords)

    ftmp = tempdir_MPI + "/test_hdf5_load_cart.h5/random"
    io.save(local_slice, ftmp, comm=comm)

    assert (local_slice == io.load(ftmp, comm=comm)).all()
    assert (local_slice == io.load(ftmp, format="hdf5", comm=comm)).all()

    path = Path(ftmp)
    assert (local_slice == io.load(path.parent, comm=comm)[path.name]).all()

    # Testing default name
    ftmp = tempdir_MPI + "/foo.h5"
    io.save(local_slice, ftmp, comm=comm)
    assert (local_slice == io.load(ftmp, comm=comm)["arr0"]).all()
    io.save(local_slice * 2, ftmp, comm=comm)
    assert (local_slice * 2 == io.load(ftmp, comm=comm)["arr1"]).all()


@skip_hdf5_mpi
@mark_mpi
@parallel_loop
def test_MPI_hdf5_all_cart(tempdir_MPI, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()

    mydict = {
        "random": {
            "arr0": get_local_array_slice(dims, (6, 4, 2, 2), "float32", coords),
            "arr1": get_local_array_slice(dims, (3, 5, 2, 1), "float32", coords),
        },
        "test": get_local_array_slice(dims, (10, 10, 10, 10), "float32", coords),
    }

    ftmp = tempdir_MPI + "/test_hdf5_load_cart.h5"
    io.save(mydict, ftmp, comm=comm)
    loaded_dict = io.load(ftmp, comm=comm, all_data=True)

    assert (mydict["random"]["arr0"] == loaded_dict["random"]["arr0"]).all()
    assert (mydict["random"]["arr1"] == loaded_dict["random"]["arr1"]).all()
    assert (mydict["test"] == loaded_dict["test"]).all()

    # Append at the end of group
    io.save(mydict, ftmp + "/grp", comm=comm)
    loaded_dict = io.load(ftmp, comm=comm, all_data=True)

    assert (mydict["random"]["arr0"] == loaded_dict["grp"]["random"]["arr0"]).all()
    assert (mydict["random"]["arr1"] == loaded_dict["grp"]["random"]["arr1"]).all()
    assert (mydict["test"] == loaded_dict["grp"]["test"]).all()

    # Ensure mapping is appended after the key
    io.save(mydict, ftmp, key="key-grp", comm=comm)
    loaded_dict = io.load(ftmp, comm=comm, all_data=True)

    assert (mydict["random"]["arr0"] == loaded_dict["key-grp"]["random"]["arr0"]).all()
    assert (mydict["random"]["arr1"] == loaded_dict["key-grp"]["random"]["arr1"]).all()
    assert (mydict["test"] == loaded_dict["key-grp"]["test"]).all()

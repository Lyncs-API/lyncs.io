import numpy
import pytest
import lyncs_io as io

from lyncs_io.testing import (
    mark_mpi,
    tempdir_MPI,
    ldomain_loop,
    topo_dim_loop,
    dtype_loop,
    get_comm,
    get_topology_dims,
    write_global_array,
)


@mark_mpi
@dtype_loop
@ldomain_loop  # enables local domain
def test_numpy_mpiio_load_comm(tempdir_MPI, dtype, ldomain):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    slc = tuple(slice(rank * ldomain[i], (rank + 1) * ldomain[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@ldomain_loop  # enables local domain
@topo_dim_loop  # enables topology dimension
def test_numpy_mpiio_load_cart(tempdir_MPI, dtype, ldomain, topo_dim):

    comm = get_comm()
    rank = comm.rank
    dims = get_topology_dims(comm, topo_dim)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(dims[i] if i < topo_dim else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=cartesian2d)

    slices = tuple(
        slice(coords[i] * ldomain[i], (coords[i] + 1) * ldomain[i])
        for i in range(topo_dim)
    )
    assert (global_array[slices] == local_array).all()


@mark_mpi
@dtype_loop
@ldomain_loop  # enables local domain
def test_numpy_mpiio_save_comm(tempdir_MPI, dtype, ldomain):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    slc = tuple(slice(rank * ldomain[i], (rank + 1) * ldomain[i]) for i in range(1))
    local_array = global_array[slc]

    io.save(local_array, ftmp, comm=comm)

    global_array = numpy.load(ftmp)
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_loop
@ldomain_loop  # enables local domain
@topo_dim_loop  # enables topology dimension
def test_numpy_mpiio_save_cart(tempdir_MPI, dtype, ldomain, topo_dim):

    comm = get_comm()
    rank = comm.rank
    dims = get_topology_dims(comm, topo_dim)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(dims[i] if i < topo_dim else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, dtype=dtype, mult=mult)
    global_array = numpy.load(ftmp)

    slices = tuple(
        slice(coords[i] * ldomain[i], (coords[i] + 1) * ldomain[i])
        for i in range(topo_dim)
    )
    local_array = global_array[slices]

    io.save(local_array, ftmp, comm=cartesian2d)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slices]).all()

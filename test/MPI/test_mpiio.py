import numpy
import pytest
from numpy.lib.format import (
    _write_array_header,
    header_data_from_array_1_0,
)

from lyncs_io.mpi_io import MpiIO, Decomposition
from lyncs_io import numpy as np

from lyncs_io.testing import (
    mark_mpi,
    ldomain_loop,
    topo_dim_loop,
    tempdir_MPI,
    write_global_array,
    get_comm,
    get_topology_dims,
)


@mark_mpi
@topo_dim_loop  # enables topology dimension
def test_mpiio_constructor(topo_dim):
    from mpi4py import MPI

    comm = get_comm()
    # check mode is set to "r" by default
    assert MpiIO(comm, "").mode == "r"

    topo = comm.Create_cart(dims=get_topology_dims(comm, topo_dim))
    mpiio = MpiIO(topo, "testfile.npy", mode="w")

    assert isinstance(mpiio.comm, MPI.Cartcomm)
    assert isinstance(mpiio.decomposition, Decomposition)
    assert mpiio.size == comm.size
    assert mpiio.rank == comm.rank
    assert mpiio.filename == "testfile.npy"
    assert mpiio.handler is None
    assert mpiio.mode == "w"


@mark_mpi
def test_mpiio_file_handler(tempdir_MPI):
    from mpi4py import MPI

    comm = get_comm()

    # when file does not exists and we try to read
    with pytest.raises(MPI.Exception):
        ftmp = tempdir_MPI + "foo.npy"
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            pass

    ftmp = tempdir_MPI + "foo.npy"
    with MpiIO(comm, ftmp, mode="w") as mpiio:
        assert mpiio.handler is not None


@mark_mpi
@ldomain_loop  # enables local domain
def test_mpiio_load_from_comm(tempdir_MPI, ldomain):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, mult=mult)
    global_array = numpy.load(ftmp)
    header = np.head(ftmp)

    # test invalid (Fortran) order
    with pytest.raises(NotImplementedError):
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            mpiio.load(header["shape"], header["dtype"], "Fortran", header["_offset"])

    # test invalid dtype
    with pytest.raises(TypeError):
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            mpiio.load(header["shape"], "j", "Fortran", header["_offset"])

    # valid usage
    with MpiIO(comm, ftmp, mode="r") as mpiio:
        local_array = mpiio.load(
            header["shape"], header["dtype"], order(header), header["_offset"]
        )

    slc = tuple(slice(rank * ldomain[i], (rank + 1) * ldomain[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()


@mark_mpi
@ldomain_loop  # enables local domain
@topo_dim_loop  # enables topology dimension
def test_mpiio_load_from_cart(tempdir_MPI, ldomain, topo_dim):

    comm = get_comm()
    rank = comm.rank
    dims = get_topology_dims(comm, topo_dim)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(dims[i] if i < topo_dim else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, mult=mult)
    global_array = numpy.load(ftmp)
    header = np.head(ftmp)

    with MpiIO(cartesian2d, ftmp, mode="r") as mpiio:
        local_array = mpiio.load(
            header["shape"], header["dtype"], order(header), header["_offset"]
        )

    slices = tuple(
        slice(coords[i] * ldomain[i], (coords[i] + 1) * ldomain[i])
        for i in range(topo_dim)
    )
    assert (global_array[slices] == local_array).all()


@mark_mpi
@ldomain_loop  # enables local domain
def test_mpiio_save_from_comm(tempdir_MPI, ldomain):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(comm.size if i == 0 else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, mult=mult)
    global_array = numpy.load(ftmp)

    slc = tuple(slice(rank * ldomain[i], (rank + 1) * ldomain[i]) for i in range(1))
    local_array = global_array[slc]

    with MpiIO(comm, ftmp, mode="w") as mpiio:
        global_shape, _, _ = mpiio.decomposition.compose(local_array.shape)
        assert global_shape == list(global_array.shape)

        if mpiio.rank == 0:
            header = header_data_from_array_1_0(local_array)
            header["shape"] = tuple(global_shape)  # needs to be tuple
            _write_array_header(mpiio.handler, header)

        mpiio.save(local_array)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slc]).all()


@mark_mpi
@ldomain_loop  # enables local domain
@topo_dim_loop  # enables topology dimension
def test_mpiio_save_from_cart(tempdir_MPI, ldomain, topo_dim):

    comm = get_comm()
    rank = comm.rank
    dims = get_topology_dims(comm, topo_dim)

    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)

    ftmp = tempdir_MPI + "foo.npy"

    mult = tuple(dims[i] if i < topo_dim else 1 for i in range(len(ldomain)))
    write_global_array(comm, ftmp, ldomain, mult=mult)
    global_array = numpy.load(ftmp)

    slices = tuple(
        slice(coords[i] * ldomain[i], (coords[i] + 1) * ldomain[i])
        for i in range(topo_dim)
    )
    local_array = global_array[slices]

    with MpiIO(cartesian2d, ftmp, mode="w") as mpiio:
        global_shape, _, _ = mpiio.decomposition.compose(local_array.shape)
        assert global_shape == list(global_array.shape)

        if mpiio.rank == 0:
            header = header_data_from_array_1_0(local_array)
            header["shape"] = tuple(global_shape)  # needs to be tuple
            _write_array_header(mpiio.handler, header)

        mpiio.save(local_array)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slices]).all()


def order(header):
    if header["_fortran_order"] is True:
        ordering = "Fortran"
    else:
        ordering = "C"

    return ordering

import numpy
import pytest
from numpy.lib.format import (
    _write_array_header,
    header_data_from_array_1_0,
)

from lyncs_io import mpi_io as io
from lyncs_io import numpy as np

from helpers import *


@pytest.mark.mpi(min_size=2)
def test_constructor():
    from mpi4py import MPI

    comm = comm_world()
    # check mode is set to "r" by default
    assert io.MpiIO(comm, "").mode == "r"

    topo = comm.Create_cart(dims=comm_dims(comm, 2))
    mpiio = io.MpiIO(topo, "testfile.npy", mode="w")

    assert isinstance(mpiio.comm, MPI.Cartcomm)
    assert isinstance(mpiio.decomposition, io.Decomposition)
    assert mpiio.size == comm.size
    assert mpiio.rank == comm.rank
    assert mpiio.filename == "testfile.npy"
    assert mpiio.handler is None
    assert mpiio.mode == "w"


@pytest.mark.mpi(min_size=2)
def test_file_handles():
    from mpi4py import MPI

    comm = comm_world()
    tmpdir = get_tmpdir(comm)

    # when file does not exists and we try to read
    with pytest.raises(MPI.Exception):
        with tmpdir as tmp:
            ftmp = tmp + "foo.npy"
            with io.MpiIO(comm, ftmp, mode="r") as mpiio:
                # Never reaches here - not sure how to handle that
                assert mpiio.handler is not None

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"
        with io.MpiIO(comm, ftmp, mode="w") as mpiio:
            assert mpiio.handler is not None


@pytest.mark.mpi(min_size=2)
def test_load_from_comm():

    comm = comm_world()
    rank = comm.rank
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
        global_array = numpy.load(ftmp)
        header = np.head(ftmp)

        # test invalid order
        with pytest.raises(ValueError):
            with io.MpiIO(comm, ftmp, mode="r") as mpiio:
                mpiio.load(
                    header["shape"], header["dtype"], "Fortran", header["_offset"]
                )

        # test invalid dtype
        with pytest.raises(TypeError):
            with io.MpiIO(comm, ftmp, mode="r") as mpiio:
                mpiio.load(header["shape"], "j", "Fortran", header["_offset"])

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
        global_array = numpy.load(ftmp)
        header = np.head(ftmp)

        # valid usage
        with io.MpiIO(comm, ftmp, mode="r") as mpiio:
            local_array = mpiio.load(
                header["shape"], header["dtype"], order(header), header["_offset"]
            )

        lbound, ubound = rank * hlen(), (rank + 1) * hlen()
        assert (global_array[lbound:ubound] == local_array).all()


@pytest.mark.mpi(min_size=2)
def test_load_from_cart():

    comm = comm_world()
    rank = comm.rank
    dims = comm_dims(comm, 2)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, hlen() * dims[0], vlen() * dims[1], 2, 2)
        global_array = numpy.load(ftmp)
        header = np.head(ftmp)

        with io.MpiIO(cartesian2d, ftmp, mode="r") as mpiio:
            local_array = mpiio.load(
                header["shape"], header["dtype"], order(header), header["_offset"]
            )

        hlbound, hubound = coords[0] * hlen(), (coords[0] + 1) * hlen()
        vlbound, vubound = coords[1] * vlen(), (coords[1] + 1) * vlen()
        assert (global_array[hlbound:hubound, vlbound:vubound] == local_array).all()


@pytest.mark.mpi(min_size=2)
def test_save_from_comm():

    comm = comm_world()
    rank = comm.rank
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
        global_array = numpy.load(ftmp)

        lbound, ubound = rank * hlen(), (rank + 1) * hlen()
        local_array = global_array[lbound:ubound]

        with io.MpiIO(comm, ftmp, mode="w") as mpiio:
            global_shape, _, _ = mpiio.decomposition.compose(local_array.shape)
            assert global_shape == list(global_array.shape)

            if mpiio.rank == 0:
                header = header_data_from_array_1_0(local_array)
                header["shape"] = tuple(global_shape)  # needs to be tuple
                _write_array_header(mpiio.handler, header)

            mpiio.save(local_array)

        global_array = numpy.load(ftmp)
        assert (local_array == global_array[lbound:ubound]).all()


@pytest.mark.mpi(min_size=2)
def test_save_from_cart():

    comm = comm_world()
    rank = comm.rank
    dims = comm_dims(comm, 2)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, hlen() * dims[0], vlen() * dims[1], 2, 2)
        global_array = numpy.load(ftmp)

        hlbound, hubound = coords[0] * hlen(), (coords[0] + 1) * hlen()
        vlbound, vubound = coords[1] * vlen(), (coords[1] + 1) * vlen()
        local_array = global_array[hlbound:hubound, vlbound:vubound]

        with io.MpiIO(comm, ftmp, mode="w") as mpiio:
            global_shape, _, _ = mpiio.decomposition.compose(local_array.shape)
            assert global_shape == list(global_array.shape)

            if mpiio.rank == 0:
                header = header_data_from_array_1_0(local_array)
                header["shape"] = tuple(global_shape)  # needs to be tuple
                _write_array_header(mpiio.handler, header)

            mpiio.save(local_array)

        global_array = numpy.load(ftmp)
        assert (local_array == global_array[hlbound:hubound, vlbound:vubound]).all()

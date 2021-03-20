import numpy
import pytest
from numpy.lib.format import (
    _write_array_header,
    header_data_from_array_1_0,
)

import lyncs_io as io
from lyncs_io import numpy as np
from helpers import *


@pytest.mark.mpi_skip()
def test_numpy_serial_load():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ftmp = tmp + "foo.npy"
        array = numpy.random.rand(hlen(), vlen(), 2, 2)
        numpy.save(ftmp, array)

        array1 = io.load(ftmp)
        assert (array == array1).all()


@pytest.mark.mpi_skip()
def ttest_numpy_serial_save():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ftmp = tmp + "foo.npy"
        array = numpy.random.rand(hlen(), vlen(), 2, 2)
        io.save(array, ftmp)

        array1 = numpy.load(ftmp)
        assert (array == array1).all()


@pytest.mark.mpi(min_size=2)
def test_numpy_load_comm():

    comm = comm_world()
    rank = comm.rank
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
        global_array = numpy.load(ftmp)

        local_array = io.load(ftmp, comm=comm)

        lbound, ubound = rank * hlen(), (rank + 1) * hlen()
        assert (global_array[lbound:ubound] == local_array).all()


@pytest.mark.mpi(min_size=2)
def test_numpy_load_cart():

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

        local_array = io.load(ftmp, comm=comm)

        hlbound, hubound = coords[0] * hlen(), (coords[0] + 1) * hlen()
        vlbound, vubound = coords[1] * vlen(), (coords[1] + 1) * vlen()
        assert (global_array[hlbound:hubound, vlbound:vubound] == local_array).all()


@pytest.mark.mpi(min_size=2)
def test_numpy_save_comm():

    comm = comm_world()
    rank = comm.rank
    tmpdir = get_tmpdir(comm)

    with tmpdir as tmp:
        ftmp = tmp + "foo.npy"

        write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
        global_array = numpy.load(ftmp)

        lbound, ubound = rank * hlen(), (rank + 1) * hlen()
        local_array = global_array[lbound:ubound]

        io.save(local_array, ftmp, comm=comm)

        global_array = numpy.load(ftmp)
        assert (global_array[lbound:ubound] == local_array).all()


@pytest.mark.mpi(min_size=2)
def test_numpy_save_cart():

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

        io.save(local_array, ftmp, comm=comm)

        global_array = numpy.load(ftmp)
        assert (local_array == global_array[hlbound:hubound, vlbound:vubound]).all()

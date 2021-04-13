import numpy
import pytest
from numpy.lib.format import (
    _write_array_header,
    header_data_from_array_1_0,
)

import lyncs_io as io
from lyncs_io import numpy as np

from lyncs_io.testing import (
    mark_mpi,
    tempdir,
    comm_world,
    comm_dims,
    hlen,
    vlen,
    order,
    write_global_array,
)


@mark_mpi
def test_numpy_load_comm(tempdir):

    comm = comm_world()
    rank = comm.rank
    ftmp = tempdir + "foo.npy"

    write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=comm)

    lbound, ubound = rank * hlen(), (rank + 1) * hlen()
    assert (global_array[lbound:ubound] == local_array).all()


@mark_mpi
def test_numpy_load_cart(tempdir):

    comm = comm_world()
    rank = comm.rank
    dims = comm_dims(comm, 2)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    ftmp = tempdir + "foo.npy"

    write_global_array(comm, ftmp, hlen() * dims[0], vlen() * dims[1], 2, 2)
    global_array = numpy.load(ftmp)

    local_array = io.load(ftmp, comm=cartesian2d)

    hlbound, hubound = coords[0] * hlen(), (coords[0] + 1) * hlen()
    vlbound, vubound = coords[1] * vlen(), (coords[1] + 1) * vlen()
    assert (global_array[hlbound:hubound, vlbound:vubound] == local_array).all()


@mark_mpi
def test_numpy_save_comm(tempdir):

    comm = comm_world()
    rank = comm.rank
    ftmp = tempdir + "foo.npy"

    write_global_array(comm, ftmp, comm.size * hlen(), vlen(), 2, 2)
    global_array = numpy.load(ftmp)

    lbound, ubound = rank * hlen(), (rank + 1) * hlen()
    local_array = global_array[lbound:ubound]

    io.save(local_array, ftmp, comm=comm)

    global_array = numpy.load(ftmp)
    assert (global_array[lbound:ubound] == local_array).all()


@mark_mpi
def test_numpy_save_cart(tempdir):

    comm = comm_world()
    rank = comm.rank
    dims = comm_dims(comm, 2)
    cartesian2d = comm.Create_cart(dims=dims)
    coords = cartesian2d.Get_coords(rank)
    ftmp = tempdir + "foo.npy"

    write_global_array(comm, ftmp, hlen() * dims[0], vlen() * dims[1], 2, 2)
    global_array = numpy.load(ftmp)

    hlbound, hubound = coords[0] * hlen(), (coords[0] + 1) * hlen()
    vlbound, vubound = coords[1] * vlen(), (coords[1] + 1) * vlen()
    local_array = global_array[hlbound:hubound, vlbound:vubound]

    io.save(local_array, ftmp, comm=cartesian2d)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[hlbound:hubound, vlbound:vubound]).all()

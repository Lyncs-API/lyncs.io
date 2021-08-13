import numpy
import pytest
from numpy.lib.format import (
    _write_array_header,
    header_data_from_array_1_0,
)

from lyncs_io.mpi_io import MpiIO, Decomposition
from lyncs_io import numpy as np
from lyncs_io.convert import to_array
from lyncs_io.numpy import _get_header_bytes

from lyncs_io.testing import (
    mark_mpi,
    lshape_loop,
    parallel_loop,
    dtype_mpi_loop,
    tempdir_MPI,
    write_global_array,
    get_comm,
    get_cart,
)


@dtype_mpi_loop
def test_dtype_to_mpi(tempdir_MPI, dtype_mpi):
    # float16 is currently not in the list
    if dtype_mpi == "float16":
        return

    comm = get_comm()
    ftmp = tempdir_MPI + "/foo.npy"
    mpi = MpiIO(comm, ftmp)
    _dict = mpi.MPI._typedict
    objs = _dict.values()

    assert mpi._dtype_to_mpi(dtype_mpi) in objs


def test_float16_not_supported(tempdir_MPI):
    comm = get_comm()
    ftmp = tempdir_MPI + "/foo.npy"
    mpi = MpiIO(comm, ftmp)

    with pytest.raises(TypeError):
        mpi._dtype_to_mpi("float16")


@mark_mpi
def test_MPI_mpiio_constructor():
    from mpi4py import MPI

    comm = get_comm()
    # check mode is set to "r" by default
    assert MpiIO(comm, "").mode == "r"

    topo = get_cart()
    mpiio = MpiIO(topo, "testfile.npy", mode="w")

    assert isinstance(mpiio.comm, MPI.Cartcomm)
    assert isinstance(mpiio.decomposition, Decomposition)
    assert mpiio.size == comm.size
    assert mpiio.rank == comm.rank
    assert mpiio.filename == "testfile.npy"
    assert mpiio.handler is None
    assert mpiio.mode == "w"


@mark_mpi
def test_MPI_mpiio_file_handler(tempdir_MPI):
    from mpi4py import MPI

    comm = get_comm()

    # when file does not exists and we try to read
    with pytest.raises(MPI.Exception):
        ftmp = tempdir_MPI + "/foo_mpiio_file_handler.npy"
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            pass

    ftmp = tempdir_MPI + "/foo_mpiio_file_handler1.npy"
    with MpiIO(comm, ftmp, mode="w") as mpiio:
        assert mpiio.handler is not None


@mark_mpi
@dtype_mpi_loop
@lshape_loop  # enables local domain
def test_MPI_mpiio_load_from_comm(tempdir_MPI, dtype_mpi, lshape):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/foo_mpiio_load_from_comm.npy"

    write_global_array(comm, ftmp, lshape, dtype=dtype_mpi)
    global_array = numpy.load(ftmp)
    assert global_array.dtype.str != dtype_mpi
    header = np.head(ftmp)

    # test invalid (Fortran) order
    with pytest.raises(NotImplementedError):
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            mpiio.load(header["shape"], header["dtype"], "Fortran", header["_offset"])

    # test invalid dtype
    with pytest.raises(TypeError):
        with MpiIO(comm, ftmp, mode="r") as mpiio:
            mpiio.load(header["shape"], "j", header["_offset"])

    # valid usage
    with MpiIO(comm, ftmp, mode="r") as mpiio:
        local_array = mpiio.load(
            header["shape"], header["dtype"], order(header), header["_offset"]
        )
        assert local_array.dtype.str != dtype_mpi

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    assert (global_array[slc] == local_array).all()


@mark_mpi
@dtype_mpi_loop
@lshape_loop
@parallel_loop
def test_MPI_mpiio_load_from_cart(tempdir_MPI, dtype_mpi, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()
    ftmp = tempdir_MPI + "/foo_mpiio_load_from_cart.npy"

    write_global_array(comm, ftmp, lshape, dtype=dtype_mpi)
    global_array = numpy.load(ftmp)
    assert global_array.dtype.str != dtype_mpi
    header = np.head(ftmp)

    with MpiIO(comm, ftmp, mode="r") as mpiio:
        local_array = mpiio.load(
            header["shape"], header["dtype"], order(header), header["_offset"]
        )
        assert local_array.dtype.str != dtype_mpi

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )
    assert (global_array[slices] == local_array).all()


@mark_mpi
@dtype_mpi_loop
@lshape_loop
def test_MPI_mpiio_save_from_comm(tempdir_MPI, dtype_mpi, lshape):

    comm = get_comm()
    rank = comm.rank
    ftmp = tempdir_MPI + "/foo_mpiio_save_from_comm.npy"

    write_global_array(comm, ftmp, lshape, dtype=dtype_mpi)
    global_array = numpy.load(ftmp)
    global_array, attrs = to_array(global_array)
    header = _get_header_bytes(attrs)
    assert global_array.dtype.str != dtype_mpi

    slc = tuple(slice(rank * lshape[i], (rank + 1) * lshape[i]) for i in range(1))
    local_array = global_array[slc]

    with MpiIO(comm, ftmp, mode="w") as mpiio:
        mpiio.save(local_array, header=header)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slc]).all()


@mark_mpi
@dtype_mpi_loop
@lshape_loop
@parallel_loop
def test_MPI_mpiio_save_from_cart(tempdir_MPI, dtype_mpi, lshape, procs):

    comm = get_cart(procs=procs)
    dims, _, coords = comm.Get_topo()
    ftmp = tempdir_MPI + "/foo_mpiio_save_from_cart.npy"

    write_global_array(comm, ftmp, lshape, dtype=dtype_mpi)
    global_array = numpy.load(ftmp)
    global_array, attrs = to_array(global_array)
    header = _get_header_bytes(attrs)
    assert global_array.dtype.str != dtype_mpi

    slices = tuple(
        slice(coords[i] * lshape[i], (coords[i] + 1) * lshape[i])
        for i in range(len(dims))
    )
    local_array = global_array[slices]

    with MpiIO(comm, ftmp, mode="w") as mpiio:
        mpiio.save(local_array, header=header)

    global_array = numpy.load(ftmp)
    assert (local_array == global_array[slices]).all()


def order(header):
    if header["fortran_order"] is True:
        ordering = "Fortran"
    else:
        ordering = "C"

    return ordering

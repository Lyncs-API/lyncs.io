import os
import numpy
import dask
import dask.array as da
import pytest

from lyncs_utils import prod
from lyncs_io.convert import to_array
from lyncs_io.dask_io import DaskIO, is_dask_array
from lyncs_io.numpy import _get_header_bytes
import lyncs_io as io

from lyncs_io.testing import (
    client,
    dtype_loop,
    shape_loop,
    workers_loop,
    chunksize_loop,
    tempdir,
    generate_rand_arr,
)


def test_Dask_daskio_abspath(client, tempdir):

    assert os.path.isabs(DaskIO(tempdir + "/foo_daskio_load.npy").filename)
    assert os.path.isabs(DaskIO("./foo_daskio_load.npy").filename)


@dtype_loop
@shape_loop
@chunksize_loop
@workers_loop
def test_Dask_daskio_load(client, tempdir, dtype, shape, chunksize, workers):

    ftmp = tempdir + "/foo_daskio_load.npy"
    x_ref = generate_rand_arr(shape, dtype)
    io.save(x_ref, ftmp)

    daskio = DaskIO(ftmp)
    header = io.numpy.head(ftmp)

    x_lazy_in = daskio.load(
        header["shape"],
        header["dtype"],
        header["_offset"],
        chunks=chunksize,
        order="F" if header["fortran_order"] else "C",
    )

    assert isinstance(x_lazy_in, da.Array)
    assert x_lazy_in.dtype.str != dtype
    assert (x_ref == x_lazy_in.compute(num_workers=workers)).all()


def test_Dask_daskio_write_exceptions(client, tempdir):

    assert not is_dask_array(numpy.zeros(10))
    assert is_dask_array(da.zeros(10))

    with pytest.raises(TypeError):
        DaskIO(tempdir + "/foo_daskio_write_exceptions.npy").save(numpy.zeros(10))


@dtype_loop
@shape_loop
@chunksize_loop
@workers_loop
def test_Dask_daskio_write(client, tempdir, dtype, shape, chunksize, workers):

    ftmp = tempdir + "/foo_daskio_write.npy"

    x_ref = generate_rand_arr(shape, dtype)
    x_ref, attrs = to_array(x_ref)
    header = _get_header_bytes(attrs)
    x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunksize)
    assert x_lazy.dtype.str != dtype

    daskio = DaskIO(ftmp)
    x_lazy_out = daskio.save(x_lazy, header=header)
    assert isinstance(x_lazy_out, da.Array)
    assert x_lazy_out.dtype.str != dtype

    x_out = x_lazy_out.compute(num_workers=workers)
    x_ref_in = io.load(ftmp)

    assert x_out.dtype.str != dtype
    assert (x_ref == x_out).all()
    assert (x_ref == x_ref_in).all()


@dtype_loop
@workers_loop
def test_Dask_daskio_write_update(client, tempdir, dtype, workers):

    ftmp = tempdir + "/foo_daskio_write_update.npy"

    domains = [
        (10,),
        (10, 10, 10),
        (10, 10),
        (10, 10, 10, 10),
    ]
    chunksize = 5

    daskio = DaskIO(ftmp)
    # write in the same file arrays of varying domains
    for domain in domains:
        size = prod(domain)
        # chunks should have the same length as domain
        chunks = tuple(chunksize for a in range(len(domain)))

        x_ref = generate_rand_arr(domain, dtype).reshape(domain)
        x_ref, attrs = to_array(x_ref)
        header = _get_header_bytes(attrs)
        x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunks)

        x_lazy_out = daskio.save(x_lazy, header=header)
        assert isinstance(x_lazy_out, da.Array)
        assert x_lazy_out.dtype.str != dtype

        x_out = x_lazy_out.compute(num_workers=workers)
        x_ref_in = io.load(ftmp)

        assert x_out.dtype.str != dtype
        assert (x_ref == x_out).all()
        assert (x_ref == x_ref_in).all()

        # ensure file size matches the size of the written array
        offset = len(header)
        filesz = offset + size * x_out.itemsize

        assert os.stat(ftmp).st_size == filesz

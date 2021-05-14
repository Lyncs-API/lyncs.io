import os
import numpy
import dask
import dask.array as da
import pytest

from lyncs_utils import prod
from lyncs_io.dask_io import DaskIO, is_dask_array
import lyncs_io as io

from lyncs_io.testing import (
    client,
    dtype_loop,
    shape_loop,
    workers_loop,
    chunksize_loop,
    tempdir,
)


@dtype_loop
@shape_loop
@chunksize_loop
@workers_loop
def test_daskio_load(client, tempdir, dtype, shape, chunksize, workers):

    ftmp = tempdir + "foo.npy"
    x_ref = numpy.random.rand(*shape).astype(dtype)
    io.save(x_ref, ftmp)

    daskio = DaskIO(ftmp)
    header = io.numpy.head(ftmp)

    x_lazy_in = daskio.load(
        header["shape"],
        header["dtype"],
        header["_offset"],
        chunks=chunksize,
        order="F" if header["_fortran_order"] else "C",
    )

    assert isinstance(x_lazy_in, da.Array)
    assert x_lazy_in.dtype.str != dtype
    assert (x_ref == x_lazy_in.compute(num_workers=workers)).all()


def test_daskio_write_exceptions(client, tempdir):

    assert not is_dask_array(numpy.zeros(10))
    assert is_dask_array(da.zeros(10))

    with pytest.raises(TypeError):
        DaskIO(tempdir + "foo.npy").save(numpy.zeros(10))


@dtype_loop
@shape_loop
@chunksize_loop
@workers_loop
def test_daskio_write(client, tempdir, dtype, shape, chunksize, workers):

    ftmp = tempdir + "foo.npy"

    x_ref = numpy.random.rand(*shape).astype(dtype)
    x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunksize)
    assert x_lazy.dtype.str != dtype

    daskio = DaskIO(ftmp)
    x_lazy_out = daskio.save(x_lazy)
    assert isinstance(x_lazy_out, da.Array)
    assert x_lazy_out.dtype.str != dtype

    x_out = x_lazy_out.compute(num_workers=workers)
    x_ref_in = io.load(ftmp)

    assert x_out.dtype.str != dtype
    assert (x_ref == x_out).all()
    assert (x_ref == x_ref_in).all()


@dtype_loop
@workers_loop
def test_daskio_write_update(client, tempdir, dtype, workers):

    ftmp = tempdir + "foo.npy"

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

        x_ref = numpy.random.rand(*domain).astype(dtype).reshape(domain)
        x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunks)

        x_lazy_out = daskio.save(x_lazy)
        assert isinstance(x_lazy_out, da.Array)
        assert x_lazy_out.dtype.str != dtype

        x_out = x_lazy_out.compute(num_workers=workers)
        x_ref_in = io.load(ftmp)

        assert x_out.dtype.str != dtype
        assert (x_ref == x_out).all()
        assert (x_ref == x_ref_in).all()

        # ensure file size matches the size of the written array
        header = io.dask_io._build_header_from_dask_array(x_lazy_out)
        offset = io.dask_io._get_dask_array_header_offset(header)
        filesz = offset + size * x_out.itemsize

        assert os.stat(ftmp).st_size == filesz

import os
import numpy
import dask
import dask.array as da

from lyncs_utils import prod
import lyncs_io as io

from lyncs_io.testing import (
    client,
    dtype_loop,
    domain_loop,
    workers_loop,
    chunksize_loop,
    tempdir,
)


@dtype_loop
@domain_loop
@chunksize_loop
@workers_loop
def test_numpy_daskio_load(client, tempdir, dtype, domain, chunksize, workers):

    ftmp = tempdir + "foo.npy"
    x_ref = numpy.random.rand(*domain).astype(dtype).reshape(domain)

    # chunks should have the same length as domain
    chunks = tuple(chunksize for a in range(len(domain)))
    x_lazy_in = io.load(ftmp, chunks=chunks)

    assert isinstance(x_lazy_in, da.Array)
    assert (x_ref == x_lazy_in.compute(num_workers=workers)).all()
    assert x_lazy_in.dtype.str != dtype


@dtype_loop
@domain_loop
@chunksize_loop
@workers_loop
def test_numpy_daskio_write(client, tempdir, dtype, domain, chunksize, workers):

    ftmp = tempdir + "foo.npy"
    size = prod(domain)
    # chunks should have the same length as domain
    chunks = tuple(chunksize for a in range(len(domain)))

    x_ref = numpy.random.rand(*domain).astype(dtype).reshape(domain)
    x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunks)
    assert x_lazy.dtype.str != dtype

    x_lazy_out = io.save(x_lazy, ftmp)
    assert isinstance(x_lazy_out, da.Array)
    assert x_lazy_out.dtype.str != dtype

    x_out = x_lazy_out.compute(num_workers=workers)
    x_ref_in = io.load(ftmp)

    assert x_out.dtype.str != dtype
    assert (x_ref == x_out).all()
    assert (x_ref == x_ref_in).all()


@dtype_loop
@workers_loop
def test_numpy_daskio_write_update(client, tempdir, dtype, workers):

    ftmp = tempdir + "foo.npy"

    domains = [
        (10,),
        (10, 10, 10),
        (10, 10),
        (10, 10, 10, 10),
    ]
    chunksize = 5

    # write in the same file arrays of varying domains
    for domain in domains:
        size = prod(domain)
        # chunks should have the same length as domain
        chunks = tuple(chunksize for a in range(len(domain)))

        x_ref = numpy.random.rand(*domain).astype(dtype).reshape(domain)
        x_lazy = da.array(x_ref, dtype=dtype).rechunk(chunks=chunks)

        x_lazy_out = io.save(x_lazy, ftmp)
        assert isinstance(x_lazy_out, da.Array)
        assert x_lazy_out.dtype.str != dtype

        x_out = x_lazy_out.compute(num_workers=workers)
        x_ref_in = io.load(ftmp)

        assert x_out.dtype.str != dtype
        assert (x_ref == x_out).all()
        assert (x_ref == x_ref_in).all()

        # ensure file size matches the size of the written array
        header = io.numpy.head(ftmp)
        offset = header["_offset"]
        filesz = offset + size * x_out.itemsize

        assert os.stat(ftmp).st_size == filesz
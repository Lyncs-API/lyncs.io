import os
import numpy
import dask
import dask.array as da

from lyncs_utils import prod
import lyncs_io as io

from lyncs_io.testing import client, domain_loop, workers_loop, chunksize_loop, tempdir


@domain_loop
@chunksize_loop
@workers_loop
def test_numpy_daskio_load(client, tempdir, domain, chunksize, workers):

    ftmp = tempdir + "foo.npy"
    size = prod(domain)

    x_ref = numpy.arange(0, size).reshape(domain)
    x_ref_out = io.save(x_ref, ftmp)

    # chunks should have the same length as domain
    chunks = tuple(chunksize for a in range(len(domain)))
    x_lazy_in = io.load(ftmp, chunks=chunks)

    assert isinstance(x_lazy_in, da.Array)

    assert (x_ref == x_lazy_in.compute(num_workers=workers)).all()


@domain_loop
@chunksize_loop
@workers_loop
def test_numpy_daskio_write(client, tempdir, domain, chunksize, workers):

    ftmp = tempdir + "foo.npy"
    size = prod(domain)
    # chunks should have the same length as domain
    chunks = tuple(chunksize for a in range(len(domain)))

    x_lazy = da.arange(0, size).reshape(domain).rechunk(chunks=chunks)
    x_ref = numpy.arange(0, size).reshape(domain)

    x_lazy_out = io.save(x_lazy, ftmp)
    assert isinstance(x_lazy_out, da.Array)

    x_out = x_lazy_out.compute(num_workers=workers)
    x_ref_in = io.load(ftmp)

    assert (x_ref == x_out).all()
    assert (x_ref == x_ref_in).all()


@workers_loop
def test_numpy_daskio_write_update(client, tempdir, workers):

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

        x_lazy = da.arange(0, size).reshape(domain).rechunk(chunks=chunks)
        x_lazy_out = io.save(x_lazy, ftmp)
        assert isinstance(x_lazy_out, da.Array)

        x_out = x_lazy_out.compute(num_workers=workers)
        x_ref = numpy.arange(0, size).reshape(domain)

        x_ref_in = io.load(ftmp)

        assert (x_ref == x_out).all()
        assert (x_ref == x_ref_in).all()

        # ensure file size matches the size of the written array
        header = io.numpy.head(ftmp)
        offset = header["_offset"]
        filesz = offset + size * x_out.itemsize

        assert os.stat(ftmp).st_size == filesz
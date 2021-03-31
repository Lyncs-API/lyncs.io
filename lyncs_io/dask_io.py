import dask
import dask.array as da
import numpy


def dask_load_array(filename, shape, dtype=None, offset=None, chunks=None):

    # TODO: get absolute path to filename
    array = numpy.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)

    return da.from_array(array, chunks=chunks)

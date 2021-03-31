import dask
import dask.array as da
import numpy


def load_chunk(filename, shape, dtype, offset, sl):
    data = numpy.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)
    return data[sl]


def dask_array(filename, shape, dtype, offset=0, blocksize=5, chunking=None):

    if chunking is not None:
        norm_chunks = da.core.normalize_chunks(chunking, shape=shape, dtype=dtype)
        print("Norm_chunks:: ", norm_chunks)

    # lazy load one chunk at a time from file
    load = dask.delayed(load_chunk)
    # lazy evaluation of all tasks to be computed
    # i.e all the load from files
    chunks = []
    for index in range(0, shape[0], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[0] - index)
        chunk = da.from_delayed(
            load(
                filename,
                shape=shape,
                dtype=dtype,
                offset=offset,
                sl=slice(index, index + chunk_size),
            ),
            shape=(chunk_size,) + shape[1:],
            dtype=dtype,
        )
        chunks.append(chunk)

    # instead of concatenating in one dimension conisder blocking
    return da.concatenate(chunks, axis=0)

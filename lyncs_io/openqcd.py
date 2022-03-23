"Functions for openqcd file format"

from array import array
import numpy
from lyncs_utils import (
    prod,
    open_file,
    read_struct,
    write_struct,
    file_size,
)
from .convert import from_array, to_array
from .lib import lib


@open_file
def head(_fp):
    "Reads the header of a configuration"
    header = read_struct(_fp, "<iiiid")
    return {
        "shape": header[:4] + (4, 3, 3),
        "dtype": "<c16",
        "_offset": 4 * 4 + 8,
        "plaq": header[-1],
    }


def load(filename, chunks=None, comm=None, **kwargs):
    """
    Load function for OpenQCD file format.
    Loads a numpy array from file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    filename : str
        Filename of the numpy array to be loaded.
    chunks: list
        How to divide the data domain. This enables the Dask API.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.
    kwargs: dict
        Additional parameters can be passed to override metadata.
        E.g. shape, dtype, etc.

    Returns:
    --------
    local_array : list
        Returns a numpy array representing the local elements of the domain.
    """

    if comm is not None and chunks is not None:
        raise ValueError("chunks and comm parameters cannot be both set")

    metadata = head(filename)
    shape = metadata["shape"]
    dtype = metadata["dtype"]
    offset = metadata["_offset"]

    if chunks is not None:
        raise NotImplementedError("Missing reordering")
        daskio = DaskIO(filename)

        return daskio.load(
            shape, dtype, offset, chunks=chunks, order=order, metadata=metadata
        )

    if comm is not None:
        raise NotImplementedError("Missing reordering")
        check_comm(comm)

        with MpiIO(comm, filename, mode="r") as mpiio:
            return from_array(mpiio.load(shape, dtype, order, offset), attrs=metadata)

    def reorder(arr):
        out = numpy.empty_like(arr)
        lib.from_openqcd(out, arr, 4, array("i", shape[:4]))
        return out

    return reorder(
        from_array(
            numpy.fromfile(
                filename, dtype=dtype, count=prod(shape), offset=offset
            ).reshape(shape),
            attrs=metadata,
        )
    )


def save(array, filename, comm=None, metadata=None):
    """
    High level interface function for lime load.
    Loads a numpy array from file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    filename : str
        Filename of the numpy array to be loaded.
    chunks: list
        How to divide the data domain. This enables the Dask API.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.
    metadata: dict
        Additional metadata to write in the header
    """

    raise NotImplementedError("Missing reordering")
    array, attrs = to_array(array)
    if metadata:
        attrs.update(metadata)

    if is_dask_array(array):
        daskio = DaskIO(filename)
        header = get_header_bytes(attrs)
        return daskio.save(array, header=header)

    if comm is not None:
        check_comm(comm)

        with MpiIO(comm, filename, mode="w") as mpiio:
            global_shape, _, _ = mpiio.decomposition.compose(array.shape)
            attrs["shape"] = tuple(global_shape)
            attrs["nbytes"] = prod(global_shape) * attrs["dtype"].itemsize
            header = get_header_bytes(attrs)
            return mpiio.save(array, header=header)

    write_data(filename, array, attrs)

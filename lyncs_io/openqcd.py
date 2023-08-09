"Functions for OpenQcd file format"

from array import array
from lyncs_utils import (
    prod,
    open_file,
    read_struct,
    write_struct,
    file_size,
)
import numpy
import plaquette
from .convert import from_array, to_array
from .lib import lib, with_lib as with_openqcd
from .utils import is_dask_array


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


@open_file(flag="wb")
def write_data(filename, arr, attrs):
    """
    Save function for OpenQCD file Format
    Saves a numpy array to file with appropriate header metadata
    """
    shape = attrs["shape"]
    plaq = attrs["plaquette"]
    out = numpy.empty_like(arr) + 55
    lib.to_openqcd(arr, out, 4, array("i", shape[:4]))
    write_struct(
        filename,
        "<iiiid",
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        plaq,
    )
    out.tofile(filename)


def save(arr, filename, comm=None, metadata=None):
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

    # raise NotImplementedError("Missing reordering")
    arr, attrs = to_array(arr)
    # OpenQcd format saves plaquette i
    attrs.update({"plaquette": plaquette.plaquette(arr)[2]})
    if metadata:
        attrs.update(metadata)
    if is_dask_array(arr):
        raise NotImplementedError("dask IO not implemented for openqcd")
    #    daskio = DaskIO(filename)
    #    header = get_header_bytes(attrs)
    #    return daskio.save(arr, header=header)

    if comm is not None:
        raise NotImplementedError("MPI IO not implemented for openqcd")
    #     check_comm(comm)
    #     with MpiIO(comm, filename, mode="w") as mpiio:
    #         global_shape, _, _ = mpiio.decomposition.compose(arr.shape)
    #         attrs["shape"] = tuple(global_shape)
    #         attrs["nbytes"] = prod(global_shape) * attrs["dtype"].itemsize
    #         header = get_header_bytes(attrs)
    #         return mpiio.save(arr, header=header)
    # arr is in internal rep
    write_data(filename, arr, attrs)

"""
Parallel IO using Dask
"""
from io import BytesIO
from time import sleep
import os
import numpy

from filelock import FileLock, Timeout


def is_dask_array(obj):
    """
    Function for checking if passed object is a dask Array
    """
    try:
        # pylint: disable=C0415
        from dask.array import Array

        return isinstance(obj, Array)
    except ImportError:
        return False


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        """
        Property for importing Dask wherever necessary
        """
        # pylint: disable=C0415
        import dask

        return dask

    def __init__(self, filename):
        # convert to absolute
        self.filename = os.path.abspath(filename)

    def load(self, domain, dtype, header_offset, chunks=None, order="C"):
        """
        Reads the global domain from a file and loads it in a dask array

        Parameters
        ----------
        domain : list
            Global data domain.
        dtype: data-type
            dask/numpy data-type for the array
        order: str
            whether data are stored in row/column
            major ('C', 'F') order in memory
        header_offset: int
            offset in bytes to where the
            data start in the file.

        Returns:
        --------
        array : dask array
            A lazy evaluated array to be computed on demand
        """
        array = numpy.memmap(
            self.filename,
            mode="r",
            shape=domain,
            dtype=dtype,
            offset=header_offset,
            order=order,
        )

        return self.dask.array.from_array(array, chunks=chunks)

    def save(self, array):
        """
        Writes the array in a npy file in parallel using dask

        Parameters
        ----------
        array : dask/numpy array
            The array to be written

        Returns:
        --------
        array : dask array
            A lazy evaluated array to be computed on demand
        """
        if not is_dask_array(array):
            raise TypeError("array should be a Dask Array")

        header = _build_header_from_dask_array(array)

        return self.dask.array.map_blocks(
            _write_blockwise_to_npy,
            array,
            self.filename,
            header,
            array.shape,
            chunks=array.chunks,
            dtype=array.dtype,
        )


def _get_dask_array_header_offset(header):
    stream = BytesIO()
    numpy.lib.format._write_array_header(stream, header)

    return len(stream.getvalue())


def _build_header_from_dask_array(array, order="C"):
    header_dict = {"shape": array.shape}
    header_dict["fortran_order"] = bool(order == "F")
    header_dict["descr"] = numpy.lib.format.dtype_to_descr(array.dtype)

    return header_dict


def _build_header_from_file(filename):

    with open(filename, "rb") as fptr:
        version = numpy.lib.format.read_magic(fptr)
        numpy.lib.format._check_version(version)
        shape, fortran_order, dtype = numpy.lib.format._read_array_header(fptr, version)

        header_dict = {"shape": shape}
        header_dict["fortran_order"] = fortran_order
        header_dict["descr"] = dtype

    return header_dict


def _write_npy_header(filename, header, interval=0.001):

    lock_path = filename + ".lock"
    lock = FileLock(lock_path)

    # wait for lock to be release if is used
    while lock.is_locked:
        sleep(interval)

    # if file does not exist or header is wrong, then we write a new file
    if not os.path.exists(filename) or header != _build_header_from_file(filename):
        try:
            # we use timeout smaller than poll_intervall so only one lock is acquired
            with lock.acquire(timeout=interval / 2, poll_intervall=interval):
                with open(filename, "wb") as fptr:
                    numpy.lib.format._write_array_header(fptr, header)
        except Timeout:
            # restart the function and wait for the writing to be completed
            _write_npy_header(filename, header, interval=interval)


def _write_blockwise_to_npy(array_block, filename, header, shape, block_info=None):
    """
    Performs a lazy blockwise write of a dask array to file.

    Parameters
    ----------
    array_block: list
        Block array.
    filename: str
        Filename where the result will be stored
    header: dict
        Numpy header to be written in the file
    shape: tuple
        Shape of the global array
    block_info: dict
        contains relevant information to the blocks
        and chunks of the array. Determined by dask
        when the function is used in operations like
        `dask.array.map_blocks`.

    Returns:
    --------
    data : slice of the memmap written to the file
    """

    _write_npy_header(filename, header)
    offset = _get_dask_array_header_offset(header)

    data = numpy.memmap(
        filename,
        mode="r+",
        shape=shape,
        dtype=array_block.dtype,
        offset=offset,
    )

    # write array in right memmap slice
    slc = tuple(slice(*loc) for loc in block_info[None]["array-location"])
    data[slc] = array_block

    return data[slc]

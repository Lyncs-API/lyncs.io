"""
Parallel IO using Dask
"""
from time import sleep
import os
import numpy

from filelock import FileLock, Timeout
from lyncs_utils import read, write

# pylint: disable=C0103
try:
    import dask

    with_dask = True
except ImportError:
    with_dask = False

from .convert import from_array
from .utils import is_dask_array


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        """
        Property for importing Dask wherever necessary
        """
        if not with_dask:
            raise ImportError(
                "Dask not available. Consider installing `lyncs_io[dask]`."
            )

        return dask

    def __init__(self, filename):
        # convert to absolute
        self.filename = os.path.abspath(filename)

    def load(self, domain, dtype, offset, chunks=None, order="C", metadata=None):
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
        offset: int
            offset in bytes to where the
            data start in the file.
        metadata: dict
            if given, the array is converted appropriately
            using `from_array`

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
            offset=offset,
            order=order,
        )
        array = self.dask.array.from_array(array, chunks=chunks)

        if metadata:
            array = self.dask.array.map_blocks(from_array, array, metadata=metadata)

        return array

    def save(self, array, header=None, offset=None):
        """
        Writes the array in a binary file in parallel using dask

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

        if offset is None:
            if header:
                offset = len(header)
            else:
                offset = 0

        if header is None:
            header = b""

        return self.dask.array.map_blocks(
            _write_blockwise_to_npy,
            array,
            self.filename,
            header,
            array.shape,
            offset,
            chunks=array.chunks,
            dtype=array.dtype,
        )


def _write_header(filename, header, interval=0.001):

    lock_path = filename + ".lock"
    lock = FileLock(lock_path)

    # wait for lock to be release if is used
    while lock.is_locked:
        sleep(interval)

    # if file does not exist or header is wrong, then we write a new file
    if not os.path.exists(filename) or header != read(filename, len(header)):
        try:
            # we use timeout smaller than poll_intervall so only one lock is acquired
            with lock.acquire(timeout=interval / 2, poll_intervall=interval):
                write(filename, header)
        except Timeout:
            # restart the function and wait for the writing to be completed
            _write_header(filename, header, interval=interval)


def _write_blockwise_to_npy(
    array_block, filename, header, shape, offset, block_info=None
):
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

    _write_header(filename, header)

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

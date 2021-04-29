"""
Parallel IO using Dask
"""
import numpy

from io import BytesIO
from .utils import touch, exists


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        """
        Property for importing Dask wherever necessary
        """
        import dask

        return dask

    def __init__(self, filename):

        self.filename = filename

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
        array = self._memmap(
            self.filename,
            mode="r",
            shape=domain,
            dtype=dtype,
            offset=header_offset,
            order=order,
        )

        return self.dask.array.from_array(array, chunks=chunks)

    def save(self, array, chunks=None):
        """
        Writes the array in a npy file in parallel using dask

        Parameters
        ----------
        array : dask/numpy array
            The array to be written
        chunks : tuple
            shape of chunks to split the array into

        Returns:
        --------
        array : dask array
            A lazy evaluated array to be computed on demand
        """

        if not isinstance(array, self.dask.array.Array):
            array = self.dask.array.from_array(array, chunks=chunks)

        stream = BytesIO()
        header = self._header_data_from_dask_array_1_0(array)
        numpy.lib.format._write_array_header(stream, header)
        offset = len(stream.getvalue())

        return self.dask.array.map_blocks(
            write_blockwise,
            array,
            self.filename,
            header,
            array.shape,
            offset,
            chunks=array.chunks,
            dtype=array.dtype,
        )

    def _memmap(
        self, filename, mode="r", shape=None, dtype=None, offset=None, order=None
    ):
        return numpy.memmap(
            filename, mode=mode, shape=shape, dtype=dtype, offset=offset, order=order
        )

    def _header_data_from_dask_array_1_0(self, array, order="C"):
        header_dict = {"shape": array.shape}
        header_dict["fortran_order"] = bool(order == "F")
        header_dict["descr"] = numpy.lib.format.dtype_to_descr(array.dtype)
        return header_dict


def write_header(filename, header):

    with open(filename, "wb+") as f:
        numpy.lib.format._write_array_header(f, header)


def write_blockwise(array_block, filename, header, shape, offset, block_info=None):

    block_id = block_info[None]["chunk-location"]

    if sum(block_id) == 0:
        write_header(filename, header)
    elif not exists(filename):
        # make sure the file is created
        touch(filename)

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

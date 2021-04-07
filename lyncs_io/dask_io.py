"""
Parallel IO using Dask
"""

from io import BytesIO
import numpy

from .utils import prod


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
        """
        if not isinstance(array, self.dask.array.Array):
            array = self.dask.array.from_array(array, chunks=chunks)

        stream = BytesIO()
        header = self._header_data_from_dask_array_1_0(array)
        numpy.lib.format._write_array_header(stream, header)
        offset = len(stream.getvalue())

        tasks = []
        slices = self.dask.array.core.slices_from_chunks(array.chunks)

        for block in range(prod(array.numblocks)):
            # Only one should write the header
            if block == 0 and header:
                task_header = self.dask.delayed(self._write_header)(
                    self.filename, header
                )
                tasks.append(task_header)

            task = self.dask.delayed(self._write_chunk)(
                self.filename,
                array,
                offset=offset,
                slice_block=slices[block],
            )
            tasks.append(task)

        return tasks

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

    def _write_header(self, filename, header):
        fptr = open(filename, "wb+")
        numpy.lib.format._write_array_header(fptr, header)
        fptr.close()

    def _write_chunk(
        self,
        filename,
        array,
        offset=None,
        order=None,
        slice_block=None,
    ):

        data = numpy.memmap(
            filename,
            mode="r+",
            shape=array.shape,
            dtype=array.dtype,
            offset=offset,
            order=order,
        )
        data[slice_block] = array[slice_block]
        return data
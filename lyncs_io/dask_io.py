import numpy
from io import BytesIO


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        import dask

        return dask

    def __init__(self, filename):

        self.filename = filename

    def load(self, domain, dtype, header_offset, chunks=None, order="C"):

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

        if not isinstance(array, self.dask.array.Array):
            array = self.dask.array.from_array(array, chunks=chunks)

        stream = BytesIO()
        header = self._header_data_from_dask_array_1_0(array)
        numpy.lib.format._write_array_header(stream, header)
        offset = len(stream.getvalue())

        tasks = []
        slices = self.dask.array.core.slices_from_chunks(array.chunks)

        for block in range(self._prod(array.numblocks)):
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
                sl=slices[block],
            )
            tasks.append(task)

        return tasks

    def _prod(self, val):
        res = 1
        for ele in val:
            res *= ele
        return res

    def _memmap(
        self, filename, mode="r", shape=None, dtype=None, offset=None, order=None
    ):
        return numpy.memmap(
            filename, mode=mode, shape=shape, dtype=dtype, offset=offset, order=order
        )

    def _header_data_from_dask_array_1_0(self, array, order="C"):
        d = {"shape": array.shape}
        d["fortran_order"] = True if order == "F" else False
        d["descr"] = numpy.lib.format.dtype_to_descr(array.dtype)
        return d

    def _write_header(self, filename, header):
        fp = open(filename, "wb+")
        numpy.lib.format._write_array_header(fp, header)
        fp.close()

    def _write_chunk(
        self,
        filename,
        array,
        offset=None,
        order=None,
        sl=None,
    ):

        data = numpy.memmap(
            filename,
            mode="r+",
            shape=array.shape,
            dtype=array.dtype,
            offset=offset,
            order=order,
        )
        data[sl] = array[sl]
        return data
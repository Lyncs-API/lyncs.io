import numpy


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        import dask

        return dask

    def __init__(self, chunks, filename, mode="r"):

        self.filename = filename
        self.handler = None
        self.mode = mode

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

        raise NotImplementedError("Writing an array with dask is not implemented yet.")

    def _memmap(
        self, filename, mode="r", shape=None, dtype=None, offset=None, order=None
    ):
        return numpy.memmap(
            filename, mode=mode, shape=shape, dtype=dtype, offset=offset, order=order
        )